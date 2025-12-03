// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Dus'li

#include <cmath>
#include <exception>
#include <float.h>
#include <iostream>
#include <stdexcept>

#include "device/octree.cuh"
#include "device/scene.cuh"
#include "device/vecops.cuh"

#include "types.hxx"

namespace device::kernels {

/**
 * Render a scene to a framebuffer using Phong reflection model with a spice.
 * @param fb      Output framebuffer.
 * @param dims    Dimensions of the window/framebuffer.
 * @param mats    Descriptor of materials used by spheres in the scene.
 * @param spheres Descriptor of spheres present in the scene.
 * @param lights  Descriptor of light sources present in the scene.
 * @param cam     Position of the camera.
 * @param ambient Ambient light constant.
 *
 * This generally follows everything that Wikipedia page for Phong reflection
 * model lists. The difference is that it also does some stuff that Wikipedia
 * page for Phong reflection model *doesn't* list. Namely:
 *
 *   - slight light falloff over distance,
 *   - Reinhard tone mapping,
 *   - gamma correction.
 *
 * Justification for all these is simple: output prettier = me happier.
 */
static __global__ void render(uchar4 *fb, dim3 dims,
    MaterialSetDescriptor *mats, SphereSetDescriptor *spheres, FOTDesc *tree,
    LightSetDescriptor *lights, float3 cpos, CamBasis cam, float fov,
    float3 ambient);

} // namespace device::kernels

namespace device {

Camera::Camera(float3 location, float yaw, float pitch, float fov)
    : location(location)
    , yaw(yaw)
    , pitch(pitch)
    , fov(fov)
{
}

void Camera::rotate_x(float rad)
{
	pitch += rad;
}

void Camera::rotate_y(float rad)
{
	yaw += rad;
}

void Camera::move_by(float3 v)
{
	location = location + v;
}

CamBasis Camera::basis()
{
	float3 fwd = make_float3(cosf(pitch) * sinf(yaw),
	    sinf(pitch),
	    -cosf(pitch) * cosf(yaw));

	float3 right = make_float3(sinf(yaw - M_PI_2), 0, -cosf(yaw - M_PI_2));
	float3 up    = cross(right, fwd);

	return std::array<float3, 3> { fwd, right, up };
}

Scene::Scene(size_t material_count, size_t sphere_count, size_t light_count)
    : lights(light_count)
    , spheres(material_count, sphere_count)
    , cam(make_float3(0, 0, 1.5), 0, 0, M_PI / 3)
{
	ambient = make_float3(0.1, 0.1, 0.1);
}

void Scene::randomize()
{
	try {
		lights.randomize();
	} catch (const std::exception &e) {
		throw;
	}
}

void Scene::render_to(Framebuffer &fb)
{
	dim3 fbdim = fb.get_dims();

	dim3 blk(16, 16);
	dim3 grd((fbdim.x + blk.x - 1) / blk.x, (fbdim.y + blk.y - 1) / blk.y);

	kernels::render<<<grd, blk>>>(fb.d_fb.get(),
	    fbdim,
	    spheres.get_mdesc(),
	    spheres.get_sdesc(),
	    spheres.get_tdesc(),
	    lights.get_desc(),
	    cam.location,
	    cam.basis(),
	    cam.fov,
	    ambient);

	cudaError_t tmp = cudaDeviceSynchronize();
	if (tmp != cudaSuccess) {
		std::cerr << cudaGetErrorString(tmp) << std::endl;
		throw std::runtime_error("Failed to synchronize with device");
	}

	fb.sync();
}

} // namespace device

namespace device::kernels {

#define STACK_MAX (64)

/**
 * Check whether a ray passes through an AABB octree node.
 */
static __device__ bool aabb_intersects(const float3 &ro, const float3 &invdir,
    const float3 &blo, const float3 &bup, float max_limit)
{
	float t1, t2;
	float tmin, tmax;

	// X slab
	t1   = (blo.x - ro.x) * invdir.x;
	t2   = (bup.x - ro.x) * invdir.x;
	tmin = fminf(t1, t2);
	tmax = fmaxf(t1, t2);

	// Y slab
	t1   = (blo.y - ro.y) * invdir.y;
	t2   = (bup.y - ro.y) * invdir.y;
	tmin = fmaxf(tmin, fminf(t1, t2));
	tmax = fminf(tmax, fmaxf(t1, t2));

	// Z slab
	t1   = (blo.z - ro.z) * invdir.z;
	t2   = (bup.z - ro.z) * invdir.z;
	tmin = fmaxf(tmin, fminf(t1, t2));
	tmax = fminf(tmax, fmaxf(t1, t2));

	if (tmax < tmin)
		return false;

	if (tmax <= 0)
		return false;

	if ((tmin > 0 ? tmin : 0) >= max_limit)
		return false;

	return true;
}

static __device__ bool sphere_hit(float &dist, float3 origin, float3 direction,
    float3 center, float radius)
{
	float3 oc = origin - center;

	// Quadratic equation constants
	float a = dot(direction, direction);
	float b = 2 * dot(oc, direction);
	float c = dot(oc, oc) - radius * radius;

	float discriminant = b * b - 4 * a * c;
	if (discriminant < 0)
		return false;

	dist = (-b - sqrtf(discriminant)) / (2 * a);
	if (dist > 0.0)
		return true;

	dist = (-b + sqrtf(discriminant)) / (2 * a);
	if (dist > 0.0)
		return true;

	return false;
}

static __device__ float3 ray_direction(int x, int y, dim3 dims, float fov,
    CamBasis cam)
{
	float aspect = (float)dims.x / (float)dims.y;
	float scale  = tanf(fov / 2);

	// Compute NDCs
	float u = 2 * ((x + 0.5f) / dims.x - 0.5f) * aspect * scale;
	float v = 2 * ((y + 0.5f) / dims.y - 0.5f) * scale;

	return normalize(u * cam[1] + v * cam[2] + cam[0]);
}

#define NO_INTERSECTION ((size_t)(~0))

static __device__ void load_aabb(FOTDesc *tree, size_t i, float3 &lo,
    float3 &up)
{
	lo = make_float3( //
	    __ldg(&tree->aabb_lo_xs[i]),
	    __ldg(&tree->aabb_lo_ys[i]),
	    __ldg(&tree->aabb_lo_zs[i]));

	up = make_float3( //
	    __ldg(&tree->aabb_up_xs[i]),
	    __ldg(&tree->aabb_up_ys[i]),
	    __ldg(&tree->aabb_up_zs[i]));
}

static __device__ void leaf_proc(FOTDesc *tree, SphereSetDescriptor *spheres,
    size_t i, float3 cam, float3 ray, float &t_min, size_t &i_min,
    float3 &c_min)
{
	size_t fst = __ldg(&tree->leaf_bases[i]);
	size_t sz  = __ldg(&tree->leaf_sizes[i]);

	for (size_t k = 0; k < sz; ++sz) {
		size_t s = __ldg(&tree->leaf_indices[fst + k]);

		float3 c = f4_xyz(__ldg(&spheres->centers[s]));
		float  r = __ldg(&spheres->radiuses[s]);

		float t;
		if (sphere_hit(t, cam, ray, c, r) && t >= 0.0f && t < t_min) {
			t_min = t;
			i_min = s;
			c_min = c;
		}
	}
}

static inline __device__ void push_single_child(size_t *children, size_t idx,
    size_t stack[STACK_MAX], size_t &sp)
{
	const size_t NONE = ~(size_t)0;

	size_t child = __ldg(&children[idx]);
	if (child != NONE)
		stack[sp++] = child;
}

static __device__ void push_children(FOTDesc *tree, size_t idx,
    size_t stack[STACK_MAX], size_t &sp)
{
	push_single_child(tree->children_0, idx, stack, sp);
	push_single_child(tree->children_1, idx, stack, sp);
	push_single_child(tree->children_2, idx, stack, sp);
	push_single_child(tree->children_3, idx, stack, sp);
	push_single_child(tree->children_4, idx, stack, sp);
	push_single_child(tree->children_5, idx, stack, sp);
	push_single_child(tree->children_6, idx, stack, sp);
	push_single_child(tree->children_7, idx, stack, sp);
}

/**
 * Cast a ray and find first sphere intersection.
 * @param p       Hit point.
 * @param n       Output, surface normal at the intersection point.
 * @param d       Output, distance from the camera.
 * @param spheres Descriptor of the spheres present in the scene.
 * @param cam     Camera position.
 * @param ray     Normalized direction vector of the ray to cast.
 *
 * @return @a NO_INTERSECTION if no sphere is on the way.
 * @return Index of the first intersected sphere.
 */
static __device__ size_t cast(float3 &p, float3 &n, float &d,
    SphereSetDescriptor *spheres, FOTDesc *tree, float3 cam, float3 ray)
{

	float3 invdir = make_float3(1.0f / ray.x, 1.0f / ray.y, 1.0f / ray.z);

	size_t stack[STACK_MAX];
	size_t sp = 0;

	stack[sp++] = 0;

	float  t_min = FLT_MAX;
	size_t i_min = NO_INTERSECTION;
	float3 c_min = make_float3(0, 0, 0);

	while (sp > 0) {
		size_t i = stack[--sp];
		float3 lo, up;

		load_aabb(tree, i, lo, up);

		// Skip if ray misses the node
		if (aabb_intersects(cam, invdir, lo, up, t_min))
			continue;

		if (__ldg(&tree->is_leaf[i])) {
			leaf_proc(tree,
			    spheres,
			    i,
			    cam,
			    ray,
			    t_min,
			    i_min,
			    c_min);
		} else
			push_children(tree, i, stack, sp);
	}

	if (i_min == NO_INTERSECTION)
		return NO_INTERSECTION;

	p = cam + t_min * ray;
	n = normalize(p - c_min);
	d = t_min;

	return i_min;
}
//{
//	float  min = FLT_MAX;
//	size_t ret = NO_INTERSECTION;
//	float3 center;
//	float3 min_center;
//
//	for (size_t i = 0; i < spheres->count; ++i) {
//		center = f4_xyz(__ldg(&spheres->centers[i]));
//
//		float radius = __ldg(&spheres->radiuses[i]);
//		float dist;
//
//		if (sphere_hit(dist, cam, ray, center, radius) && dist < min) {
//			min_center = center;
//			min        = dist;
//			ret        = i;
//		}
//	}
//
//	p = cam + min * ray;
//	n = normalize(p - min_center);
//	d = min;
//
//	return ret;
//}

static __device__ size_t material_idx(SphereSetDescriptor *spheres, size_t idx)
{
	return idx == NO_INTERSECTION ? idx : spheres->materials[idx];
}

/**
 * Compute reflection of an incident on the surface characterized by a normal.
 * @param incident Vector that is to be reflected.
 * @param n        Normal characterizing a reflective surface.
 *
 * @return Reflected vector.
 */
static __device__ float3 reflect(float3 incident, float3 n)
{
	return incident - ((2 * dot(incident, n)) * n);
}

/**
 * Apply Reinhard tone mapping and gamma correct a color.
 * @param rgb Input color.
 *
 * @return Resulting, clamped color.
 */
static __device__ float3 f3_tone_map_and_correct(float3 rgb)
{
	float a = 1.2;

	float r = powf(rgb.x / (1 + rgb.x), a);
	float g = powf(rgb.y / (1 + rgb.y), a);
	float b = powf(rgb.z / (1 + rgb.z), a);

	return clamp(make_float3(r, g, b), 0, 1);
}

/**
 * Compute a color of a pixel using Phong equation for point illumination.
 * @param rgb    Ambient light of the scene. Output is placed here as well.
 * @param p      Hit point.
 * @param d      Distance from the camera.
 * @param mat    Index of the material appropriate for this pixel.
 * @param mats   Descriptors of materials used by spheres in the scene.
 * @param lights Descriptors of light sources present in the scene.
 * @param n      Surface normal at this point.
 * @param cam    Camera position.
 *
 * Apart from runni
ng the Phong formula this also slightly attenuates light.
 */
static __device__ void compute_color(float3 &rgb, float3 p, float d, size_t mat,
    MaterialSetDescriptor *mats, LightSetDescriptor *lights, float3 n,
    float3 cam)
{
	float3 v = normalize(cam - p);

	float3 ka = f4_xyz(mats->ambients[mat]);
	float3 kd = f4_xyz(mats->diffuses[mat]);
	float3 ks = f4_xyz(mats->speculars[mat]);
	float  a  = mats->shininess[mat];

	float3 color = rgb * ka;

	for (size_t i = 0; i < lights->count; ++i) {
		float3 l  = normalize(f4_xyz(lights->locations[i]) - p);
		float3 id = f4_xyz(lights->diffuses[i]);
		float3 is = f4_xyz(lights->speculars[i]);

		float dist = sqrtf(dot(l, l));
		l          = l / dist;

		// Light attenuation (not in Phong, but makes it prettier.)
		float att = 1.0f / (1.0f + 0.1f * dist + 0.02 * dist * dist);

		// Diffuse term
		float  nl   = fmaxf(0.0, dot(l, n));
		float3 diff = id * (nl * kd);

		// Specular term
		float3 r    = reflect(make_float3(0, 0, 0) - l, n);
		float  rv   = fmaxf(0.f, dot(r, v));
		float3 spec = is * (powf(rv, a) * ks);

		// Compute final (attenuated) color
		color = color + att * (diff + spec);
	}

	rgb = color;
}

// Doxygen doc comment is near the top of the file.
static __global__ void render(uchar4 *fb, dim3 dims,
    MaterialSetDescriptor *mats, SphereSetDescriptor *spheres, FOTDesc *tree,
    LightSetDescriptor *lights, float3 cpos, CamBasis cam, float fov,
    float3 ambient)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	// In case window dimensions are a little silly.
	if (x >= dims.x || y >= dims.y)
		return;

	float  d; // Distance of ray intersection point from the camera.
	float3 n; // Surface normal at the ray intersection point.
	float3 p; // Point where ray hits the surface.

	// Find first sphere that the ray hits (naively, and suboptimally)
	float3 ray = ray_direction(x, y, dims, fov, cam);
	size_t idx = cast(p, n, d, spheres, tree, cpos, ray);

	// Retrieve index of a material that was hit by the ray.
	size_t mat = material_idx(spheres, idx);

	float3 rgb = ambient;
	if (mat != NO_INTERSECTION)
		compute_color(rgb, p, d, mat, mats, lights, n, cpos);

	// Not strictly in Phong model, but makes the result prettier.
	rgb = f3_tone_map_and_correct(rgb);

	fb[y * dims.x + x] = make_rgba(/**/
	    (u8)(255 * rgb.x),
	    (u8)(255 * rgb.y),
	    (u8)(255 * rgb.z),
	    255);
}

} // namespace device::kernels
