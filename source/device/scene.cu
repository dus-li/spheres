// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Dus'li

#include <exception>
#include <float.h>
#include <stdexcept>

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
    MaterialSetDescriptor *mats, SphereSetDescriptor *spheres,
    LightSetDescriptor *lights, float3 cam, float3 ambient);

} // namespace device::kernels

namespace device {

Scene::Scene(size_t material_count, size_t sphere_count, size_t light_count)
    : lights(light_count)
    , spheres(material_count, sphere_count)
{
	cam     = make_float3(0.0, 0.0, 1.5);
	ambient = make_float3(0.1, 0.1, 0.1);
}

void Scene::reposition_cam(float3 where)
{
	cam = where;
}

void Scene::randomize()
{
	try {
		lights.randomize();
		spheres.randomize();
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
	    lights.get_desc(),
	    cam,
	    ambient);

	cudaError_t tmp = cudaDeviceSynchronize();
	if (tmp != cudaSuccess)
		throw std::runtime_error("Failed to synchronize with device");

	fb.sync();
}

} // namespace device

namespace device::kernels {

static __device__ bool sphere_hit(float &dist, float3 origin, float3 direction,
    float3 center, float radius)
{
	float3 oc = f3_sub(origin, center);

	// Quadratic equation constants
	float a = f3_dot(direction, direction);
	float b = 2 * f3_dot(oc, direction);
	float c = f3_dot(oc, oc) - radius * radius;

	float discriminant = b * b - 4 * a * c;
	if (discriminant < 0)
		return false;

	dist = (-b - sqrtf(discriminant)) / (2 * a);
	return dist > 0.0;
}

static __device__ float3 ray_direction(int x, int y, dim3 dims, float fov)
{
	float aspect = (float)dims.x / (float)dims.y;
	float scale  = tanf(fov / 2);

	// Compute NDCs
	float u = (2 * (x + 0.5f) / dims.x - 1) * aspect * scale;
	float v = (1 - 2 * (y + 0.5f) / dims.y) * scale;

	return f3_norm(make_float3(u, v, -1.0f));
}

#define NO_INTERSECTION ((size_t)(~0))

/**
 * Cast a ray and find first sphere intersection.
 * @param n       Output, surface normal at the intersection point.
 * @param d       Output, distance from the camera.
 * @param spheres Descriptor of the spheres present in the scene.
 * @param cam     Camera position.
 * @param ray     Normalized direction vector of the ray to cast.
 *
 * @return @a NO_INTERSECTION if no sphere is on the way.
 * @return Index of the first intersected sphere.
 */
static __device__ size_t cast(float3 &n, float &d, SphereSetDescriptor *spheres,
    float3 cam, float3 ray)
{
	float  min = FLT_MAX;
	size_t ret = NO_INTERSECTION;

	for (size_t i = 0; i < spheres->count; ++i) {
		float3 center = f4_xyz(spheres->centers[i]);
		float  radius = spheres->radiuses[i];
		float  dist;

		if (sphere_hit(dist, cam, ray, center, radius) && dist < min) {
			min = dist;
			ret = i;
		}
	}

	n = f3_norm(f3_sub(cam, f3_add(cam, f3_sc_mul(min, ray))));
	d = min;

	return ret;
}

#define PI_OVER_2 (1.57f) // roughly

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
static __device__ float3 f3_reflect(float3 incident, float3 n)
{
	return f3_sub(f3_sc_mul(2 * f3_dot(incident, n), n), incident);
}

static __device__ float clamp(float what, float lo, float up)
{
	return fmaxf(lo, fminf(what, up));
}

static __device__ float3 f3_clamp(float3 v, float lo, float up)
{
	return make_float3(clamp(v.x, lo, up),
	    clamp(v.y, lo, up),
	    clamp(v.z, lo, up));
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

	return f3_clamp(make_float3(r, g, b), 0, 1);
}

/**
 * Compute a color of a pixel using Phong equation for point illumination.
 * @param rgb    Ambient light of the scene. Output is placed here as well.
 * @param d      Distance from the camera.
 * @param mat    Index of the material appropriate for this pixel.
 * @param mats   Descriptors of materials used by spheres in the scene.
 * @param lights Descriptors of light sources present in the scene.
 * @param n      Surface normal at this point.
 * @param cam    Camera position.
 *
 * Apart from running the Phong formula this also slightly attenuates light.
 */
static __device__ void compute_color(float3 &rgb, float d, size_t mat,
    MaterialSetDescriptor *mats, LightSetDescriptor *lights, float3 n,
    float3 cam)
{
	float3 ka = f4_xyz(mats->ambients[mat]);
	float3 kd = f4_xyz(mats->diffuses[mat]);
	float3 ks = f4_xyz(mats->speculars[mat]);
	float  a  = mats->shininess[mat];

	// Light attenuation (Not strictly in Phong, but result is prettier.)
	float att = 1.0 / (1 + .2 * d);

	// We are passed the ambient light constant in rgb already.
	rgb = f3_cwise_mul(rgb, ka);

	for (size_t i = 0; i < lights->count; ++i) {
		float3 l  = f3_norm(f4_xyz(lights->locations[i]));
		float3 id = f4_xyz(lights->diffuses[i]);
		float3 is = f4_xyz(lights->speculars[i]);

		// Diffuse term
		float  nl   = f3_dot(l, n);
		float3 diff = f3_cwise_mul(id, f3_sc_mul(nl, kd));

		// Specular term
		float3 r    = f3_reflect(l, n);
		float  rv   = fmaxf(0.f, f3_dot(r, cam));
		float3 spec = f3_cwise_mul(is, f3_sc_mul(powf(rv, a), ks));

		// Compute final (attenuated) color
		rgb = f3_sc_mul(att, f3_add(rgb, f3_add(diff, spec)));
	}
}

// Doxygen doc comment is near the top of the file.
static __global__ void render(uchar4 *fb, dim3 dims,
    MaterialSetDescriptor *mats, SphereSetDescriptor *spheres,
    LightSetDescriptor *lights, float3 cam, float3 ambient)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	// In case window dimensions are a little silly.
	if (x >= dims.x || y >= dims.y)
		return;

	float  d; // Distance of ray intersection point from the camera.
	float3 n; // Surface normal at the ray intersection point.

	// Find first sphere that the ray hits (naively, and suboptimally)
	float3 ray = ray_direction(x, y, dims, PI_OVER_2);
	size_t idx = cast(n, d, spheres, cam, ray);

	// Retrieve index of a material that was hit by the ray.
	size_t mat = material_idx(spheres, idx);

	float3 rgb = ambient;
	if (mat != NO_INTERSECTION)
		compute_color(rgb, d, mat, mats, lights, n, cam);

	// Not strictly in Phong model, but makes the result prettier.
	rgb = f3_tone_map_and_correct(rgb);

	fb[y * dims.x + x] = make_rgba(/**/
	    (u8)(255 * rgb.x),
	    (u8)(255 * rgb.y),
	    (u8)(255 * rgb.z),
	    255);
}

} // namespace device::kernels
