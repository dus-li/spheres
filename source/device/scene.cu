// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Dus'li

#include <cmath>
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
	    lights.get_desc(),
	    cam.location,
	    cam.basis(),
	    cam.fov,
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
    SphereSetDescriptor *spheres, float3 cam, float3 ray)
{
	float  min = FLT_MAX;
	size_t ret = NO_INTERSECTION;
	float3 center;
	float3 min_center;

	for (size_t i = 0; i < spheres->count; ++i) {
		center = f4_xyz(__ldg(&spheres->centers[i]));

		float radius = __ldg(&spheres->radiuses[i]);
		float dist;

		if (sphere_hit(dist, cam, ray, center, radius) && dist < min) {
			min_center = center;
			min        = dist;
			ret        = i;
		}
	}

	p = cam + min * ray;
	n = normalize(p - min_center);
	d = min;

	return ret;
}

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
 * Apart from running the Phong formula this also slightly attenuates light.
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
    MaterialSetDescriptor *mats, SphereSetDescriptor *spheres,
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
	size_t idx = cast(p, n, d, spheres, cpos, ray);

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
