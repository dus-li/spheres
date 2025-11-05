// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Dus'li

#include <exception>
#include <float.h>
#include <stdexcept>

#include "device/randomizer.cuh"
#include "device/scene.cuh"
#include "device/vecops.cuh"

#include "types.hxx"

namespace device {

namespace kernels {

	__global__ void render(uchar4 *fb, dim3 dims,
	    MaterialSetDescriptor *mats, SphereSetDescriptor *spheres,
	    LightSetDescriptor *lights, float3 cam, float3 ambient);

} // namespace kernels

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

__device__ bool sphere_hit(float &dist, float3 origin, float3 direction,
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

__device__ float3 ray_direction(int x, int y, dim3 dims, float fov)
{
	float aspect = (float)dims.x / (float)dims.y;
	float scale  = tanf(fov / 2);

	// Compute NDCs
	float u = (2 * (x + 0.5f) / dims.x - 1) * aspect * scale;
	float v = (1 - 2 * (y + 0.5f) / dims.y) * scale;

	return f3_norm(make_float3(u, v, -1.0f));
}

#define NO_INTERSECTION ((size_t)(~0))

__device__ size_t stare(float3 &n, SphereSetDescriptor *spheres, float3 cam,
    float3 ray)
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

	n = f3_norm(f3_sub(cam, f3_add(cam, f3_mul(min, ray))));

	return ret;
}

#define PI_OVER_2 (1.57f) // roughly

__device__ size_t material_idx(SphereSetDescriptor *spheres, size_t idx)
{
	return idx == NO_INTERSECTION ? idx : spheres->materials[idx];
}

__device__ float3 f3_reflect(float3 incident, float3 n)
{
	return f3_sub(incident, f3_mul(2 * f3_dot(incident, n), n));
}

__device__ void compute_color(float3 &rgb, size_t mat,
    MaterialSetDescriptor *mats, LightSetDescriptor *lights, float3 n,
    float3 cam)
{
	float3 ka = f4_xyz(mats->ambients[mat]);
	float3 kd = f4_xyz(mats->diffuses[mat]);
	float3 ks = f4_xyz(mats->speculars[mat]);
	float  a  = mats->shininess[mat];

	for (size_t i = 0; i < lights->count; ++i) {
		float3 l  = f3_norm(f4_xyz(lights->locations[i]));
		float3 id = f4_xyz(lights->diffuses[i]);
		float3 is = f4_xyz(lights->speculars[i]);

		// Diffuse term
		float  nl   = f3_dot(n, l);
		float3 diff = f3_mul_cwise(id, f3_mul(nl, kd));

		// Specular term
		float3 r    = f3_reflect(f3_neg(l), n);
		float  rv   = fmaxf(0.f, f3_dot(r, cam));
		float3 spec = f3_mul_cwise(is, f3_mul(powf(rv, a), ks));

		rgb = f3_add(rgb, f3_add(diff, spec));
	}
}

__global__ void render(uchar4 *fb, dim3 dims, MaterialSetDescriptor *mats,
    SphereSetDescriptor *spheres, LightSetDescriptor *lights, float3 cam,
    float3 ambient)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	// In case window dimensions are a little silly.
	if (x >= dims.x || y >= dims.y)
		return;

	// Find first sphere that the ray hits (naively, and suboptimally)
	float3 n;
	float3 ray = ray_direction(x, y, dims, PI_OVER_2);
	size_t idx = stare(n, spheres, cam, ray);
	size_t mat = material_idx(spheres, idx);

	float3 rgb = ambient;
	if (mat != NO_INTERSECTION)
		compute_color(rgb, mat, mats, lights, n, cam);

	fb[y * dims.x + x] = make_rgba(/**/
	    (u8)(255 * fminf(rgb.x, 1.0f)),
	    (u8)(255 * fminf(rgb.y, 1.0f)),
	    (u8)(255 * fminf(rgb.z, 1.0f)),
	    255);
}

} // namespace kernels
