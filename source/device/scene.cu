// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Dus'li

#include <exception>
#include <stdexcept>

#include "device/randomizer.cuh"
#include "device/scene.cuh"

#include "types.hxx"

namespace device {

namespace kernels {

	__global__ void render_old(uchar4 *fb, dim3 dims,
	    MaterialSetDescriptor *mats, SphereSetDescriptor *spheres,
	    float3 cam);

	__global__ void render2(uchar4 *fb, dim3 dims,
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

	kernels::render_old<<<grd, blk>>>(fb.d_fb.get(),
	    fbdim,
	    spheres.get_mdesc(),
	    spheres.get_sdesc(),
	    cam);

	cudaError_t tmp = cudaDeviceSynchronize();
	if (tmp != cudaSuccess)
		throw std::runtime_error("Failed to synchronize with device");

	fb.sync();
}

} // namespace device

namespace device::kernels {

/******************************************************************************/
/* EVERYTHING BELOW THIS LINE IS UGLY, TEMPORARY AND POSSIBLY INCORRECT       */
/******************************************************************************/

#if (__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__)
  #define make_color(r, g, b, a) make_uchar4(a, b, g, r)
#else
  #define make_color(r, g, b, a) make_uchar4(r, g, b, a)
#endif

__device__ float3 f4_xyz(float4 f4)
{
	return make_float3(f4.x, f4.y, f4.z);
}

__device__ float3 f3_sub(float3 a, float3 b)
{
	return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ float f3_dot(float3 a, float3 b)
{
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ float3 f3_norm(float3 v)
{
	float invlen = rsqrtf(f3_dot(v, v));
	return make_float3(v.x * invlen, v.y * invlen, v.z * invlen);
}

__device__ bool hit_sphere(float3 ro, float3 rd, float3 center, float radius,
    float &t)
{
	float3 oc           = f3_sub(ro, center);
	float  a            = f3_dot(rd, rd);
	float  b            = 2.0f * f3_dot(oc, rd);
	float  c            = f3_dot(oc, oc) - radius * radius;
	float  discriminant = b * b - 4 * a * c;

	if (discriminant < 0)
		return false;

	t = (-b - sqrtf(discriminant)) / (2.0f * a);
	return t > 0.0f;
}

__global__ void render_old(uchar4 *fb, dim3 dims, MaterialSetDescriptor *mats,
    SphereSetDescriptor *spheres, float3 cam)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= dims.x || y >= dims.y)
		return;

	// normalized ray direction
	float  u = (2.0f * (x + 0.5f) / dims.x - 1.0f) * (float)dims.x / dims.y;
	float  v = 1.0f - 2.0f * (y + 0.5f) / dims.y;
	float3 ray = f3_norm(make_float3(u, v, -1.0f));

	float tMin   = 1e9;
	int   hitIdx = -1;

	// dont look at this. its not gonna be like that in the final version
	for (int i = 0; i < spheres->count; ++i) {
		float3 center = f4_xyz(spheres->centers[i]);
		float  r      = spheres->radiuses[i];
		float  t;

		if (hit_sphere(cam, ray, center, r, t)) {
			if (t < tMin) {
				tMin   = t;
				hitIdx = i;
			}
		}
	}

	// for testing purposes just use diffuse constant as color.
	float3 color = make_float3(0.0f, 0.0f, 0.0f);
	if (hitIdx >= 0) {
		int    midx = spheres->materials[hitIdx];
		float4 kd   = mats->diffuses[midx];
		color       = f4_xyz(kd);
	}

	fb[y * dims.x + x] = make_color((u8)(fminf(color.x, 1.0f) * 255),
	    (u8)(fminf(color.y, 1.0f) * 255),
	    (u8)(fminf(color.z, 1.0f) * 255),
	    255);
}

__global__ void render2(uchar4 *fb, dim3 dims, MaterialSetDescriptor *mats,
    SphereSetDescriptor *spheres, LightSetDescriptor *lights, float3 cam,
    float3 ambient);

} // namespace kernels
