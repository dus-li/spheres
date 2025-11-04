// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Dus'li

#include <exception>
#include <stdexcept>

#include "device/randomizer.cuh"
#include "device/sphere.cuh"

#include "types.hxx"

namespace device {

MaterialSet::MaterialSet(size_t count)
    : count(count)
{
	try {
		speculars = make_unique_cuda<float4>(count);
		diffuses  = make_unique_cuda<float4>(count);
		ambients  = make_unique_cuda<float4>(count);
		shininess = make_unique_cuda<float>(count);
	} catch (const std::exception &e) {
		throw;
	}
}

void MaterialSet::randomize()
{
	try {
		rand_fill_float4_1(speculars.get(), count, 0.1f, 1.0f);
		rand_fill_float4_1(diffuses.get(), count, 0.2f, 1.0f);
		rand_fill_float4_1(ambients.get(), count, 0.05f, 0.2f);
		rand_fill_float(shininess.get(), count, 4.0f, 128.0f);
	} catch (const std::exception &e) {
		throw;
	}
}

SphereSet::SphereSet(size_t count)
    : count(count)
{
	try {
		centers   = make_unique_cuda<float4>(count);
		radiuses  = make_unique_cuda<float>(count);
		materials = make_unique_cuda<size_t>(count);
	} catch (const std::exception &e) {
		throw;
	}
}

void SphereSet::randomize(size_t material_count)
{
	const size_t lo = 0;

	try {
		rand_fill_float4_0(centers.get(), count, -0.5f, 0.5f);
		rand_fill_float(radiuses.get(), count, 0.1f, 0.2f);
		rand_fill_size_t(materials.get(), count, lo, material_count);
	} catch (const std::exception &e) {
		throw;
	}
}

Spheres::Spheres(size_t material_count, size_t sphere_count)
    : materials(material_count)
    , spheres(sphere_count)
{
	try {
		mdesc = make_unique_cuda<MaterialSetDescriptor>(1);
		sdesc = make_unique_cuda<SphereSetDescriptor>(1);

		init_mdesc();
		init_sdesc();
	} catch (const std::exception &e) {
		throw;
	}
}

void Spheres::randomize()
{
	materials.randomize();
	spheres.randomize(materials.count);
}

void Spheres::init_mdesc()
{
	MaterialSetDescriptor tmp = { /**/
		.speculars = materials.speculars.get(),
		.diffuses  = materials.diffuses.get(),
		.ambients  = materials.ambients.get(),
		.shininess = materials.shininess.get()
	};

	cudaError_t rv;

	rv = cudaMemcpy(mdesc.get(), &tmp, sizeof(tmp), cudaMemcpyHostToDevice);
	if (rv != cudaSuccess)
		throw std::runtime_error("Failed to copy to device");
}

void Spheres::init_sdesc()
{
	SphereSetDescriptor tmp = { /**/
		.centers   = spheres.centers.get(),
		.radiuses  = spheres.radiuses.get(),
		.materials = spheres.materials.get(),
		.count     = spheres.count
	};

	cudaError_t rv;

	rv = cudaMemcpy(sdesc.get(), &tmp, sizeof(tmp), cudaMemcpyHostToDevice);
	if (rv != cudaSuccess)
		throw std::runtime_error("Failed to copy to device");
}

MaterialSetDescriptor *Spheres::get_mdesc()
{
	return mdesc.get();
}

SphereSetDescriptor *Spheres::get_sdesc()
{
	return sdesc.get();
}

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

__global__ void kernels::render(uchar4 *fb, int width, int height,
    MaterialSetDescriptor *mats, SphereSetDescriptor *spheres, float3 cam)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= width || y >= height)
		return;

	// normalized ray direction
	float  u   = (2.0f * (x + 0.5f) / width - 1.0f) * (float)width / height;
	float  v   = 1.0f - 2.0f * (y + 0.5f) / height;
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

	fb[y * width + x] = make_color((u8)(fminf(color.x, 1.0f) * 255),
	    (u8)(fminf(color.y, 1.0f) * 255),
	    (u8)(fminf(color.z, 1.0f) * 255),
	    255);
}

} // namespace device
