// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Dus'li

#include <algorithm>
#include <exception>
#include <functional>
#include <random>
#include <stdexcept>

#include "device/sphere.cuh"

#include "types.hxx"

template <typename U, typename T>
using builder = std::function<U(std::function<T()>)>;

template <typename U, typename T, typename G>
static void rand__(U *dest, size_t n, T lo, T up, builder<U, T> build)
{
	std::random_device rd;

	std::mt19937 gen(rd());
	cudaError_t  ret;

	G dist(lo, up);

	auto   next = [&dist, &gen]() { return dist(gen); };
	size_t size = sizeof(U) * n;

	std::vector<U> tmp(n);
	std::generate(tmp.begin(), tmp.end(), [&build, &next]() {
		return build(next);
	});

	ret = cudaMemcpy(dest, tmp.data(), size, cudaMemcpyHostToDevice);
	if (ret != cudaSuccess)
		throw std::runtime_error("Failed to copy to device");
}

/**
 * Fill a buffer of compound types with random reals drawn from U[lo, up].
 * @tparam U     Compound type.
 * @tparam T     Single element type.
 * @param  dest  Destination buffer.
 * @param  n     Number of elements in the buffer.
 * @param  lo    Lower bound for the random numbers, inclusive.
 * @param  up    Upper bound for the random numbers, exclusive.
 * @param  build Builder function constructing a compound type instance.
 *
 * @throws std::runtime_error
 * @see randi_
 */
template <typename U, typename T>
static inline void randf_(U *dest, size_t n, T lo, T up, builder<U, T> build)
{
	using D = std::uniform_real_distribution<T>;

	try {
		rand__<U, T, D>(dest, n, lo, up, build);
	} catch (const std::exception &e) {
		throw;
	}
}

/**
 * Fill a buffer of compound types with random integers drawn from U[lo, up].
 * @tparam U     Compound type.
 * @tparam T     Single element type.
 * @param  dest  Destination buffer.
 * @param  n     Number of elements in the buffer.
 * @param  lo    Lower bound for the random numbers, inclusive.
 * @param  up    Upper bound for the random numbers, exclusive.
 * @param  build Builder function constructing a compound type instance.
 *
 * @throws std::runtime_error
 * @see randf_
 *
 * @todo Check @a up value at compile-time.
 */
template <typename U, typename T>
static inline void randi_(U *dest, size_t n, T lo, T up, builder<U, T> build)
{
	using D = std::uniform_int_distribution<T>;

	// Unlike uniform_float_distribution(), the integer one produces values
	// from a closed interval, rather than a one-side open one. To keep
	// interface uniform, we will subtract one, but first its good to check
	// if we can do that :DD
	//
	// Probably better to const and static_assert<> but meh, its not
	// critical, ill improve this later.
	if (up - 1 < lo || up < up - 1)
		throw std::runtime_error("Incorrect upper randi_() bound");

	up -= 1;

	try {
		rand__<U, T, D>(dest, n, lo, up, build);
	} catch (const std::exception &e) {
		throw;
	}
}

static builder<float4, float> f4build0 = [](std::function<float()> next) {
	return make_float4(next(), next(), next(), 0.0);
};

static builder<float4, float> f4build1 = [](std::function<float()> next) {
	return make_float4(next(), next(), next(), 1.0);
};

static builder<float, float> f1build = [](std::function<float()> next) {
	return next();
};

static builder<size_t, size_t> szbuild = [](std::function<size_t()> next) {
	return next();
};

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
		randf_(speculars.get(), count, 0.1f, 1.0f, f4build1);
		randf_(diffuses.get(), count, 0.2f, 1.0f, f4build1);
		randf_(ambients.get(), count, 0.05f, 0.2f, f4build1);
		randf_(shininess.get(), count, 4.0f, 128.0f, f1build);
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
	const size_t mat_lo = 0;

	try {
		randf_(centers.get(), count, -0.5f, 0.5f, f4build0);
		randf_(radiuses.get(), count, 0.1f, 0.2f, f1build);
		randi_(materials.get(), count, mat_lo, material_count, szbuild);
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
