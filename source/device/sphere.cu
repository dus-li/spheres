// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Dus'li

#include <exception>
#include <iostream>
#include <memory>
#include <random>
#include <ranges>
#include <stdexcept>
#include <string>

#include "device/octree.cuh"
#include "device/randomizer.cuh"
#include "device/sphere.cuh"
#include "device/vecops.cuh"

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

static std::array<float4, 3> gen_random_mat_consts(void)
{
	static std::random_device rd;
	static std::mt19937       gen(rd());

	static std::uniform_real_distribution<float> dif(0.5, 0.8);
	static std::uniform_real_distribution<float> amb(0.05, 0.15);

	float4 d = make_float4(dif(gen), dif(gen), dif(gen), 1.0);
	float4 a = make_float4(amb(gen), amb(gen), amb(gen), 1.0);
	float4 s = make_float4(1, 1, 1, 3) - (d + a);

	return { s, d, a };
}

/** Host to device CUDA memcpy. */
static inline int __htod_cpy(void *to, const void *from, size_t n)
{
	return cudaSuccess != cudaMemcpy(to, from, n, cudaMemcpyHostToDevice);
}

void MaterialSet::randomize()
{
	std::vector<float4> _speculars(count);
	std::vector<float4> _diffuses(count);
	std::vector<float4> _ambients(count);

	for (size_t i = 0; i < count; ++i) {
		auto tmp = gen_random_mat_consts();

		_speculars[i] = tmp[0];
		_diffuses[i]  = tmp[1];
		_ambients[i]  = tmp[2];
	}

	size_t size = count * sizeof(float4);

	int ret = __htod_cpy(speculars.get(), _speculars.data(), size) ?:
	    __htod_cpy(diffuses.get(), _diffuses.data(), size)         ?:
	    __htod_cpy(ambients.get(), _ambients.data(), size);
	if (ret)
		throw rand_err_cons(-RAND_ERR_COPY);

	try {
		rand_fill_float(shininess.get(), count, 1.0f, 12.0f);
	} catch (const std::exception &e) {
		throw;
	}
}

static constexpr inline float3 make_cfloat3(float x, float y, float z)
{
	float3 ret = { .x = x, .y = y, .z = z };

	return ret;
}

constexpr AABB HostSphereSet::scene_bounds()
{
	float lo = CENTERS_LO - RADIUSES_UP;
	float up = CENTERS_UP + RADIUSES_UP;

	return AABB(make_cfloat3(lo, lo, lo), make_cfloat3(up, up, up));
}

HostSphereSet::HostSphereSet(size_t count)
    : count(count)
    , centers(count)
    , radiuses(count)
    , materials(count)
{
}

void HostSphereSet::randomize(size_t material_count)
{
	std::random_device rd;
	std::mt19937       gen(rd());

	std::uniform_real_distribution<float> center(CENTERS_LO, CENTERS_UP);
	std::uniform_real_distribution<float> radius(RADIUSES_LO, RADIUSES_UP);
	std::uniform_int_distribution<size_t> material(0, material_count - 1);

	std::generate(centers.begin(), centers.end(), [&gen, &center]() {
		return make_float4(center(gen), center(gen), center(gen), 0.0);
	});

	std::generate(radiuses.begin(), radiuses.end(), [&gen, &radius]() {
		return radius(gen);
	});

	std::generate(materials.begin(), materials.end(), [&gen, &material]() {
		return material(gen);
	});
}

SphereSet::SphereSet(const HostSphereSet &set)
    : count(set.count)
{
	try {
		centers   = make_unique_cuda<float4>(count);
		radiuses  = make_unique_cuda<float>(count);
		materials = make_unique_cuda<size_t>(count);
	} catch (const std::exception &e) {
		throw;
	}

	size_t csz = count * sizeof(float4);
	size_t rsz = count * sizeof(float);
	size_t msz = count * sizeof(size_t);

	int ret = __htod_cpy(centers.get(), set.centers.data(), csz) ?:
	    __htod_cpy(radiuses.get(), set.radiuses.data(), rsz)     ?:
	    __htod_cpy(materials.get(), set.materials.data(), msz);
	if (ret)
		throw rand_err_cons(-RAND_ERR_COPY);
}

Spheres::Spheres(size_t material_count, size_t sphere_count)
    : materials(material_count)
{
	try {
		mdesc = make_unique_cuda<MaterialSetDescriptor>(1);
		sdesc = make_unique_cuda<SphereSetDescriptor>(1);

		HostSphereSet hset(sphere_count);
		Octree        octree(hset, HostSphereSet::scene_bounds());

		materials.randomize();
		hset.randomize(material_count);

		spheres = std::make_unique<SphereSet>(hset);
		tree    = octree.flatten();

		init_mdesc();
		init_sdesc();
	} catch (const std::exception &e) {
		throw;
	}
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
		.centers   = spheres->centers.get(),
		.radiuses  = spheres->radiuses.get(),
		.materials = spheres->materials.get(),
		.count     = spheres->count
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

} // namespace device
