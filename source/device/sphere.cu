// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Dus'li

#include <exception>
#include <iostream>
#include <random>
#include <ranges>
#include <stdexcept>
#include <string>

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
static inline int __htod_cpy(void *to, void *from, size_t n)
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
	try {
		rand_fill_float4_0(centers.get(), count, -0.5f, 0.5f);
		rand_fill_float(radiuses.get(), count, 0.005f, 0.03f);
		rand_fill_size_t(materials.get(), count, 0, material_count);
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

} // namespace device
