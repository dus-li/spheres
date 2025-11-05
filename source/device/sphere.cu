// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Dus'li

#include <exception>
#include <stdexcept>

#include "device/randomizer.cuh"
#include "device/sphere.cuh"

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
		rand_fill_float(shininess.get(), count, 0.5f, 30.0f);
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

} // namespace device
