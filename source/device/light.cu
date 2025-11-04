// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Dus'li

#include <stdexcept>

#include "device/light.cuh"
#include "device/randomizer.cuh"

namespace device {

Lights::Lights(size_t count)
    : count(count)
{
	try {
		locations = make_unique_cuda<float4>(count);
		diffuses  = make_unique_cuda<float4>(count);
		speculars = make_unique_cuda<float4>(count);
		desc      = make_unique_cuda<LightSetDescriptor>(1);
	} catch (const std::exception &e) {
		throw;
	}

	LightSetDescriptor tmp = {
		/**/
		.locations = locations.get(),
		.diffuses  = diffuses.get(),
		.speculars = speculars.get(),
		.count     = count,
	};

	cudaError_t rv;
	rv = cudaMemcpy(desc.get(), &tmp, sizeof(tmp), cudaMemcpyHostToDevice);
	if (rv != cudaSuccess)
		throw std::runtime_error("Failed to copy to device");
}

void Lights::randomize()
{
	try {
		rand_fill_float4_0(locations.get(), count, -2.f, 2.f);
		rand_fill_float4(diffuses.get(), count, 0.3f, 0.9f);
		rand_fill_float4(speculars.get(), count, 0.5f, 1.f);
	} catch (const std::exception &e) {
		throw;
	}
}

LightSetDescriptor *Lights::get_desc()
{
	return desc.get();
}

} // namespace device
