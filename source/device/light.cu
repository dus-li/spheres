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

static inline float __lightloc(std::function<float()> next, float lo, float hi)
{
	float ret;

	do
		ret = next();
	while (ret >= lo && ret <= hi);

	return ret;
}

DECLARE_NAMED_FILLER(rand_fill_light_loc, _rand_fillf, float4, float,
    float _x = __lightloc(next, -0.6, 0.6); //
    float _y = __lightloc(next, -0.6, 0.6); //
    float _z = __lightloc(next, -0.6, 0.6); //
    make_float4(_x, _y, _z, 0.0f);          //
)

void Lights::randomize()
{
	try {
		rand_fill_light_loc(locations.get(), count, -2.0f, 2.0f);
		rand_fill_float4(diffuses.get(), count, 0.3f, 0.8f);
		rand_fill_float4(speculars.get(), count, 0.1f, 0.3f);
	} catch (const std::exception &e) {
		throw;
	}
}

LightSetDescriptor *Lights::get_desc()
{
	return desc.get();
}

} // namespace device
