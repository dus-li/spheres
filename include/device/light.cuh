// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Dus'li

#pragma once

#include "device/unique.cuh"

namespace device {

/**
 * Auxiliary structure for passing data in @ref Lights to kernels.
 *
 * Instances of this structure are obtainable through a call to
 * @ref Lights::get_desc.
 */
struct LightSetDescriptor {
	float4 *locations;
	float4 *diffuses;
	float4 *speculars;
	size_t  count;
};

/** Complete set of information about light sources present in the scene. */
class Lights {
	unique_cuda<float4> locations;
	unique_cuda<float4> diffuses;
	unique_cuda<float4> speculars;
	size_t              count;

	// Cached descriptor passed to kernels.
	unique_cuda<LightSetDescriptor> desc;

  public:
	Lights(size_t count);

	/** Populate light source parameters with random data. */
	void randomize();

	/**
	 * Obtain a descriptor of light sources, usable in CUDA kernels.
	 * @return A device pointer to a product-type with array pointers.
	 */
	LightSetDescriptor *get_desc();
};

} // namespace device
