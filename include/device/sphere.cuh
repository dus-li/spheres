// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Dus'li

#pragma once

#include <cstddef>

#include "device/unique.cuh"

namespace device {

/** Structure of arrays containing material parameters. */
struct MaterialSet {
	unique_cuda<float4> speculars; ///< Specular reflection constants.
	unique_cuda<float4> diffuses;  ///< Diffuse reflection constants.
	unique_cuda<float4> ambients;  ///< Ambient reflection constants.
	unique_cuda<float>  shininess; ///< Shininess constants.
	size_t              count;     ///< Number of elements.

	MaterialSet(size_t count);
	void randomize();
};

/** Structure of arrays containing sphere parameters. */
struct SphereSet {
	unique_cuda<float4> centers;   ///< Sphere centers.
	unique_cuda<float>  radiuses;  ///< Radiuses of the spheres.
	unique_cuda<size_t> materials; ///< Indices to a @ref MaterialSet.
	size_t              count;     ///< Number of elements.

	SphereSet(size_t count);
	void randomize(size_t material_count);
};

/**
 * Auxiliary structure for passing data in @ref MaterialSet to kernels.
 *
 * Instances are generally obtainable through a call to @ref Spheres::get_mdesc.
 */
struct MaterialSetDescriptor {
	float4 *speculars;
	float4 *diffuses;
	float4 *ambients;
	float  *shininess;
};

/**
 * Auxiliary structure for passing data in @ref SphereSet to kernels.
 *
 * Instances are generally obtainable through a call to @ref Spheres::get_sdesc.
 */
struct SphereSetDescriptor {
	float4 *centers;
	float  *radiuses;
	size_t *materials;
	size_t  count;
};

/** Complete set of information about spheres present in the scene. */
class Spheres {
	MaterialSet materials;
	SphereSet   spheres;

	// Cached entries passed to kernels.
	unique_cuda<MaterialSetDescriptor> mdesc;
	unique_cuda<SphereSetDescriptor>   sdesc;

	void init_mdesc();
	void init_sdesc();

  public:
	Spheres(size_t material_count, size_t sphere_count);

	/** Populate sphere and material structures with random data. */
	void randomize();

	/**
	 * Obtain a descriptor of materials usable in CUDA kernels.
	 * @return A device pointer to a product-type with array pointers.
	 */
	MaterialSetDescriptor *get_mdesc();

	/**
	 * Obtain a descriptor of spheres usable in CUDA kernels.
	 * @return A device pointer to a product-type with array pointers.
	 */
	SphereSetDescriptor *get_sdesc();
};

} // namespace device
