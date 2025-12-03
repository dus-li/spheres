// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Dus'li

#pragma once

#include <cstddef>

#include "device/aabb.cuh"
#include "device/flattened_octree.cuh"
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

/**
 * Structure of arrays containing sphere parameters, kept in host memory.
 *
 * There also exists @ref SphereSet which keeps the same data in GPU memory.
 * The justification for existance of two separate structs with the same
 * contents is that we need to preserve sphere data in main memory for a little
 * longer in order to enable building an Octree. The flow is as follows:
 *
 *   1. Spheres are initialized on host.
 *   2. Resulting instance of a HostSphereSet is used for Octree initialization.
 *   3. The same instance is used for creation of an instance of @ref SphereSet.
 *   4. The host-side instance's resources are freed.
 */
struct HostSphereSet {
	static constexpr float CENTERS_LO  = -0.5;
	static constexpr float CENTERS_UP  = 0.5;
	static constexpr float RADIUSES_LO = 0.005;
	static constexpr float RADIUSES_UP = 0.03;

	std::vector<float4> centers;   ///< Sphere centers.
	std::vector<float>  radiuses;  ///< Radiuses of the spheres.
	std::vector<size_t> materials; ///< Indices to a @ref MaterialSet.
	size_t              count;     ///< Number of elements.

	/** Compute vertices of bounding box limiting the scene. */
	static constexpr AABB scene_bounds();

	HostSphereSet(size_t count);
	void randomize(size_t material_count);
};

/**
 * A device-side counterpart to @ref HostSphereSet.
 * @see HostSphereSet
 */
struct SphereSet {
	unique_cuda<float4> centers;   ///< Sphere centers.
	unique_cuda<float>  radiuses;  ///< Radiuses of the spheres.
	unique_cuda<size_t> materials; ///< Indices to a @ref MaterialSet.
	size_t              count;     ///< Number of elements.

	SphereSet(const HostSphereSet &set);
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
	MaterialSet                      materials;
	std::unique_ptr<SphereSet>       spheres;
	std::unique_ptr<FlattenedOctree> tree;

	// Cached entries passed to kernels.
	unique_cuda<MaterialSetDescriptor> mdesc;
	unique_cuda<SphereSetDescriptor>   sdesc;
	unique_cuda<FOTDesc>               tdesc;

	void init_mdesc();
	void init_sdesc();
	void init_tdesc();

  public:
	Spheres(size_t material_count, size_t sphere_count);

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

	FOTDesc *get_tdesc();
};

} // namespace device
