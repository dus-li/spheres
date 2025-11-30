// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Dus'li

#pragma once

#include <memory>
#include <vector>

#include "device/aabb.cuh"
#include "device/sphere.cuh"
#include "device/unique.cuh"

namespace device {

/**
 * A device-allocated octree SoA.
 * @see Octree
 */
struct FlattenedOctree {
	// Bounding boxes' vertices coordinates.
	unique_cuda<float> aabb_lo_xs;
	unique_cuda<float> aabb_up_xs;
	unique_cuda<float> aabb_lo_ys;
	unique_cuda<float> aabb_up_ys;
	unique_cuda<float> aabb_lo_zs;
	unique_cuda<float> aabb_up_zs;

	// Indices of child nodes.
	unique_cuda<size_t> children_0;
	unique_cuda<size_t> children_1;
	unique_cuda<size_t> children_2;
	unique_cuda<size_t> children_3;
	unique_cuda<size_t> children_4;
	unique_cuda<size_t> children_5;
	unique_cuda<size_t> children_6;
	unique_cuda<size_t> children_7;

	/// Concatenation of all leaf sphere index vectors.
	unique_cuda<size_t> leaf_indices;

	unique_cuda<size_t> leaf_bases; ///< Indices into leaf_indices.
	unique_cuda<size_t> leaf_sizes; ///< Popcounts of leaf_indices.

	unique_cuda<size_t> is_leaf;

	size_t count;
};

/** Host-side equivalent to @ref FlattenedOctree. */
struct FlattenedOctreeHost;

/**
 * An on-host octree recursive data type.
 *
 * These are not fit to be passed to the GPU. However, they are easier to
 * construct. Users should create an object of this class and then obtain a
 * flattened, GPU-allocated SoA instance via a call to @ref Octree::flatten.
 *
 * This implementation stores indices to an array in its leaves. This means that
 * by itself it isn't very useful and must be paired with some indexable
 * collection to become useful.
 */
class Octree {
	static constexpr size_t MAX_DEPTH        = 10;
	static constexpr size_t MAX_LEAF_SPHERES = 8;

	std::vector<size_t>     idx;     ///< Indices of intersected objects.
	std::unique_ptr<Octree> link[8]; ///< Pointers to children.
	AABB                    aabb;    ///< Bounding box.
	size_t                  depth;   ///< Depth of the node.
	bool                    leaf;    ///< If true = we are a leaf.

	size_t push_node(FlattenedOctreeHost &out);

  public:
	Octree(const HostSphereSet   &spheres,
	    const std::vector<size_t> running_idx, const AABB &box,
	    size_t depth);

	Octree(const HostSphereSet &spheres, const AABB &box);

	/** Compute a GPU-allocated SoA representation. */
	unique_cuda<FlattenedOctree> flatten();
};

} // namespace device
