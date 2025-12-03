// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Dus'li

#pragma once

#include <memory>
#include <vector>

#include "device/aabb.cuh"
#include "device/flattened_octree.cuh"
#include "device/sphere.cuh"

namespace device {

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
	std::unique_ptr<FlattenedOctree> flatten();
};

} // namespace device
