// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Dus'li

#pragma once

#include "device/unique.cuh"

namespace device {

/**
 * Flattened octree descriptor.
 * @see FlattenedOctree
 */
struct FOTDesc {
	float  *aabb_lo_xs;
	float  *aabb_up_xs;
	float  *aabb_lo_ys;
	float  *aabb_up_ys;
	float  *aabb_lo_zs;
	float  *aabb_up_zs;
	size_t *children_0;
	size_t *children_1;
	size_t *children_2;
	size_t *children_3;
	size_t *children_4;
	size_t *children_5;
	size_t *children_6;
	size_t *children_7;
	size_t *leaf_indices;
	size_t *leaf_bases;
	size_t *leaf_sizes;
	size_t *is_leaf;

	size_t count;
};

/**
 * A device-allocated octree SoA.
 * @see Octree
 */
struct FlattenedOctree {
	// bounding boxes' vertices cooradinates.
	unique_cuda<float> aabb_lo_xs;
	unique_cuda<float> aabb_up_xs;
	unique_cuda<float> aabb_lo_ys;
	unique_cuda<float> aabb_up_ys;
	unique_cuda<float> aabb_lo_zs;
	unique_cuda<float> aabb_up_zs;

	// indices of child nodes.
	unique_cuda<size_t> children_0;
	unique_cuda<size_t> children_1;
	unique_cuda<size_t> children_2;
	unique_cuda<size_t> children_3;
	unique_cuda<size_t> children_4;
	unique_cuda<size_t> children_5;
	unique_cuda<size_t> children_6;
	unique_cuda<size_t> children_7;

	/// concatenation of all leaf sphere index vectors.
	unique_cuda<size_t> leaf_indices;

	unique_cuda<size_t> leaf_bases; ///< indices into leaf_indices.
	unique_cuda<size_t> leaf_sizes; ///< popcounts of leaf_indices.

	unique_cuda<size_t> is_leaf;

	size_t count;

	void to_desc(unique_cuda<FOTDesc> &ret);
};

} // namespace device
