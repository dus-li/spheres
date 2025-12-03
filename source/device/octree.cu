// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Dus'li

#include <algorithm>
#include <memory>
#include <numeric>

#include "device/octree.cuh"
#include "device/vecops.cuh"

namespace device {

template <typename T>
static inline void try_copy(T *dest, const T *src, size_t n)
{
	if (cudaMemcpy(dest, src, n, cudaMemcpyHostToDevice) != cudaSuccess)
		throw std::runtime_error("Failed to copy to device");
}

template <typename T>
static inline unique_cuda<T> try_clone(const std::vector<T> &src)
{
	size_t size = sizeof(T) * src.size();

	try {
		unique_cuda<T> ret = make_unique_cuda<T>(src.size());
		try_copy(ret.get(), src.data(), size);

		return ret;
	} catch (const std::exception &e) {
		throw;
	}
}

void FlattenedOctree::to_desc(unique_cuda<FOTDesc> &ret)
{
	FOTDesc tmp;

	try {
		tmp.aabb_lo_xs   = aabb_lo_xs.get();
		tmp.aabb_up_xs   = aabb_up_xs.get();
		tmp.aabb_lo_ys   = aabb_lo_ys.get();
		tmp.aabb_up_ys   = aabb_up_ys.get();
		tmp.aabb_lo_zs   = aabb_lo_zs.get();
		tmp.aabb_up_zs   = aabb_up_zs.get();
		tmp.children_0   = children_0.get();
		tmp.children_1   = children_1.get();
		tmp.children_2   = children_2.get();
		tmp.children_3   = children_3.get();
		tmp.children_4   = children_4.get();
		tmp.children_5   = children_5.get();
		tmp.children_6   = children_6.get();
		tmp.children_7   = children_7.get();
		tmp.leaf_indices = leaf_indices.get();
		tmp.leaf_bases   = leaf_bases.get();
		tmp.leaf_sizes   = leaf_sizes.get();
		tmp.is_leaf      = is_leaf.get();

		try_copy(ret.get(), &tmp, sizeof(tmp));
	} catch (const std::exception &e) {
		throw;
	}
}

static inline std::vector<size_t> all_spheres(const HostSphereSet &spheres)
{
	std::vector<size_t> ret(spheres.count);

	std::iota(ret.begin(), ret.end(), 0);

	return ret;
}

#define spot printf("%s:%d\n", __FILE__, __LINE__)

Octree::Octree(const HostSphereSet &spheres,
    const std::vector<size_t> running_idx, const AABB &box, size_t depth)
    : aabb(box)
    , link { nullptr }
    , depth(depth)
    , leaf(false)
{
	spot;
	const size_t idx_sz = running_idx.size();

	// Compute indices of intersecting spheres.
	idx.reserve(idx_sz);
	for (size_t i : running_idx) {
		float3 center = f4_xyz(spheres.centers[i]);
		if (aabb.intersects(center, spheres.radiuses[i]))
			idx.push_back(i);
	}

	// If we qualify to become a leaf we can call it a day here.
	if (depth >= MAX_DEPTH || idx.size() <= MAX_LEAF_SPHERES) {
		spot;
		leaf = true;
		return;
	}

	// Sub-divide into children.
	std::array<AABB, 8> cs;
	aabb.split(cs);

	// Compute running_idx for children.
	std::array<std::vector<size_t>, 8> is;
	for (size_t baby = 0; baby < 8; ++baby) {
		printf("cs[%ld].lo = %f %f %f\n",
		    baby,
		    cs[baby].lo.x,
		    cs[baby].lo.y,
		    cs[baby].lo.z);
		printf("cs[%ld].up = %f %f %f\n",
		    baby,
		    cs[baby].up.x,
		    cs[baby].up.y,
		    cs[baby].up.z);

		std::vector<size_t> &cidx = is[baby];

		for (size_t i : running_idx) {
			float3 center = f4_xyz(spheres.centers[i]);
			if (cs[baby].intersects(center, spheres.radiuses[i]))
				cidx.push_back(i);
		}
	}

	// Check if at least one child has fewer spheres intersecting. If
	// none does, then this subdivision is pointless, because it fails
	// to reduce sphere count and thus does not aid in searching the space.
	auto p = [&idx_sz](auto &v) { return v.size() < idx_sz; };
	if (std::none_of(is.begin(), is.end(), p)) {
		spot;
		leaf = true;
		return;
	}

	// Recurse
	for (size_t i = 0; i < 8; ++i) {
		using std::make_unique;
		link[i] = make_unique<Octree>(spheres, is[i], cs[i], depth + 1);
	}
}

Octree::Octree(const HostSphereSet &spheres, const AABB &box)
    : Octree(spheres, all_spheres(spheres), box, 0)
{
}

/** Host-side equivalent to @ref FlattenedOctree. */
struct FlattenedOctreeHost {
	// Bounding boxes' vertices coordinates.
	std::vector<float> aabb_lo_xs;
	std::vector<float> aabb_up_xs;
	std::vector<float> aabb_lo_ys;
	std::vector<float> aabb_up_ys;
	std::vector<float> aabb_lo_zs;
	std::vector<float> aabb_up_zs;

	// Indices of child nodes.
	std::array<std::vector<size_t>, 8> children;

	/// Concatenation of all leaf sphere index vectors.
	std::vector<size_t> leaf_indices;

	std::vector<size_t> leaf_bases; ///< Indices into leaf_indices.
	std::vector<size_t> leaf_sizes; ///< Popcounts of leaf_indices.

	std::vector<size_t> is_leaf;

	size_t size() const { return aabb_lo_xs.size(); }

	std::unique_ptr<FlattenedOctree> to_device();
};

std::unique_ptr<FlattenedOctree> FlattenedOctreeHost::to_device()
{
	auto ret = std::make_unique<FlattenedOctree>();

	try {
		// Bounding boxes
		ret->aabb_lo_xs = try_clone(aabb_lo_xs);
		ret->aabb_up_xs = try_clone(aabb_up_xs);
		ret->aabb_lo_ys = try_clone(aabb_lo_ys);
		ret->aabb_up_ys = try_clone(aabb_up_ys);
		ret->aabb_lo_zs = try_clone(aabb_lo_zs);
		ret->aabb_up_zs = try_clone(aabb_up_zs);

		// Children
		ret->children_0 = try_clone(children[0]);
		ret->children_1 = try_clone(children[1]);
		ret->children_2 = try_clone(children[2]);
		ret->children_3 = try_clone(children[3]);
		ret->children_4 = try_clone(children[4]);
		ret->children_5 = try_clone(children[5]);
		ret->children_6 = try_clone(children[6]);
		ret->children_7 = try_clone(children[7]);

		// Leaf info
		ret->leaf_indices = try_clone(leaf_indices);
		ret->leaf_bases   = try_clone(leaf_bases);
		ret->leaf_sizes   = try_clone(leaf_sizes);
		ret->is_leaf      = try_clone(is_leaf);

		// Put it together
		return ret;
	} catch (const std::exception &e) {
		throw;
	}
}

size_t Octree::push_node(FlattenedOctreeHost &out)
{
	const size_t SIZE_MAX = std::numeric_limits<size_t>::max();

	size_t outidx = out.size();

	// AABB
	out.aabb_lo_xs.push_back(aabb.lo.x);
	out.aabb_up_xs.push_back(aabb.up.x);
	out.aabb_lo_ys.push_back(aabb.lo.y);
	out.aabb_up_ys.push_back(aabb.up.y);
	out.aabb_lo_zs.push_back(aabb.lo.z);
	out.aabb_up_zs.push_back(aabb.up.z);

	// Initialize children to a known sentinel value.
	std::for_each(out.children.begin(),
	    out.children.end(),
	    [&SIZE_MAX](auto &v) { v.push_back(SIZE_MAX); });

	out.leaf_bases.push_back(SIZE_MAX);
	out.leaf_sizes.push_back(0);
	out.is_leaf.push_back(leaf);

	// If we are a leaf - cat out nodes to leaf_bases and return.
	if (leaf) {
		size_t fst = out.leaf_indices.size();
		for (size_t i : idx)
			out.leaf_indices.push_back(i);

		out.leaf_bases[outidx] = fst;
		out.leaf_sizes[outidx] = idx.size();

		return outidx;
	}

	for (size_t i = 0; i < 8; ++i) {
		if (link[i] == nullptr)
			continue;

		out.children[i][outidx] = link[i]->push_node(out);
	}

	return outidx;
}

std::unique_ptr<FlattenedOctree> Octree::flatten()
{
	FlattenedOctreeHost tmp;
	push_node(tmp);

	spot;
	return tmp.to_device();
}

} // namespace device
