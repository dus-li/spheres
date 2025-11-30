// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Dus'li

#include <algorithm>
#include <memory>
#include <numeric>

#include "device/octree.cuh"
#include "device/vecops.cuh"

namespace device {

static inline std::vector<size_t> all_spheres(const HostSphereSet &spheres)
{
	std::vector<size_t> ret(spheres.count);

	std::iota(ret.begin(), ret.end(), 0);

	return ret;
}

Octree::Octree(const HostSphereSet &spheres,
    const std::vector<size_t> running_idx, const AABB &box, size_t depth)
    : aabb(box)
    , link { nullptr }
    , depth(depth)
    , leaf(false)
{
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
		leaf = true;
		return;
	}

	// Sub-divide into children.
	std::array<AABB, 8> cs;
	aabb.split(cs);

	// Compute running_idx for children.
	std::array<std::vector<size_t>, 8> is;
	for (size_t baby = 0; baby < 8; ++baby) {
		std::vector<size_t> &cidx = is[baby];
		cidx.reserve(idx_sz);

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

unique_cuda<FlattenedOctree> Octree::flatten()
{
	// TODO
	return nullptr;
}

} // namespace device
