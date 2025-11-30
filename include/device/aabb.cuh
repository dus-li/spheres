// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Dus'li

#pragma once

#include <array>

namespace device {

/** Axis-aligned bounding box. */
struct AABB {
	float3 lo;
	float3 up;

	constexpr AABB(float3 lo = { 0, 0, 0 }, float3 up = { 0, 0, 0 })
	    : lo(lo)
	    , up(up) { };

	/** Check for sphere intersection. */
	bool intersects(float3 center, float radius);

	/** Split the box into 8 subboxes of equal volume. */
	void split(std::array<AABB, 8> &out);
};

} // namespace device
