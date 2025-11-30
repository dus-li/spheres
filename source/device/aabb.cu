// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Dus'li

#include "device/aabb.cuh"
#include "device/vecops.cuh"

namespace device {

static constexpr inline float axis_distance(float point, float lo, float up)
{
	return point < lo ? lo - point : point > up ? point - up : 0.0f;
}

bool AABB::intersects(float3 center, float radius)
{
	float dx = axis_distance(center.x, lo.x, up.x);
	float dy = axis_distance(center.y, lo.y, up.y);
	float dz = axis_distance(center.z, lo.z, up.z);

	return dx * dx + dy * dy + dz * dz <= radius * radius;
}

void AABB::split(std::array<AABB, 8> &out)
{
	const size_t POS_X = (1 << 0);
	const size_t POS_Y = (1 << 1);
	const size_t POS_Z = (1 << 2);

	float3 mid = 0.5f * (lo + up);

	// Sub-box indices are encoded depending on their location in relation
	// to the center of the AABB:
	//
	//             -Z                     +Y
	//    +--------+--------+         +--------+
	//    |        |        |         |        |
	//    |  0X1b  |  0X0b  |         |  X1Xb  |
	//    |        |        |         |        |
	// +X +--------+--------+ -X      +--------+
	//    |        |        |         |        |
	//    |  1X1b  |  1X0b  |         |  X0Xb  |
	//    |        |        |         |        |
	//    +--------+--------+         +--------+
	//             +Z                     -Y
	//
	for (size_t i = 0; i < out.size(); ++i) {
		out[i].lo.x = (i & POS_X) ? mid.x : lo.x;
		out[i].up.x = (i & POS_X) ? up.x : mid.x;

		out[i].lo.y = (i & POS_Y) ? mid.y : lo.y;
		out[i].up.y = (i & POS_Y) ? up.y : mid.y;

		out[i].lo.z = (i & POS_Z) ? mid.z : lo.z;
		out[i].up.z = (i & POS_Z) ? up.z : mid.z;
	}
}

} // namespace device
