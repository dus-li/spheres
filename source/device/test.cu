// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Dus'li

#include "device/test.cuh"

#include "types.hxx"

namespace device {

__global__ void render(uchar4 *fb, int width, int height, float t)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= width || y >= height)
		return;

	int time = t * 50;

	u8 r = (x + time) % 256;
	u8 g = (x + y + time) % 256;
	u8 b = (y + time) % 256;

	fb[y * width + x] = make_uchar4(r, g, b, 255);
}

} // namespace device
