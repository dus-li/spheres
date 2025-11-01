// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Dus'li

#include "device/test.cuh"

namespace device {

__global__ void test()
{
	printf("Hello from device thread #%u!\n", threadIdx.x);
}

} // namespace device
