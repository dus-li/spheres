// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Dus'li

#include <iostream>

#include "host/test.hxx"

#include "device/test.cuh"

int main()
{
	std::cout << "Hello from host!" << std::endl;

	host::test();
	device::test<<<1, 4>>>();

	cudaDeviceSynchronize();

	return 0;
}
