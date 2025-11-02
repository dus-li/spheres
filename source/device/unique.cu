// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Dus'li

#include <stdexcept>

#include "device/unique.cuh"

namespace device {

void deleters_cuda::CudaDeleter::operator()(void *p) const noexcept
{
	if (p)
		cudaFree(p);
}

} // namespace device
