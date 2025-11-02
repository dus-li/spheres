// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Dus'li

namespace device {

template <typename T> unique_cuda<T> make_unique_cuda(size_t count)
{
	T *d_raw = nullptr;

	cudaError_t ret = cudaMalloc(&d_raw, count * sizeof(T));
	if (ret != cudaSuccess)
		throw std::runtime_error("Failed to allocate device memeory");

	return unique_cuda<T>(d_raw);
}

} // namespace device
