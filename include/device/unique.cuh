// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Dus'li

#pragma once

#include <cstddef>
#include <memory>
#include <stdexcept>

namespace device {

/** Functors allowing usage of std::unique_ptr for CUDA-allocated data. */
namespace deleters_cuda {

	struct CudaDeleter {
		void operator()(void *p) const noexcept;
	};

} // namespace deleters_cuda

template <typename T>
using unique_cuda = std::unique_ptr<T, deleters_cuda::CudaDeleter>;

/**
 * Allocate a unique CUDA buffer.
 * @tparam T     Type of elements in the buffer.
 * @param  count Number of elements in the buffer.
 *
 * @return A unique pointer to newly CUDA-allocated data.
 * @throws std::runtime_error Failed to allocated memory.
 */
template <typename T> unique_cuda<T> make_unique_cuda(size_t count);

} // namespace device

#include "template/unique.tpp"
