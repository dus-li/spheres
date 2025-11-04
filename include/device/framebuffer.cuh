// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Dus'li

#pragma once

#include <cstddef>
#include <tuple>
#include <vector>

#include "device/unique.cuh"

#include "types.hxx"

namespace device {

using std::vector;

/** A pairing of syncable device and host buffers. */
class Framebuffer {
	size_t   count; ///< Number of RGBA cells in the buffer.
	size_t   size;  ///< Size of each of the buffers, in bytes.
	unsigned width;
	unsigned height;

  public:
	unique_cuda<uchar4> d_fb; ///< Device-side buffer.
	vector<u32>         h_fb; ///< Host-side buffer.

	/**
	 * Allocate a new framebuffer.
	 * @param count Number of RGBA cells in the buffer.
	 *
	 * @throws std::runtime_error Failed to allocate a buffer.
	 */
	Framebuffer(unsigned width, unsigned height);

	/**
	 * Copy contents of the devicce buffer to the host buffer.
	 * @throws std::runtime_error Failed to copy.
	 */
	void sync();

	dim3 get_dims();
};

} // namespace device
