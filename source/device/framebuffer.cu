// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Dus'li

#include <stdexcept>

#include "device/framebuffer.cuh"

namespace device {

Framebuffer::Framebuffer(unsigned width, unsigned height)
    : count(width * height)
    , width(width)
    , height(height)
{
	size = count * sizeof(uchar4);
	h_fb.reserve(count);

	try {
		d_fb = make_unique_cuda<uchar4>(count);
	} catch (const std::exception &e) {
		throw;
	}
}

void Framebuffer::sync()
{
	cudaError_t r;

	r = cudaMemcpy(h_fb.data(), d_fb.get(), size, cudaMemcpyDeviceToHost);
	if (r != cudaSuccess)
		throw std::runtime_error("Failed to copy to host");
}

dim3 Framebuffer::get_dims()
{
	return dim3(width, height);
}

} // namespace device
