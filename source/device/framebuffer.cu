// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Dus'li

#include <stdexcept>
#include <string>

#include "host/font.hxx"

#include "device/framebuffer.cuh"
#include "device/vecops.cuh"

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

void Framebuffer::h_put_pixel(size_t x, size_t y, uchar4 color)
{
	// TODO: civilize this
	u32 tmp = *(u32 *)&color;

	h_fb[y * width + x] = tmp;
}

void Framebuffer::h_text(size_t x, size_t y, const std::string &txt,
    uchar4 color)
{
	// const host::FontSingleton &atlas = host::FontSingleton::get();

	// auto      g       = atlas.glyphs();
	// const u8 *bmp     = atlas.bitmap();
	// size_t    width   = atlas.atlas_width();
	// size_t    height  = atlas.atlas_height();
	// size_t    xcursor = x;

	// for (char c : txt) {
	//	if (!atlas.char_supported(c))
	//		continue;

	//	const stbtt_bakedchar &bc = g[c - atlas.FIRST_CHAR];

	//	size_t g_w = bc.x1 - bc.x0;
	//	size_t g_h = bc.y1 - bc.y0;

	//	for (size_t iy = 0; iy < g_h; ++iy) {
	//		for (size_t ix = 0; ix < g_w; ++ix) {
	//			size_t ax = bc.x0 + ix;
	//			size_t ay = bc.y0 + iy;
	//			u8     a  = bmp[ay * width + ax];

	//			if (a == 0)
	//				continue;

	//			// TODO: blend with background
	//			size_t xdst = xcursor + bc.xoff + ix;
	//			size_t ydst = y + bc.yoff + iy;
	//			h_put_pixel(xdst, ydst, color);
	//		}
	//	}

	//	xcursor += bc.xadvance;
	//}
}

} // namespace device
