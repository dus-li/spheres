// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Dus'li

#include <exception>
#include <iostream>
#include <string>

#include "host/appwindow.hxx"

#include "device/framebuffer.cuh"
#include "device/test.cuh"

using device::Framebuffer;

void error_wrapper(cudaError_t err)
{
	if (err != cudaSuccess) {
		std::string msg("Error: ");
		msg.append(cudaGetErrorString(err));
		throw std::runtime_error(msg);
	}
}

int main()
{
	const unsigned width  = 640;
	const unsigned height = 480;

	Framebuffer fb(width * height);
	cudaError_t ret;
	float       time = 0;

	dim3 blk(16, 16);
	dim3 grd((width + blk.x - 1) / blk.x, (height + blk.y - 1) / blk.y);

	try {
		host::AppWindow app("Test Window", width, height, fb.h_fb);

		while (app.is_running()) {
			app.handle_events();

			device::render<<<grd, blk>>>(fb.d_fb.get(),
			    width,
			    height,
			    time);
			ret = cudaDeviceSynchronize();
			error_wrapper(ret);

			fb.sync();
			time += .016;

			app.update();
			SDL_Delay(16);
		}
	} catch (const std::exception &e) {
		std::cerr << "Error: " << e.what() << std::endl;
		return 1;
	}

	return 0;
}
