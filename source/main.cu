// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Dus'li

#include <exception>
#include <iostream>
#include <string>

#include "host/appwindow.hxx"

#include "device/framebuffer.cuh"
#include "device/sphere.cuh"

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

	dim3   blk(32, 32);
	dim3   grd((width + 31) / 32, (height + 31) / 32);
	float3 cam = make_float3(0.0, 0.0, 1.5);

	try {
		device::Spheres s(10, 20);
		s.randomize();

		host::AppWindow app("Test Window", width, height, fb.h_fb);

		while (app.is_running()) {
			app.handle_events();

			device::kernels::render<<<grd, blk>>>(fb.d_fb.get(),
			    width,
			    height,
			    s.get_mdesc(),
			    s.get_sdesc(),
			    cam);
			cudaError_t ret = cudaDeviceSynchronize();
			error_wrapper(ret);

			fb.sync();
			app.update();

			SDL_Delay(16);
		}
	} catch (const std::exception &e) {
		std::cerr << "Error: " << e.what() << std::endl;
		return 1;
	}

	return 0;
}
