// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Dus'li

#include <exception>
#include <iostream>

#include "host/appwindow.hxx"

#include "device/framebuffer.cuh"
#include "device/scene.cuh"

int main()
{
	const unsigned width  = 1280;
	const unsigned height = 720;

	try {
		device::Framebuffer fb(width, height);
		device::Scene       scene(5, 20, 4);
		scene.randomize();

		host::AppWindow app("Spheres", width, height, fb.h_fb);

		while (app.is_running()) {
			app.handle_events();
			scene.render_to(fb);
			app.update();
			SDL_Delay(16);
		}
	} catch (const std::exception &e) {
		std::cerr << "Error: " << e.what() << std::endl;
		return 1;
	}

	return 0;
}
