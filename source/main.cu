// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Dus'li

#include <exception>
#include <iostream>

#include "host/appwindow.hxx"

#include "device/test.cuh"

int main()
{
	const unsigned width  = 640;
	const unsigned height = 480;

	u32 *fb = new uint32_t[width * height * sizeof(uint32_t)];

	try {
		host::AppWindow app("Test Window", width, height, fb);

		while (app.is_running()) {
			app.handle_events();
			app.update();
			SDL_Delay(16);
		}
	} catch (const std::exception &e) {
		std::cerr << "Error: " << e.what() << std::endl;
		return 1;
	}

	return 0;
}
