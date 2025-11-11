// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Dus'li

#include <exception>

#include <SDL2/SDL_events.h>

#include "device/appstate.cuh"
#include "device/vecops.cuh"

const float mouse_sensitivity = 0.002;
const float kbd_sensitivity   = 0.02;

namespace device {

AppState::AppState(size_t materials, size_t spheres, size_t lights, u32 width,
    u32 height, const char *title)
// clang-format off
try
    : fb(width, height)
    , scene(materials, spheres, lights)
    , window(title, width, height, fb.h_fb)
// clang-format on
{
	scene.randomize();

	// Moving camera & closing window
	window.register_callback(SDL_KEYDOWN, [this](const SDL_Event &e) {
		CamBasis basis = scene.cam.basis();

		switch (e.key.keysym.sym) {
		case SDLK_w:
			scene.cam.move_by(kbd_sensitivity * basis[0]);
			break;
		case SDLK_s:
			scene.cam.move_by(-kbd_sensitivity * basis[0]);
			break;
		case SDLK_a:
			scene.cam.move_by(-kbd_sensitivity * basis[1]);
			break;
		case SDLK_d:
			scene.cam.move_by(kbd_sensitivity * basis[1]);
			break;
		case SDLK_q:
			window.close();
			break;
		}
	});

	// Rotating camers
	window.register_callback(SDL_MOUSEMOTION, [this](const SDL_Event &e) {
		scene.cam.rotate_y(-mouse_sensitivity * e.motion.xrel);
		scene.cam.rotate_x(-mouse_sensitivity * e.motion.yrel);
	});
} catch (const std::exception &e) {
	throw;
}

void AppState::run()
{
	try {
		while (window.is_running()) {
			window.handle_events();
			scene.render_to(fb);
			window.update();
			SDL_Delay(16);
		}
	} catch (const std::exception &e) {
		throw;
	}
}

} // namespace device
