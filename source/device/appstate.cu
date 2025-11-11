// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Dus'li

#include <exception>

#include <SDL2/SDL_events.h>
#include <SDL2/SDL_mouse.h>
#include <SDL2/SDL_video.h>

#include "device/appstate.cuh"
#include "device/vecops.cuh"

const float mouse_sensitivity = 0.002;
const float kbd_sensitivity   = 0.02;

namespace device {

void AppState::make_moves()
{
	CamBasis  basis = scene.cam.basis();
	const u8 *kbd   = window.get_kbd_state();

	if (kbd[SDL_SCANCODE_W])
		scene.cam.move_by(kbd_sensitivity * basis[0]);

	if (kbd[SDL_SCANCODE_S])
		scene.cam.move_by(-kbd_sensitivity * basis[0]);

	if (kbd[SDL_SCANCODE_A])
		scene.cam.move_by(-kbd_sensitivity * basis[1]);

	if (kbd[SDL_SCANCODE_D])
		scene.cam.move_by(kbd_sensitivity * basis[1]);
}

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

	// Window closing
	window.register_callback(SDL_KEYDOWN, [this](const SDL_Event &e) {
		if (e.key.keysym.sym == SDLK_q)
			window.close();
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
			make_moves();
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
