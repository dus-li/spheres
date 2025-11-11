// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Dus'li

#pragma once

#include <cstddef>

#include "host/appwindow.hxx"

#include "device/framebuffer.cuh"
#include "device/scene.cuh"

#include "types.hxx"

namespace device {

class AppState {
	Framebuffer     fb;
	Scene           scene;
	host::AppWindow window;

	void make_moves();

  public:
	AppState(size_t materials, size_t spheres, size_t lights,
	    u32 width = 1280, u32 height = 720, const char *title = "Spheres");

	void run();
};

} // namespace device
