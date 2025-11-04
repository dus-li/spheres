// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Dus'li

#pragma once

#include <cstddef>

#include "device/framebuffer.cuh"
#include "device/light.cuh"
#include "device/sphere.cuh"

namespace device {

class Scene {
	Lights  lights;
	Spheres spheres;
	float3  cam;
	float3  ambient;

  public:
	Scene(size_t material_count, size_t sphere_count, size_t light_count);
	void reposition_cam(float3 where);
	void randomize();
	void render_to(Framebuffer &fb);
};

} // namespace device
