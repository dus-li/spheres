// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Dus'li

#pragma once

#include <cstddef>

#include "device/framebuffer.cuh"
#include "device/light.cuh"
#include "device/sphere.cuh"

namespace device {

/** Complete set of information necessary to render a scene. */
class Scene {
	Lights  lights;
	Spheres spheres;
	float3  cam;     ///< Camera location.
	float3  ambient; ///< Ambient light constant.

  public:
	Scene(size_t material_count, size_t sphere_count, size_t light_count);

	/** Place the camera at a new location. */
	void reposition_cam(float3 where);

	/** Randomize all scene components. */
	void randomize();

	/** Raycast and render the scene to a framebuffer. */
	void render_to(Framebuffer &fb);
};

} // namespace device
