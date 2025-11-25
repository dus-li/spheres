// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Dus'li

#pragma once

#include <array>
#include <cstddef>

#include "device/framebuffer.cuh"
#include "device/light.cuh"
#include "device/sphere.cuh"

namespace device {

using CamBasis = std::array<float3, 3>;

class Camera {
	friend class Scene;

	float3 location;
	float  yaw;
	float  pitch;
	float  fov;

  public:
	Camera(float3 location, float yaw, float pitch, float fov);
	void     rotate_y(float rad);
	void     rotate_x(float rad);
	void     move_by(float3 v);
	CamBasis basis();
};

/** Complete set of information necessary to render a scene. */
class Scene {
	Lights  lights;
	Spheres spheres;
	float3  ambient; ///< Ambient light constant.

  public:
	Camera cam;

	Scene(size_t material_count, size_t sphere_count, size_t light_count);

	/** Randomize all scene components. */
	void randomize();

	/**
	 * Raycast and render the scene to a framebuffer.
	 * @param fb Target framebuffer.
	 */
	void render_to(Framebuffer &fb);
};

} // namespace device
