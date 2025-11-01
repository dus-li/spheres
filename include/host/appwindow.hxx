// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Dus'li

#pragma once

#include <memory>
#include <string>

#include <SDL2/SDL.h>

#include "types.hxx"

/**
 * @file  appwindow.hxx
 * @brief Application window management.
 *
 * This file exposes a simple wrapper interface around SDL2 that allows creation
 * and display of a window with a supplied framebuffer.
 *
 * @todo Input handling
 */

namespace host {

using std::string;
using std::unique_ptr;

/** Functors allowing creation of unique_ptr-based types for SDL resources. */
namespace deleters_sdl {

	struct WindowDeleter {
		void operator()(SDL_Window *w) const;
	};

	struct RendererDeleter {
		void operator()(SDL_Renderer *w) const;
	};

	struct TextureDeleter {
		void operator()(SDL_Texture *w) const;
	};

} // namespace deleters_sdl

using UniqueWindow   = unique_ptr<SDL_Window, deleters_sdl::WindowDeleter>;
using UniqueRenderer = unique_ptr<SDL_Renderer, deleters_sdl::RendererDeleter>;
using UniqueTexture  = unique_ptr<SDL_Texture, deleters_sdl::TextureDeleter>;

/** An application window class. */
class AppWindow {
	unsigned       width;    ///< Width of the window.
	unsigned       height;   ///< Height of the window.
	string         name;     ///< Title of the window.
	u32           *buf;      ///< Framebuffer.
	bool           running;  ///< False if application was closed.
	UniqueWindow   window;   ///< SDL window.
	UniqueRenderer renderer; ///< SDL renderer.
	UniqueTexture  texture;  ///< SDL texture displayed by renderer.

  public:
	/**
	 * Construct a new application window.
	 * @param name   Title of the window.
	 * @param width  Width of the window.
	 * @param height Height of the window.
	 * @param buf    Framebuffer that is to be displayed by the window.
	 *
	 * @throws std::runtime_error When internal object creation fails.
	 */
	AppWindow(string name, unsigned width, unsigned height, u32 *buf);
	~AppWindow();

	/**
	 * Check whether the application has been closed.
	 *
	 * @return @a false If the application was closed.
	 * @return @a true If the application was not closed.
	 */
	bool is_running();

	/** Process incoming window events. */
	void handle_events();

	/** Update displayed window with contents of the framebuffer. */
	void update();
}; // class AppWindow

} // namespace host
