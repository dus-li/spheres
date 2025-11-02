// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Dus'li

#include <stdexcept>

#include "host/appwindow.hxx"

namespace host {

void deleters_sdl::WindowDeleter::operator()(SDL_Window *w) const
{
	if (w)
		SDL_DestroyWindow(w);
}

void deleters_sdl::RendererDeleter::operator()(SDL_Renderer *r) const
{
	if (r)
		SDL_DestroyRenderer(r);
}

void deleters_sdl::TextureDeleter::operator()(SDL_Texture *t) const
{
	if (t)
		SDL_DestroyTexture(t);
}

AppWindow::AppWindow(string name, unsigned width, unsigned height, Buffer buf)
    : width(width)
    , height(height)
    , name(name)
    , buf(buf)
    , running(true)
{
	if (SDL_Init(SDL_INIT_VIDEO))
		throw std::runtime_error("Failed to initialize SDL");

	window = UniqueWindow(SDL_CreateWindow(name.c_str(),
	    SDL_WINDOWPOS_CENTERED,
	    SDL_WINDOWPOS_CENTERED,
	    width,
	    height,
	    SDL_WINDOW_SHOWN));
	if (!window)
		throw std::runtime_error("Failed to create a window");

	renderer = UniqueRenderer(
	    SDL_CreateRenderer(window.get(), -1, SDL_RENDERER_ACCELERATED));
	if (!renderer)
		throw std::runtime_error("Failed to create a renderer");

	texture = UniqueTexture(SDL_CreateTexture(renderer.get(),
	    SDL_PIXELFORMAT_RGBA8888,
	    SDL_TEXTUREACCESS_STREAMING,
	    width,
	    height));
	if (!texture)
		throw std::runtime_error("Failed to create a texture");
}

AppWindow::~AppWindow()
{
	SDL_Quit();
}

bool AppWindow::is_running()
{
	return running;
}

void AppWindow::handle_events()
{
	SDL_Event e;

	while (SDL_PollEvent(&e)) {
		if (e.type == SDL_QUIT)
			running = false;
	}
}

void AppWindow::update()
{
	unsigned stride = 4 * width;

	SDL_UpdateTexture(texture.get(), nullptr, buf.data(), stride);
	SDL_RenderClear(renderer.get());
	SDL_RenderCopy(renderer.get(), texture.get(), nullptr, nullptr);
	SDL_RenderPresent(renderer.get());
}

} // namespace host
