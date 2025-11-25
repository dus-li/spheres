// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Dus'li

#pragma once

#include <cstddef>

#include "host/stb_truetype.h"

#include "types.hxx"

namespace host {

class FontSingleton {
  public:
	static constexpr char FIRST_CHAR = 32;
	static constexpr char CHAR_COUNT = 96;

  private:
	stbtt_bakedchar data[CHAR_COUNT];
	u8             *bitmap_buf = nullptr;
	size_t          width      = 512;
	size_t          height     = 512;
	float           px_height  = 20;

	FontSingleton()
	{
		extern const u8 raw_font[];

		bitmap_buf = new u8[width * height];
		stbtt_BakeFontBitmap(raw_font,
		    0,
		    px_height,
		    bitmap_buf,
		    width,
		    height,
		    FIRST_CHAR,
		    CHAR_COUNT,
		    data);
	}

	~FontSingleton() { delete[] bitmap_buf; }

  public:
	static FontSingleton &get()
	{
		static FontSingleton instance;
		return instance;
	}

	FontSingleton(const FontSingleton &)            = delete;
	FontSingleton &operator=(const FontSingleton &) = delete;

	const stbtt_bakedchar *glyphs() const;
	const u8              *bitmap() const;
	size_t                 atlas_width() const;
	size_t                 atlas_height() const;
	float                  pixel_height() const;
	bool                   char_supported(char c) const;
};

} // namespace host
