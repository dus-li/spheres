// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Dus'li

#include "host/font.hxx"

namespace host {

const u8 raw_font[] = {
#embed "../../resources/courier-prime.regular.ttf"
};

const stbtt_bakedchar *FontSingleton::glyphs() const
{
	return data;
}

const u8 *FontSingleton::bitmap() const
{
	return bitmap_buf;
}

size_t FontSingleton::atlas_width() const
{
	return width;
}
size_t FontSingleton::atlas_height() const
{
	return height;
}
float FontSingleton::pixel_height() const
{
	return px_height;
}

bool FontSingleton::char_supported(char c) const
{
	return c >= FIRST_CHAR && c < FIRST_CHAR + CHAR_COUNT;
}

} // namespace host
