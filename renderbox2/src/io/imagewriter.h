
/**
	renderbox2 - a physically based gpu renderer for research purposes
    Copyright (C) - 2014 - Srinath Ravichandran

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef IMAGE_WRITER_H
#define IMAGE_WRITER_H

// Application specific headers.

// Cuda specific headers.
#include <helper_math.h>

// Standard c++ headers.
#include <cstdint>
#include <string>

namespace renderbox2
{

	//
	// ImageWriter class writes output in a variety of file formats.
	// Supported file formats {EXR, JPEG}
	//

	class ImageWriter
	{
	public:

		void write(const std::string& filename, float* pixels, uint32_t xres, uint32_t yres, uint32_t total_xres, uint32_t total_yres, uint32_t x_offset, uint32_t y_offset);

	private:

		int to_byte(float v) { return int(std::pow((float)clamp(v, 0.0f, 1.0f), (float)1.0f / 2.2f) * 255 + .5); }

		void write_exr(const std::string& filename, float* pixels, uint32_t xres, uint32_t yres, uint32_t total_xres, uint32_t total_yres, uint32_t x_offset, uint32_t y_offset);

		void write_jpeg(const std::string& filename, float* pixels, uint32_t xres, uint32_t yres);
	};

}			// !namespace renderbox2

#endif		// !IMAGE_WRITER_H
