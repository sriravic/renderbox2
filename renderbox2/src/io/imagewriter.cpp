
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

// Application specific headers.
#include <3rdparty/jpeg/jpeg.h>
#include <3rdparty/miniexr/miniexr.h>
#include <core/defs.h>
#include <io/imagewriter.h>

// Cuda specific headers.

// Standard c++ headers.
#include <iostream>
#include <stdlib.h>

namespace renderbox2
{
	void ImageWriter::write(const std::string& filename, float* pixels, uint32_t xres, uint32_t yres, uint32_t total_xres, uint32_t total_yres, uint32_t xoffset, uint32_t yoffset)
	{
		if (filename.size() >= 5)
		{
			size_t suffix_offset = filename.size() - 4;
			if (!strcmp(filename.c_str() + suffix_offset, ".exr") ||
				!strcmp(filename.c_str() + suffix_offset, ".EXR"))
			{
				write_exr(filename, pixels, xres, yres, total_xres, total_yres, xoffset, yoffset);
			}
			else if (!strcmp(filename.c_str() + suffix_offset, ".jpg") ||
					 !strcmp(filename.c_str() + suffix_offset, ".JPG"))
			{
				write_jpeg(filename, pixels, xres, yres);
			}
			else if (!strcmp(filename.c_str() + suffix_offset, ".png") ||
				!strcmp(filename.c_str() + suffix_offset, ".PNG"))
			{

			}
		}
		else
		{
			std::cerr << "Error : Unable to determine filetype from suffix in filename." << std::endl;
			return;
		}
	}


	//
	// All image writing utilities.
	//

	void ImageWriter::write_exr(const std::string& filename, float* pixels, uint32_t xres, uint32_t yres, uint32_t total_xres, uint32_t total_yres, uint32_t xoffset, uint32_t yoffset)
	{
		// convert all float pixels to half pixels.
		uint16_t *short_pixels = new uint16_t[xres * yres * 4];
		size_t outsize;

		for (uint32_t idx = 0; idx < xres * yres * 4; idx++)
		{
			short_pixels[idx] = FloatToHalf(pixels[idx]);
		}

		miniexr_write(xres, yres, 4, short_pixels, &outsize);
		FILE* file;
		
		errno_t error = fopen_s(&file, filename.c_str(), "wb");
		fwrite(short_pixels, 1, outsize, file);
		fclose(file);

		SAFE_RELEASE_ARRAY(short_pixels);
	}

	void ImageWriter::write_jpeg(const std::string& filename, float* pixels, uint32_t xres, uint32_t yres)
	{
		unsigned char *output = new unsigned char[xres * yres * 4];
		char r, g, b, x = 0;
		uint index = 0;
		for (uint i = 0; i < xres * yres; i++) {
			r = to_byte(pixels[i * 4 + 0]);
			g = to_byte(pixels[i * 4 + 1]);
			b = to_byte(pixels[i * 4 + 2]);
			output[index++] = r;
			output[index++] = g;
			output[index++] = b;
			output[index++] = x;
		}
		std::cout << "\nDone Converting. Writing to file\n";
		jo_write_jpg(filename.c_str(), output, xres, yres, 100);
		SAFE_RELEASE_ARRAY(output);
	}
}
