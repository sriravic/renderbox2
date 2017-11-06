
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

#ifndef FILM_H
#define FILM_H

// Application specific headers.
#include <core/filter.h>
#include <memory/memoryallocator.h>

// Cuda specific headers.
#include <cuda_runtime.h>
#include <vector_types.h>

// Standard c++ headers.
#include <cstdint>
#include <string>

namespace renderbox2
{

	//
	// Internal structure to represent pixels on the film.
	//

	struct Pixel
	{
		__host__ __device__ Pixel()
		{
			for (uint32_t i = 0; i < 3; i++)
			{
				Lrgb[i] = splatRGB[i] = 0.0f;
				weightsum = 0.0f;
			}
		}

		float Lrgb[3];
		float weightsum;
		float splatRGB[3];
		float pad; 
	};


	//
	// Film class stores pixel data in device memory.
	//

	class Film
	{
	public:

		Film(uint32_t xres, uint32_t yres, Filter* filter, MemoryAllocator& allocator);

		~Film()
		{
			m_allocator.free(m_pixels);
			m_allocator.free(m_filter_table);
		}

		uint32_t get_width() const { return m_xres; }
		uint32_t get_height() const { return m_yres; }

		void write_image(const std::string& filename);

		DevicePointer get_pixels() { return m_pixels; }
		DevicePointer get_filter_table() { return m_filter_table; }
		Filter*		  get_filter() { return m_filter; }

	private:
		
		DevicePointer		m_pixels;
		DevicePointer		m_filter_table;
		const uint32_t		m_xres;
		const uint32_t		m_yres;
		MemoryAllocator&	m_allocator;
		Filter*				m_filter;
	};

}			// !namespace renderbox2

#endif		// !FILM_H
