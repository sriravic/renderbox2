
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
#include <core/defs.h>
#include <core/film.h>
#include <io/imagewriter.h>

// Cuda specific headers.

// Standard c++ headers.

namespace renderbox2
{

	//
	// Film class implementation.
	//

	Film::Film(uint32_t width, uint32_t height, Filter* filter, MemoryAllocator& allocator)
		: m_allocator(allocator)
		, m_xres(width)
		, m_yres(height)
	{
		m_filter = filter;
		m_pixels = m_allocator.allocate(sizeof(Pixel) * m_xres * m_yres);
		m_filter_table = m_allocator.allocate(sizeof(float) * FILTER_TABLE_SIZE * FILTER_TABLE_SIZE);

		checkCuda(cudaMemset(m_pixels.m_ptr, 0, sizeof(Pixel) * m_xres * m_yres));

		// Precompute filter weights and copy them into device memory.
		float* filter_table = new float[FILTER_TABLE_SIZE * FILTER_TABLE_SIZE];

		int idx = 0;
		for (int y = 0; y < FILTER_TABLE_SIZE; ++y)
		{
			float fy = ((float)y + 0.5f) * filter->m_ywidth / FILTER_TABLE_SIZE;
			for (int x = 0; x < FILTER_TABLE_SIZE; ++x)
			{
				float fx = ((float)y + 0.5f) * filter->m_xwidth / FILTER_TABLE_SIZE;
				filter_table[idx++] = filter->evaluate(fx, fy);
			}
		}

		checkCuda(cudaMemcpy(m_filter_table.m_ptr, filter_table, sizeof(float) * FILTER_TABLE_SIZE * FILTER_TABLE_SIZE, cudaMemcpyHostToDevice));
	}

	void Film::write_image(const std::string& filename)
	{
		// Copy the data from gpu memory into device memory.
		Pixel* pixel_data = new Pixel[m_xres * m_yres];
		float* data = new float[m_xres * m_yres * 4];

		checkCuda(cudaMemcpy(pixel_data, m_pixels.m_ptr, sizeof(Pixel) * m_xres * m_yres, cudaMemcpyDeviceToHost));

		for (uint32_t idx = 0; idx < m_xres * m_yres; idx++)
		{
			// strip data and store;
			const Pixel& pixel = pixel_data[idx];
			float weight_sum = pixel.weightsum;
			if (weight_sum != 0.0f)
			{
				float inv_weight = 1.0f / weight_sum;
				data[idx * 4 + 0] = fmax(0.0f, pixel.Lrgb[0] * inv_weight);
				data[idx * 4 + 1] = fmax(0.0f, pixel.Lrgb[1] * inv_weight);
				data[idx * 4 + 2] = fmax(0.0f, pixel.Lrgb[2] * inv_weight);
				data[idx * 4 + 3] = 0.0f;
			}
		}

		// We need to extract the xyz values alone from the pixel buffer and then pass it to the image writer.
		ImageWriter iw;
		iw.write(filename, data, m_xres, m_yres, 0, 0, 0, 0);

		SAFE_RELEASE_ARRAY(pixel_data);
		SAFE_RELEASE_ARRAY(data);
	}
}
