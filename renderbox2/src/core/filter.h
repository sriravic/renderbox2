
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

#ifndef FILTER_H
#define FILTER_H

// Application specific headers.

// Cuda specific headers.
#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_math.h>

// Standard c++ headers.

namespace renderbox2
{

#define FILTER_TABLE_SIZE	16

	//
	// Filter classes implemented.
	//

	
	//
	// Base class for all filters implemented in renderbox2.
	//

	struct Filter
	{
		
		__host__ __device__ Filter(float xw, float yw)
			: m_xwidth(xw)
			, m_ywidth(yw)
			, m_inv_xwidth(1.0f / xw)
			, m_inv_ywidth(1.0f / yw)
		{
			// Empty constructor.
		}

		__host__ __device__ virtual float evaluate(float x, float y) const = 0;

		const float m_xwidth, m_ywidth;
		const float m_inv_xwidth, m_inv_ywidth;
	};


	//
	// Box Filter.
	//

	struct FilterBox : public Filter
	{
		__host__ __device__ FilterBox(float xw, float yw)
			: Filter(xw, yw)
		{
			// Empty constructor.
		}

		__host__ __device__ float evaluate(float x, float y) const
		{
			return 1.0f;
		}
	};


	//
	// Triangle Filter.
	//

	struct FilterTriangle : public Filter
	{
		__host__ __device__ FilterTriangle(float xw, float yw)
			: Filter(xw, yw)
		{
			// Empty constructor.
		}

		__host__ __device__ float evaluate(float x, float y) const
		{
			return fmaxf(0.0f, m_xwidth - fabsf(x)) * fmaxf(0.0f, m_ywidth - fabsf(y));				
		}
	};

	
	//
	// Gaussian Filter.
	// 

	struct FilterGaussian : public Filter
	{
		__host__ __device__ FilterGaussian(float xw, float yw, float a)
			: Filter(xw, yw)
			, alpha(a)
			, expX(expf(-alpha * m_xwidth * m_xwidth))
			, expY(expf(-alpha * m_ywidth * m_ywidth))
		{
			// Empty constructor.
		}

		__host__ __device__ float evaluate(float x, float y) const
		{
			return gaussian(x, expX) * gaussian(y, expY);
		}

	private:
		const float alpha;
		const float expX, expY;

		// Gaussian utility functions.
		__host__ __device__ float gaussian(float d, float expv) const
		{
			return fmaxf(0.0f, float(expf(-alpha * d * d) - expv));
		}
	};
}			// !namespace renderbox2

#endif		// !FILTER_H
