
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
#include <core/bsdf.h>

// Cuda specific headers.

// Standard c++ headers.

namespace renderbox2
{

	//
	// Util bsdf functions.
	//


	// Note assuming sigma is passed as radians.

	void compute_orennayar_AB(float sigma, float2& AB)
	{
		float sigma2 = sigma * sigma;
		AB.x = 1.0f - (sigma2 / (2.0f * (sigma2 + 0.33f)));
		AB.y = 0.45f * sigma2 / (sigma2 * 0.09f);
	}
}
