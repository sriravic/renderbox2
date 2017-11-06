
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

#ifndef DIFF_GEOM_H
#define DIFF_GEOM_H

// Application specific headers.

// Cuda specific headers.
#include <vector_types.h>

// Standard c++ headers.

namespace renderbox2
{
	
	//
	// Differential geometry structure representing the geometry at intersections.
	//

	struct DifferentialGeometry
	{
		float3 p;
		float3 normal;
		float2 uv;
		float3 dpdu, dpdv;
		float3 dndu, dndv;
		float3 dpdx, dpdy;
		float  dudx, dudy, dvdx, dvdy;
	};

}			// !namespace renderbox2

#endif		// !DIFF_GEOM_H