
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

#ifndef BSDF_H
#define BSDF_H

// Application specific headers.

// Cuda specific headers.
#include <cuda.h>
#include <vector_types.h>

// Standard c++ headers.

namespace renderbox2
{
	
	//
	// Various Bsdfs supported in renderbox2
	//

	enum BxDFType
	{
		BxDF_REFLECTION = 1 << 0,
		BxDF_TRANSMISSION = 1 << 1,
		BxDF_DIFFUSE = 1 << 2,
		BxDF_GLOSSY = 1 << 3,
		BxDF_SPECULAR = 1 << 4,
		BxDF_ALL_TYPES = BxDF_DIFFUSE | BxDF_GLOSSY | BxDF_SPECULAR,
		BxDF_ALL_REFLECTION = BxDF_REFLECTION | BxDF_ALL_TYPES,
		BxDF_ALL_TRANSMISSION = BxDF_TRANSMISSION | BxDF_ALL_TYPES,
		BxDF_ALL = BxDF_ALL_REFLECTION | BxDF_ALL_TRANSMISSION
	};


	//
	// Type of Individual bsdfs supported.
	//

	enum BsdfType
	{
		BSDF_LAMBERTIAN,
		BSDF_GLASS,
		BSDF_MIRROR,
		BSDF_PHONG,
		BSDF_OREN_NAYAR,
		BSDF_BLINN_MICROFACET,
	};

	
	//
	// Utility bsdf functions.
	//

	void compute_orennayar_AB(const float sigma, float2& AB);

};			// !namespace renderbox

#endif		// !BSDF_H
