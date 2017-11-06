
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

#ifndef MONTE_CARLO_H
#define MONTE_CARLO_H

// Application specific headers.
#include <core/defs.h>

// Cuda specific headers.
#include <cuda.h>
#include <helper_math.h>
#include <vector_types.h>

// Standard c++ headers.

namespace renderbox2
{

	//
	// This file contains all the sampling routines used within renderbox2
	// NOTE: the random numbers provided to these routines are assumed to be well chosen.
	//


	//
	// Disc Sampling Routines.
	//

	__device__ __host__ float2 rejection_sampling_disk(int seed);
	
	__device__ __host__ float2 uniform_sample_disk(float u1, float u2);
	
	__device__ __host__ float2 concentric_sample_disk(float u1, float u2);


	//
	// Sphere and Hemisphere sampling routines.
	//

	__device__ __host__ float3 uniform_sample_hemisphere(float u1, float u2);
	
	__device__ __host__ float3 uniform_sample_sphere(float u1, float u2);
	
	__device__ __host__ float3 cosine_sample_hemisphere_normal(float u1, float u2, float3& normal);
	
	__device__ __host__ float3 cosine_sample_hemisphere(float u1, float u2);

	// Pdf with respect to total solid angle.
	__device__ __host__ __inline__ float uniform_hemisphere_pdf()				{ return INV_TWOPI; }
	
	__device__ __host__ __inline__ float uniform_sphere_pdf()					{ return INV_FOURPI; }
	
	__device__ __host__ __inline__ float cosine_hemisphere_pdf(float costheta)	{ return costheta * INV_PI; }


	//
	// Triangle sampling routines.
	//

	__device__ __host__ float3 uniform_sample_triangle(const float3& v0, const float3& v1, const float3& v2, float u1, float u2);


	// Given a point on a surface and direction towards a triangle, this method computes the probability of generating a ray that intersects with the triangle.
	// Note: We inherently assume the point and the triangle are visible to each other.
	// The probability is with respect to solid angle of the triangle subtended at the point.
	__device__ __host__ float sample_triangle_pdf(const float3& pt, const float3& direction, const float3& v0, const float3& v1, const float3& v2);


	//
	// Quad sampling routines.
	//

	//
	// MIS computations.
	//

	__device__ __host__ __inline__ float balanced_heuristic(int nf, float fPdf, int ng, float gPdf) { return (nf * fPdf) / (nf * fPdf + ng * gPdf); }
	
	__device__ __host__ __inline__ float power_heuristic(int nf, float fPdf, int ng, float gPdf)
	{
		float f = nf * fPdf;
		float g = ng * gPdf;
		return f*f / (f*f + g*g);
	}


};				// !namespace renderbox2

#endif			// !MONTE_CARLO_H