
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

#ifndef UTIL_H
#define UTIL_H

// Application specific headers.
#include <core/defs.h>

// Cuda specific headers.
#include <cuda_runtime.h>

// Standard c++ headers.
#include <cmath>
#include <cstdint>

namespace renderbox2
{
	
	//
	// Util radian - degree conversion functions.
	//

	__inline__ __host__ __device__ float to_radians(float deg) {
		return ((float)M_PI / 180.f) * deg;
	}

	__inline__ __host__ __device__ float to_degrees(float rad) {
		return (180.f / (float)M_PI) * rad;
	}


	//
	// Cuda vector primitives get methods.
	//

	__inline__ __host__ __device__ float get(const float2& V, int dim)
	{
		assert(dim >= 0 && dim <= 1);
		if (dim == 0) return V.x;
		else return V.y;
	}

	__inline__ __host__ __device__ float get(const float3& V, int dim)
	{
		assert(dim >= 0 && dim <= 2);
		if (dim == 0) return V.x;
		else if (dim == 1) return V.y;
		else return V.z;
	}

	__inline__ __host__ __device__ float& get(float3& V, int dim)
	{
		assert(dim >= 0 && dim < 3);
		if (dim == 0) return V.x;
		else if (dim == 1) return V.y;
		else return V.z;
	}

	__inline__ __host__ __device__ float get(const float4& V, int dim)
	{
		assert(dim >= 0 && dim <= 3);
		if (dim == 0) return V.x;
		else if (dim == 1) return V.y;
		else if (dim == 2) return V.z;
		else return V.w;
	}

	__inline__ __host__ __device__ int get(const int3& V, int dim)
	{
		assert(dim >= 0 && dim < 3);
		if (dim == 0) return V.x;
		else if (dim == 1) return V.y;
		else return V.z;
	}

	__inline__ __host__ __device__ uint get(const uint3& V, uint dim)
	{
		assert(dim < 3);
		if (dim == 0) return V.x;
		else if (dim == 1) return V.y;
		else return V.z;
	}

	__inline__ __host__ __device__ float get_min(const float3& V)
	{
		float ret = V.x;
		if (V.y < ret) ret = V.y;
		if (V.z < ret) ret = V.z;
		return ret;
	}

	__inline__ __host__ __device__ float get_max(const float3& V)
	{
		float ret = V.x;
		if (V.y > ret) ret = V.y;
		if (V.z > ret) ret = V.z;
		return ret;
	}

	__inline__ __host__ __device__ float get_sum(const float3& V)
	{
		return V.x + V.y + V.z;
	}

	
	//
	// Other util methods.
	//
	
	template<typename T>
	__inline__ __host__ __device__ T sqr(T val) { return val*val; }

	__inline__ __device__ __host__ uint32_t next_power_2(uint32_t val)
	{
		return 1 << (1 + static_cast<uint32_t>(floorf(log2f(static_cast<float>(val)))));
	}


	//
	// Some hashing functions.
	//

	__inline__ __device__ uint32_t simplehash(int32_t a)
	{
		a = (a + 0x7ed55d16) + (a << 12);
		a = (a ^ 0xc761c23c) ^ (a >> 19);
		a = (a + 0x165667b1) + (a << 5);
		a = (a + 0xd3a2646c) ^ (a << 9);
		a = (a + 0xfd7046c5) + (a << 3);
		a = (a ^ 0xb55a4f09) ^ (a >> 16);
		return a;
	}

	__inline__ __device__ uint32_t wang_hash(uint32_t seed)
	{
		seed = (seed ^ 61) ^ (seed >> 16);
		seed *= 9;
		seed = seed ^ (seed >> 4);
		seed *= 0x27d4eb2d;
		seed = seed ^ (seed >> 15);
		return seed;
	}

	__inline__ __host__ __device__ float luminance(const float3& color)
	{
		return 0.212671f * color.x + 0.715160f * color.y + 0.072169f * color.z;
	}

	__inline__ __host__ __device__ float luminance(const float4& color)
	{
		return 0.212671f * color.x + 0.715160f * color.y + 0.072169f * color.z;
	}

	__inline__ __host__ __device__ bool is_black(const float3& color)
	{
		return (color.x == 0.0f && color.y == 0.0f && color.z == 0.0f);
	}

	__inline__ __host__ __device__ bool is_black(const float4& color)
	{
		return (color.x == 0.0f && color.y == 0.0f && color.z == 0.0f);
	}


	//
	// Simple structure that helps in generating visibility rays.
	//

	struct VisibilityTesterElement
	{
		__host__ __device__ void set_segment(const float3& p0, const float pEpsilon1, const float3& p1, const float pEpsilon2)
		{
			float dist = distance(p0, p1);
			m_visibility_ray = Ray(p0, normalize(p1 - p0), pEpsilon1, dist * (1.0f - pEpsilon2));
		}

		Ray m_visibility_ray;
	};

}			// !namespace renderbox2

#endif		// !UTIL_H
