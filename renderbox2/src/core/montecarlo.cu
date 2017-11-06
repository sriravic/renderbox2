
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
#include <core/primitives.h>
#include <core/montecarlo.h>

// Cuda specific headers.

// Standard c++ headers.
#include <thrust/random.h>

namespace renderbox2
{

	//
	// Disc Sampling.
	//

	__device__ __host__ float2 rejection_sample_disk(int seed)
	{
		thrust::default_random_engine rng(seed);
		thrust::uniform_real_distribution<float> u01(0, 1);
		float sx, sy;
		do
		{
			sx = u01(rng);
			sy = u01(rng);
		} while (sx * sx + sy * sy > 1.0f);

		return make_float2(sx, sy);
	}

	__device__ __host__ float2 uniform_sample_disk(float u1, float u2)
	{
		float r = u1;
		float phi = 2 * M_PI * u2;
		float x = r * cos(phi);
		float y = r * sin(phi);
		return make_float2(x, y);
	}

	__device__ __host__ float2 concentric_sample_disk(float u1, float u2)
	{
		float r, theta;
		float sx = 1.0f - 2.0f * u1;
		float sy = 1.0f - 2.0f * u2;

		// Handle degeneracy at the origin.
		if (sx == 0.0 && sy == 0.0)
		{
			return make_float2(0.0f, 0.0f);
		}
		if (sx >= -sy)
		{
			if (sx > sy)
			{
				// Handle first region of disk.
				r = sx;
				if (sy > 0.0) theta = sy / r;
				else          theta = 8.0f + sy / r;
			}
			else
			{
				// Handle second region of disk.
				r = sy;
				theta = 2.0f - sx / r;
			}
		}
		else
		{
			if (sx <= sy)
			{
				// Handle third region of disk.
				r = -sx;
				theta = 4.0f - sy / r;
			}
			else
			{
				// Handle fourth region of disk.
				r = -sy;
				theta = 6.0f + sx / r;
			}
		}

		theta *= M_PI / 4.f;
		float x = r * cosf(theta);
		float y = r * sinf(theta);
		return make_float2(x, y);
	}


	//
	// Sphere Sampling Routines.
	//

	__device__ __host__ float3 uniform_sample_hemisphere(float u1, float u2)
	{
		float z = u1;
		float r = sqrtf(max(0.0f, 1.0f - z*z));
		float phi = 2 * M_PI * u2;
		float x = r * cos(phi);
		float y = r * sin(phi);
		return make_float3(x, y, z);
	}

	__device__ __host__ float3 uniform_sample_sphere(float u1, float u2)
	{
		float z = 1.0f - 2.0f * u1;
		float r = sqrtf(max(0.0f, 1.0f - z*z));
		float phi = 2 * M_PI * u1;
		float x = r * cos(phi);
		float y = r * sin(phi);
		return make_float3(x, y, z);
	}

	__device__ __host__ float3 cosine_sample_hemisphere_normal(float u1, float u2, float3& normal)
	{
		float phi = 2 * M_PI * u1;
		float theta = acos(sqrtf(1.f - u2));
		float3 Z = normal;
		float3 perp;

		if (Z.x < Z.y && Z.x < Z.z)
			perp = make_float3(1.0f, 0.0f, 0.0f);
		else if (Z.y < Z.x && Z.y < Z.z)
			perp = make_float3(0.0f, 1.0f, 0.0f);
		else
			perp = make_float3(0.0f, 0.0f, 1.0f);

		float3 X = normalize(cross(normal, perp));
		float3 Y = normalize(cross(normal, X));
		float3 ret = X * cos(phi) * sin(theta) + Y * sin(phi) * sin(theta) + Z * cos(phi);
		return ret;
	}

	__device__ __host__ float3 cosine_sample_hemisphere(float u1, float u2)
	{
		float2 xy = concentric_sample_disk(u1, u2);
		float z = sqrtf(max(0.0f, 1.0f - xy.x*xy.x - xy.y*xy.y));
		return make_float3(xy, z);
	}


	//
	// Triangle Sampling Routines.
	//

	__device__ __host__ float3 uniform_sample_triangle(const float3& v0, const float3& v1, const float3& v2, float u1, float u2)
	{
		float su1 = sqrtf(u1);
		float u = 1.0f - su1;
		float v = u2 * su1;
		float3 p = v0 * u + v1 * v + v2 * (1.0f - u - v);
		return p;
	}

	__device__ __host__ float sample_triangle_pdf(const float3& pt, const float3& direction, const float3& v0, const float3& v1, const float3& v2)
	{
		Ray r(pt, direction, 1.0e-3f);
		float3 uvt;

		if (!intersect_ray_triangle(v0, v1, v2, r, uvt)) return 0.0f;

		float r2 = distance2(pt, r(uvt.z));
		float costheta_there = fabsf(dot(compute_normal(v0, v1, v2), make_float3(-direction.x, -direction.y, -direction.z)));
		float area = compute_area(v0, v1, v2);

		// Solid angle subtended at point = dA * costheta_there / r2;
		// probability  = 1/Solid angle
		// r2 / (area * costheta)
		return r2 / (area * costheta_there);
	}
}
