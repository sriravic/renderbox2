
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
#include <core/primitives.h>

// Cuda specific headers.

// Standard c++ headers.

namespace renderbox2
{
	__device__ __host__ bool intersect_ray_box(const AABB& box, const Ray& R, float2& thit)
	{
		float3 minTs = (box.m_min - R.origin()) / R.direction();
		float3 maxTs = (box.m_max - R.origin()) / R.direction();

		float nearT = fminf(minTs.x, maxTs.x);
		nearT = fmaxf(nearT, fminf(minTs.y, maxTs.y));
		nearT = fmaxf(nearT, fminf(minTs.z, maxTs.z));

		float farT = fmaxf(minTs.x, maxTs.x);
		farT = fminf(farT, fmaxf(minTs.y, maxTs.y));
		farT = fminf(farT, fmaxf(minTs.z, maxTs.z));

		// condition to have ray inside box test successful.
		nearT = R.tmin() > nearT ? R.tmin() : nearT;
		farT = R.tmax() < farT ? R.tmax() : farT;

		thit = make_float2(nearT, farT);
		return nearT <= farT && 0 < farT;
	}

	__device__ __host__ bool intersect_ray_triangle(const float3& v0, const float3& v1, const float3& v2, const Ray& R, float3& uvt)
	{
		float3 edge1, edge2;
		float3 pvec, qvec, tvec;
		float det, inv_det;
		edge1 = v1 - v0;
		edge2 = v2 - v0;
		pvec = cross(R.direction(), edge2);
		det = dot(edge1, pvec);
		if (det < EPSILON && det > -EPSILON) return false;
		inv_det = (1.0f) / det;
		tvec = R.origin() - v0;
		uvt.x = dot(tvec, pvec) * inv_det;
		if (uvt.x < 0.0f || uvt.x > 1.0f) return false;
		qvec = cross(tvec, edge1);
		uvt.y = dot(R.direction(), qvec) * inv_det;
		if (uvt.y < 0.0f || uvt.x + uvt.y > 1.0f) return false;
		uvt.z = dot(edge2, qvec) * inv_det;
		return true;
	}
}
