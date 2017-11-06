
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

#ifndef PRIMITIVES_H
#define PRIMITIVES_H

// Application specific headers.

// Cuda specific headers.
#include <cuda.h>
#include <helper_math.h>
#include <vector_types.h>
#include <vector_functions.h>

// Standard c++ headers.
#include <cfloat>

namespace renderbox2
{
	//
	// Triangle - basic modeling primitive used.
	//

	struct Triangle
	{
		float3 m_v[3];
		float3 m_n[3];
		float2 m_uv[3];
	};


	//
	// AABB struct - basic bounding primitive used.
	//

	struct AABB
	{
		float3 m_min;
		float3 m_max;


		__host__ __device__ AABB()
		{
			m_min = make_float3(FLT_MAX);
			m_max = make_float3(-FLT_MAX);
		}

		__host__ __device__ AABB(const float3& bmin, const float3& bmax)
		{
			m_min = bmin;
			m_max = bmax;
		}

		__host__ __device__ AABB(const AABB& A)
		{
			m_min = A.m_min;
			m_max = A.m_max;
		}

		
		//
		// Functions on the AABB.
		//

		__host__ __device__ bool valid() const
		{
			return (m_min.x <= m_max.x && m_min.y <= m_max.y && m_min.z <= m_max.z);
		}

		__host__ __device__ float3 midpt() const
		{
			return (m_min + m_max) * 0.5f;
		}		

		__host__ __device__ void insert(const float3& pt)
		{
			m_min = fminf(m_min, pt);
			m_max = fmaxf(m_max, pt);
		}

		__host__ __device__ void grow(const AABB& A)
		{
			insert(A.m_min);
			insert(A.m_max);
		}

		__host__ __device__ void intersect(const AABB& A)
		{
			m_min = fmaxf(m_min, A.m_min);
			m_max = fminf(m_max, A.m_max);
		}

		__host__ __device__ float volume() const
		{
			if (!valid())
			{
				return 0.0f;
			}
			else
			{
				return (m_max.x - m_min.x) * (m_max.y - m_min.y) * (m_max.z - m_min.z);
			}
		}

		__host__ __device__ float area() const
		{
			if (!valid())
			{
				return 0.0f;
			}
			else
			{
				float3 d = m_max - m_min; 
				return (d.x*d.y + d.y*d.z + d.z*d.x)*2.0f;
			}
		}

		__host__ __device__ bool contains(const float3& pt) const
		{
			return (pt.x >= m_min.x && pt.x <= m_max.x && pt.y >= m_min.y && pt.y <= m_max.y && pt.z >= m_min.z && pt.z <= m_max.z);
		}
	};


	//
	// Ray struct.
	//

	struct Ray
	{

		__host__ __device__ Ray()
		{
			m_data[0] = m_data[1] = make_float4(0.0f);
		}

		__host__ __device__ Ray(const float3& origin, const float3& direction, float _tmin = 0.0f, float _tmax = FLT_MAX)
		{
			m_data[0].x = origin.x;		m_data[0].y = origin.y;		m_data[0].z = origin.z;		m_data[0].w = _tmin;
			m_data[1].x = direction.x;	m_data[1].y = direction.y;	m_data[1].z = direction.z;	m_data[1].w = _tmax;
		}

		__host__ __device__ Ray(const float4& o, const float4& d)
		{
			m_data[0] = o;
			m_data[1] = d;
		}

		__host__ __device__ Ray(const Ray& R)
		{
			m_data[0] = R.m_data[0];
			m_data[1] = R.m_data[1];
		}
		
		__host__ __device__ __inline__ float3 operator() (float t) const
		{
			return origin() + direction() * t;
		}

		__host__ __device__ __inline__ float3 origin() const
		{
			return make_float3(m_data[0].x, m_data[0].y, m_data[0].z);
		}

		__host__ __device__ __inline__ float3 direction() const
		{
			return make_float3(m_data[1].x, m_data[1].y, m_data[1].z);
		}

		__host__ __device__ __inline__ float tmin() const
		{
			return m_data[0].w;
		}

		__host__ __device__ __inline__ float tmax() const
		{
			return m_data[1].w;
		}

		__host__ __device__ __inline__ void set_tmin(const float tmin)
		{
			m_data[0].w = tmin;
		}

		__host__ __device__ __inline__ void set_tmax(const float tmax)
		{
			m_data[1].w = tmax;
		}

	private:
		float4 m_data[2];	
	};


	//
	// Frame struct used for moving in and out of different coordinate frames. Extremely simplistic.
	//

	struct Frame
	{
		__host__ __device__ Frame()
		{
			mX = make_float3(1.0f, 0.0f, 0.0f);
			mY = make_float3(0.0f, 1.0f, 0.0f);
			mZ = make_float3(0.0f, 0.0f, 1.0f);
		}

		__host__ __device__ Frame(const float3& X, const float3& Y, const float3& Z)
		{
			mX = X;
			mY = Y;
			mZ = Z;
		}

		__host__ __device__ void set_from_z(const float3& z)
		{
			float3 tmpZ = mZ = normalize(z);
			float3 tmpX = (abs(tmpZ.x) > 0.99f) ? make_float3(0, 1, 0) : make_float3(1, 0, 0);
			mY = normalize(cross(tmpZ, tmpX));
			mX = cross(mY, tmpZ);
		}

		__host__ __device__ __inline__ float3 to_world(const float3& w) const { return mX * w.x + mY * w.y + mZ * w.z; }
		__host__ __device__ __inline__ float3 to_local(const float3& w) const { return make_float3(dot(w, mX), dot(w, mY), dot(w, mZ)); }

		__host__ __device__ float3 binormal() const { return mX; }
		__host__ __device__ float3 tangent()  const { return mY; }
		__host__ __device__ float3 normal()   const { return mZ; }

		float3 mX, mY, mZ;
	};


	//
	// Free functions that operaton on primitives.
	//

	__host__ __device__ __inline__ float distance(const float3& v0, const float3& v1)
	{
		float3 vec = v1 - v0;
		return sqrtf(dot(vec, vec));
	}

	__host__ __device__ __inline__ float distance2(const float3& v0, const float3& v1)
	{
		float3 vec = v1 - v0;
		return dot(vec, vec);
	}

	// Assumes a lhs coordinate system and a clockwise ordering of vertices of triangle
	__host__ __device__ __inline__ float3 compute_normal(const float3& v0, const float3& v1, const float3& v2)
	{
		return normalize(cross(v1 - v0, v2 - v0));
	}

	// area of a triangle = 0.5 * |e1 * e2|
	__host__ __device__ __inline__ float compute_area(const float3& v0, const float3& v1, const float3& v2)
	{
		return 0.5f * length(compute_normal(v0, v1, v2));
	}


	//
	// Various ray - shape intersection routines.
	//

	__host__ __device__ bool intersect_ray_box(const AABB& box, const Ray& R, float2& thit);

	__host__ __device__ bool intersect_ray_triangle(const float3& v0, const float3& v1, const float3& v2, const Ray& R, float3& uvt);

}			// !namespace renderbox2

#endif		// !PRIMITIVES_H