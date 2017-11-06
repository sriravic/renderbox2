
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
#include <core/transform.h>
#include <core/util.h>

// Cuda specific headers.

// Standard c++ headers.

namespace renderbox2
{
	__host__ __device__ Transform scale(float sx, float sy, float sz) {
		Matrix4x4 ret;
		ret(0, 0) = sx;
		ret(1, 1) = sy;
		ret(2, 2) = sz;
		ret(3, 3) = 1.0f;
		return Transform(ret);
	}

	__host__ __device__ Transform translate(float x, float y, float z) {
		// we assume column vectors
		// so last row is the translate column
		Matrix4x4 ret;
		ret(0, 3) = x;
		ret(1, 3) = y;
		ret(2, 3) = z;
		return Transform(ret);
	}

	__host__ __device__ Transform rotate_x(float theta) {
		Matrix4x4 ret;
		float sin_t = sinf(theta);
		float cos_t = cosf(theta);
		ret(1, 1) = cos_t; ret(1, 2) = -sin_t;
		ret(2, 1) = sin_t; ret(2, 2) = cos_t;
		return Transform(ret);
	}

	__host__ __device__ Transform rotate_y(float theta) {
		Matrix4x4 ret;
		float sin_t = sinf(theta);
		float cos_t = cosf(theta);
		ret(0, 0) = cos_t; ret(0, 2) = sin_t;
		ret(2, 0) = -sin_t; ret(2, 2) = cos_t;
		return Transform(ret);
	}

	__host__ __device__ Transform rotate_z(float theta) {
		Matrix4x4 ret;
		float sin_t = sinf(theta);
		float cos_t = cosf(theta);
		ret(0, 0) = cos_t; ret(0, 1) = -sin_t;
		ret(1, 0) = sin_t; ret(1, 1) = cos_t;
		return Transform(ret);
	}


	//
	// Compute the lookat transform used in perspective projection.
	//

	__host__ __device__ Transform lookat(const float3& pos, const float3& lookat, const float3& up)
	{
		float m[4][4];
		m[0][3] = pos.x;
		m[1][3] = pos.y;
		m[2][3] = pos.z;
		m[3][3] = 1.0f;

		float3 dir = normalize(lookat - pos);
		float3 left = normalize(cross(normalize(up), dir));
		float3 new_up = cross(dir, left);

		m[0][0] = left.x;
		m[1][0] = left.y;
		m[2][0] = left.z;
		m[3][0] = 0.0f;

		m[0][1] = new_up.x;
		m[1][1] = new_up.y;
		m[2][1] = new_up.z;
		m[3][1] = 0.0f;

		m[0][2] = dir.x;
		m[1][2] = dir.y;
		m[2][2] = dir.z;
		m[3][2] = 0.0f;

		Matrix4x4 camToWorld(m);
		return Transform(camToWorld.invert(), camToWorld);
	}


	//
	// Computes the perspective camera transformation matrix
	//

	__host__ __device__ Transform perspective(float fov, float n, float f)
	{
		Matrix4x4 persp = Matrix4x4(1.0f, 0.0f, 0.0f, 0.0f,
									0.0f, 1.0f, 0.0f, 0.0f,
									0.0f, 0.0f, f / (f - n), -f * n / (f - n),
									0.0f, 0.0f, 1.0f, 0.0f);

		float inv_tan_ang = 1.0f / tanf(to_radians(fov) / 2.0f);
		return scale(inv_tan_ang, inv_tan_ang, 1.0f) * Transform(persp);
	}

}
