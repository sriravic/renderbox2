
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

#ifndef TRANSFORM_H
#define TRANSFORM_H

// Application specific headers.
#include <core/defs.h>
#include <core/primitives.h>

// Cuda specific headers.
#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_math.h>

// Standard c++ headers.
#include <cassert>

#ifndef MATRIX
#define MATRIX(element) m[element/4][element%4]
#endif

namespace renderbox2
{

	//
	// Matrix class.
	//

	class Matrix4x4
	{
	public:

		__host__ __device__ Matrix4x4()
		{
			for (int i = 0; i < 4; i++)
			{
				for (int j = 0; j < 4; j++)
				{
					if (i == j) m[i][j] = 1.0f;
					else m[i][j] = 0.0f;
				}
			}
		}

		__host__ __device__ Matrix4x4(float _m00, float _m01, float _m02, float _m03,
									  float _m10, float _m11, float _m12, float _m13,
								      float _m20, float _m21, float _m22, float _m23,
									  float _m30, float _m31, float _m32, float _m33)
		{
			m[0][0] = _m00; m[0][1] = _m01; m[0][2] = _m02; m[0][3] = _m03;
			m[1][0] = _m10; m[1][1] = _m11; m[1][2] = _m12; m[1][3] = _m13;
			m[2][0] = _m20; m[2][1] = _m21; m[2][2] = _m22; m[2][3] = _m23;
			m[3][0] = _m30; m[3][1] = _m31; m[3][2] = _m32; m[3][3] = _m33;
		}

		__host__ __device__ Matrix4x4(const Matrix4x4& M)
		{
			for (int i = 0; i < 4; i++)
			{
				for (int j = 0; j < 4; j++)
				{
					m[i][j] = M.m[i][j];
				}
			}
		}

		__host__ __device__ Matrix4x4(float _m[4][4])
		{
			for (int i = 0; i < 4; i++)
			{
				for (int j = 0; j < 4; j++)
				{
					m[i][j] = _m[i][j];
				}
			}
		}


		//
		// get/set functions.
		//
		
		__host__ __device__ void set_col(int col, const float4& val)
		{
			assert(col >= 0 && col <= 3);
			m[0][col] = val.x;
			m[1][col] = val.y;
			m[2][col] = val.z;
			m[3][col] = val.w;
		}

		__host__ __device__ void set_row(int row, const float4& val)
		{
			assert(row >= 0 && row <= 3);
			m[row][0] = val.x;
			m[row][1] = val.y;
			m[row][2] = val.z;
			m[row][3] = val.w;
		}

		__host__ __device__ float4 get_row(int row) const
		{
			assert(row >= 0 && row <= 3);
			return make_float4(m[row][0], m[row][1], m[row][2], m[row][3]);
		}

		__host__ __device__ float4 get_col(int col) const
		{
			assert(col >= 0 && col <= 3);
			return make_float4(m[0][col], m[1][col], m[2][col], m[3][col]);
		}


		//
		// Matrix operator functions.
		//

		__host__ __device__ void operator= (const Matrix4x4& M)
		{
			for (int i = 0; i < 4; i++)
			{
				for (int j = 0; j < 4; j++)
				{
					m[i][j] = M.m[i][j];
				}
			}
		}

		__host__ __device__ float operator() (int r, int c) const
		{
			assert(r >= 0 && r <= 3 && c >= 0 && c <= 3);
			return m[r][c];
		}

		__host__ __device__ float& operator() (int r, int c)
		{
			assert(r >= 0 && r <= 3 && c >= 0 && c <= 3);
			return m[r][c];
		}

		__host__ __device__ Matrix4x4 transpose() const
		{
			Matrix4x4 ret;
			for (int row = 0; row < 4; row++) {
				for (int col = 0; col < 4; col++) {
					ret(row, col) = m[col][row];
				}
			}
			return ret;
		}

		__host__ __device__ Matrix4x4 operator* (const Matrix4x4& mat) const
		{
			Matrix4x4 ret;
			for (int i = 0; i < 4; i++)
			{
				for (int j = 0; j < 4; j++)
				{
					ret.m[i][j] = dot(get_row(i), mat.get_col(j));
				}
			}
			return ret;
		}

		__host__ __device__ float4 operator() (const float4& vector) const
		{
			float4 ret;
			ret.x = dot(get_row(0), vector);
			ret.y = dot(get_row(1), vector);
			ret.z = dot(get_row(2), vector);
			ret.w = dot(get_row(3), vector);
			return ret;
		}

		__host__ __device__ float3 operator() (const float3& vector, bool point) const
		{
			float4 ret = (*this)(make_float4(vector, float(int(point))));		// if a point, use a 1.0f at the end, else a vector has 0.0f as the last component
			if (point)
				ret /= ret.w;
			return make_float3(ret.x, ret.y, ret.z);
		}


		//
		// Matrix inverse computation.
		//

		__host__ __device__ Matrix4x4 invert() const
		{
			float inv[16], det;
			int i;

			inv[0] = MATRIX(5)  * MATRIX(10) * MATRIX(15) -
				MATRIX(5)  * MATRIX(11) * MATRIX(14) -
				MATRIX(9)  * MATRIX(6)  * MATRIX(15) +
				MATRIX(9)  * MATRIX(7)  * MATRIX(14) +
				MATRIX(13) * MATRIX(6)  * MATRIX(11) -
				MATRIX(13) * MATRIX(7)  * MATRIX(10);

			inv[4] = -MATRIX(4)  * MATRIX(10) * MATRIX(15) +
				MATRIX(4)  * MATRIX(11) * MATRIX(14) +
				MATRIX(8)  * MATRIX(6)  * MATRIX(15) -
				MATRIX(8)  * MATRIX(7)  * MATRIX(14) -
				MATRIX(12) * MATRIX(6)  * MATRIX(11) +
				MATRIX(12) * MATRIX(7)  * MATRIX(10);

			inv[8] = MATRIX(4)  * MATRIX(9) * MATRIX(15) -
				MATRIX(4)  * MATRIX(11) * MATRIX(13) -
				MATRIX(8)  * MATRIX(5) * MATRIX(15) +
				MATRIX(8)  * MATRIX(7) * MATRIX(13) +
				MATRIX(12) * MATRIX(5) * MATRIX(11) -
				MATRIX(12) * MATRIX(7) * MATRIX(9);

			inv[12] = -MATRIX(4)  * MATRIX(9) * MATRIX(14) +
				MATRIX(4)  * MATRIX(10) * MATRIX(13) +
				MATRIX(8)  * MATRIX(5) * MATRIX(14) -
				MATRIX(8)  * MATRIX(6) * MATRIX(13) -
				MATRIX(12) * MATRIX(5) * MATRIX(10) +
				MATRIX(12) * MATRIX(6) * MATRIX(9);

			inv[1] = -MATRIX(1)  * MATRIX(10) * MATRIX(15) +
				MATRIX(1)  * MATRIX(11) * MATRIX(14) +
				MATRIX(9)  * MATRIX(2) * MATRIX(15) -
				MATRIX(9)  * MATRIX(3) * MATRIX(14) -
				MATRIX(13) * MATRIX(2) * MATRIX(11) +
				MATRIX(13) * MATRIX(3) * MATRIX(10);

			inv[5] = MATRIX(0)  * MATRIX(10) * MATRIX(15) -
				MATRIX(0)  * MATRIX(11) * MATRIX(14) -
				MATRIX(8)  * MATRIX(2) * MATRIX(15) +
				MATRIX(8)  * MATRIX(3) * MATRIX(14) +
				MATRIX(12) * MATRIX(2) * MATRIX(11) -
				MATRIX(12) * MATRIX(3) * MATRIX(10);

			inv[9] = -MATRIX(0)  * MATRIX(9) * MATRIX(15) +
				MATRIX(0)  * MATRIX(11) * MATRIX(13) +
				MATRIX(8)  * MATRIX(1) * MATRIX(15) -
				MATRIX(8)  * MATRIX(3) * MATRIX(13) -
				MATRIX(12) * MATRIX(1) * MATRIX(11) +
				MATRIX(12) * MATRIX(3) * MATRIX(9);

			inv[13] = MATRIX(0)  * MATRIX(9) * MATRIX(14) -
				MATRIX(0)  * MATRIX(10) * MATRIX(13) -
				MATRIX(8)  * MATRIX(1) * MATRIX(14) +
				MATRIX(8)  * MATRIX(2) * MATRIX(13) +
				MATRIX(12) * MATRIX(1) * MATRIX(10) -
				MATRIX(12) * MATRIX(2) * MATRIX(9);

			inv[2] = MATRIX(1)  * MATRIX(6) * MATRIX(15) -
				MATRIX(1)  * MATRIX(7) * MATRIX(14) -
				MATRIX(5)  * MATRIX(2) * MATRIX(15) +
				MATRIX(5)  * MATRIX(3) * MATRIX(14) +
				MATRIX(13) * MATRIX(2) * MATRIX(7) -
				MATRIX(13) * MATRIX(3) * MATRIX(6);

			inv[6] = -MATRIX(0)  * MATRIX(6) * MATRIX(15) +
				MATRIX(0)  * MATRIX(7) * MATRIX(14) +
				MATRIX(4)  * MATRIX(2) * MATRIX(15) -
				MATRIX(4)  * MATRIX(3) * MATRIX(14) -
				MATRIX(12) * MATRIX(2) * MATRIX(7) +
				MATRIX(12) * MATRIX(3) * MATRIX(6);

			inv[10] = MATRIX(0)  * MATRIX(5) * MATRIX(15) -
				MATRIX(0)  * MATRIX(7) * MATRIX(13) -
				MATRIX(4)  * MATRIX(1) * MATRIX(15) +
				MATRIX(4)  * MATRIX(3) * MATRIX(13) +
				MATRIX(12) * MATRIX(1) * MATRIX(7) -
				MATRIX(12) * MATRIX(3) * MATRIX(5);

			inv[14] = -MATRIX(0)  * MATRIX(5) * MATRIX(14) +
				MATRIX(0)  * MATRIX(6) * MATRIX(13) +
				MATRIX(4)  * MATRIX(1) * MATRIX(14) -
				MATRIX(4)  * MATRIX(2) * MATRIX(13) -
				MATRIX(12) * MATRIX(1) * MATRIX(6) +
				MATRIX(12) * MATRIX(2) * MATRIX(5);

			inv[3] = -MATRIX(1) * MATRIX(6) * MATRIX(11) +
				MATRIX(1) * MATRIX(7) * MATRIX(10) +
				MATRIX(5) * MATRIX(2) * MATRIX(11) -
				MATRIX(5) * MATRIX(3) * MATRIX(10) -
				MATRIX(9) * MATRIX(2) * MATRIX(7) +
				MATRIX(9) * MATRIX(3) * MATRIX(6);

			inv[7] = MATRIX(0) * MATRIX(6) * MATRIX(11) -
				MATRIX(0) * MATRIX(7) * MATRIX(10) -
				MATRIX(4) * MATRIX(2) * MATRIX(11) +
				MATRIX(4) * MATRIX(3) * MATRIX(10) +
				MATRIX(8) * MATRIX(2) * MATRIX(7) -
				MATRIX(8) * MATRIX(3) * MATRIX(6);

			inv[11] = -MATRIX(0) * MATRIX(5) * MATRIX(11) +
				MATRIX(0) * MATRIX(7) * MATRIX(9) +
				MATRIX(4) * MATRIX(1) * MATRIX(11) -
				MATRIX(4) * MATRIX(3) * MATRIX(9) -
				MATRIX(8) * MATRIX(1) * MATRIX(7) +
				MATRIX(8) * MATRIX(3) * MATRIX(5);

			inv[15] = MATRIX(0) * MATRIX(5) * MATRIX(10) -
				MATRIX(0) * MATRIX(6) * MATRIX(9) -
				MATRIX(4) * MATRIX(1) * MATRIX(10) +
				MATRIX(4) * MATRIX(2) * MATRIX(9) +
				MATRIX(8) * MATRIX(1) * MATRIX(6) -
				MATRIX(8) * MATRIX(2) * MATRIX(5);

			det = MATRIX(0) * inv[0] + MATRIX(1) * inv[4] + MATRIX(2) * inv[8] + MATRIX(3) * inv[12];

			assert(det != 0.0f);

			det = 1.0f / det;

			Matrix4x4 ret;
			for (i = 0; i < 16; i++)
				ret(i / 4, i % 4) = inv[i] * det;

			return ret;
		}

	private:

		float m[4][4];
	};


	//
	// Transform class.
	//

	class Transform
	{
	public:

		__host__ __device__ Transform()
		{
			// Empty constructor.
		}

		__host__ __device__ Transform(float m[4][4])
		{
			m_matrix = Matrix4x4(m[0][0], m[0][1], m[0][2], m[0][3],
								 m[1][0], m[1][1], m[1][2], m[1][3],
								 m[2][0], m[2][1], m[2][2], m[2][3],
								 m[3][0], m[3][1], m[3][2], m[3][3]);

			m_inv_matrix = m_matrix.invert();
		}

		__host__ __device__ Transform(const Matrix4x4& matrix)
			: m_matrix(matrix)
			, m_inv_matrix(matrix.invert())
		{
			// Do nothing.
		}

		__host__ __device__ Transform(const Matrix4x4& matrix, const Matrix4x4& inv_matrix)
			: m_matrix(matrix)
			, m_inv_matrix(inv_matrix)
		{
			// Do nothing.
		}


		//
		// Transform class functions.
		// 

		__host__ __device__ Transform inverse()
		{
			return Transform(m_inv_matrix, m_matrix);
		}

		__host__ __device__ Transform transpose()
		{
			return Transform(m_matrix.transpose(), m_inv_matrix.transpose());
		}


		//
		// Util methods.
		//

		__host__ __device__ const Matrix4x4 get_matrix() const { return m_matrix; }

		__host__ __device__ const Matrix4x4 get_inv_matrix() const { return m_inv_matrix; }


		//
		// Some operators.
		//

		__host__ __device__ Transform operator* (const Transform& T) const
		{
			Matrix4x4 matrix = m_matrix * T.get_matrix();
			Matrix4x4 inv_matrix = T.get_inv_matrix() * m_inv_matrix;
			return Transform(matrix, inv_matrix);
		}

		__host__ __device__ float3 operator() (const float3& pt, bool point) const
		{
			return m_matrix(pt, point);
		}

		__host__ __device__ Ray operator() (const Ray& ray) const
		{
			float3 transformed_o = (*this)(ray.origin(), true);
			float3 transformed_d = (*this)(ray.direction(), false);
			return Ray(transformed_o, transformed_d);
		}

	private:

		Matrix4x4 m_matrix;
		Matrix4x4 m_inv_matrix;
	};


	//
	// Free helper functions.
	//

	__host__ __device__ Transform scale(float sx, float sy, float sz);

	__host__ __device__ Transform translate(float x, float y, float z);

	__host__ __device__ Transform rotate_x(float theta);

	__host__ __device__ Transform rotate_y(float theta);

	__host__ __device__ Transform rotate_z(float theta);

	__host__ __device__ Transform lookat(const float3& pos, const float3& lookat, const float3& up);

	__host__ __device__ Transform perspective(float fov, float n, float f);

}			// !namespace renderbox2

#undef MATRIX

#endif		// !TRANSFORM_H
