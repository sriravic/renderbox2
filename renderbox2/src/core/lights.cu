
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
#include <core/lights.h>

// Cuda specific headers.

// Standard c++ headers.

namespace renderbox2
{

	//
	// DiffuseAreaLight class implementation.
	//

	__device__ __host__ DiffuseAreaLight::DiffuseAreaLight()
	{
		m_position[0] = m_position[1] = m_position[2] = make_float3(0.0f);
	}

	__device__ __host__ DiffuseAreaLight::DiffuseAreaLight(const float3& v0, const float3& v1, const float3& v2, const float3& emit)
	{
		m_position[0] = v0;
		m_position[1] = v1;
		m_position[2] = v2;
		m_face_normal = compute_normal(v0, v1, v2);
		m_lemit = emit;

		m_frame.set_from_z(m_face_normal);

		m_area = compute_area(v0, v1, v2);
		m_inv_area = 1.0f / m_area;
	}

	__device__ __host__ float3 DiffuseAreaLight::L(const float3& light_direction)
	{
		if (dot(m_face_normal, light_direction) > 0.0f) return m_lemit;
		else return make_float3(0.0f);
	}

	__device__ __host__ float3 DiffuseAreaLight::sample_L(const float3& evaluation_pt, const float epsilon, const LightSample& sample, float3& wi, float& pdf, VisibilityTesterElement& visibility_element)
	{
		float3 light_pt = uniform_sample_triangle(m_position[0], m_position[1], m_position[2], sample.m_u0, sample.m_u1);
		wi = normalize(light_pt - evaluation_pt);
		pdf = sample_triangle_pdf(evaluation_pt, wi, m_position[0], m_position[1], m_position[2]);
		visibility_element.set_segment(evaluation_pt, epsilon, light_pt, 1.0e-3f);
		const float3 wi_dash = make_float3(-wi.x, -wi.y, -wi.z);
		return L(wi_dash);
	}

	// The probability returned is with respect to p(a) * projected_solid_angle probability.
	__host__ __device__ float3 DiffuseAreaLight::Le(const LightSample& sample, const float2& dir_sample, float3& pt_on_light, float3& dir_from_light, float& pdf)
	{
		pt_on_light = uniform_sample_triangle(m_position[0], m_position[1], m_position[2], sample.m_u0, sample.m_u1);
		dir_from_light = uniform_sample_hemisphere(dir_sample.x, dir_sample.y);
		if (dot(dir_from_light, m_face_normal)) dir_from_light *= -1.0f;

		pdf = m_inv_area * uniform_hemisphere_pdf() / abs(dot(m_face_normal, dir_from_light));
		return L(dir_from_light);
	}

	__host__ __device__ float DiffuseAreaLight::pdf(const float3& evaluation_pt, const float3& direction_to_light)
	{
		return sample_triangle_pdf(evaluation_pt, direction_to_light, m_position[0], m_position[1], m_position[2]);
	}
}
