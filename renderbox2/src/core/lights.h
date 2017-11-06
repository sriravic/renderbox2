
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

#ifndef LIGHTS_H
#define LIGHTS_H

// Application specifc headers.
#include <core/montecarlo.h>
#include <core/primitives.h>
#include <core/reflection.h>
#include <core/util.h>

// Cuda specific headers.

// Standard c++ headers.

namespace renderbox2
{

	//
	// As of now, we support only diffuse area lights so that it better fits the bidirectional path integration framework.
	//

	class DiffuseAreaLight
	{
	public:

		__host__ __device__ DiffuseAreaLight();

		__host__ __device__ DiffuseAreaLight(const float3& v0, const float3& v1, const float3& v2, const float3& emit);

		__host__ __device__ float3 power() const;

		// Given random numbers, this method generates a starting point on the light and a direction from which light is shot into the scene.
		// It also provides the probability with respect to p(a) * projected_solid angle of generating this light ray sample.
		__host__ __device__ float3 Le(const LightSample& sample, const float2& dir_sample, float3& pt_on_light, float3& dir_from_light, float& pdf);


		// This function determines the radiance received from the light in a particular direction.
		__host__ __device__ float3 L(const float3& light_direction);


		// Given a point at which direct lighting has to be evaluated, this function samples a point on the light from which radiance can be received.
		// Returned probability is the probability of choosing that point on the light with respect to area.
		__host__ __device__ float3 sample_L(const float3& evaluation_pt, const float epsilon, const LightSample& sample, float3& wi, float& pdf, VisibilityTesterElement& visibility_element);

		// Compute the pdf of sampling a direction from an evaluation point towards the light.
		__host__ __device__ float pdf(const float3& evaluation_pt, const float3& dir_towards_light);

	private:
		float3 m_position[3];
		float3 m_face_normal;
		float3 m_lemit;
		Frame  m_frame;
		float  m_area;
		float  m_inv_area;
	};

};			// !namesapce renderbox2

#endif		// !LIGHTS_H