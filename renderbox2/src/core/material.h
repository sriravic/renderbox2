
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

#ifndef MATERIAL_H
#define MATERIAL_H

// Application specific headers.
#include <core/bsdf.h>
#include <core/params.h>
#include <core/reflection.h>

// Cuda specific headers.

// Standard c++ headers.
#include <cstdint>

namespace renderbox2
{

#define MAX_LAYERS 4

	//
	// Material system implementation.
	// Each material can have at the max 8 layers.
	// Each layer has a type + id of a particular bsdf stored in memory.
	// They are evaluated by fetching bsdf parameters appropriately.
	//

	struct Material
	{
		BxDFType layer_bxdf_type[MAX_LAYERS];
		BsdfType layer_bsdf_type[MAX_LAYERS];
		uint32_t layer_bsdf_id[MAX_LAYERS];
		uint32_t num_valid_layers;
		uint32_t m_emitter;
	};

	
	//
	// Sampling the material bsdfs.
	//

	__host__ __device__ float3 material_sample_f(MaterialBuffer material_buffer, uint32_t material_id, const float3& wo_world, const Frame& shading_frame, const float3& geometric_normal,
		const BsdfSample& sample, float3& wi_world, float& pdf, BxDFType flags, BxDFType& sampled_type);

	__host__ __device__ float material_pdf(MaterialBuffer material_buffer, uint32_t material_id, const float3& wo_world, const float3& wi_world, const Frame& shading_frame, BxDFType flags);

	__host__ __device__ float3 material_f(const MaterialBuffer material_buffer, uint32_t material_id, const float3& wo_world, const float3& wi_world, const float3& geometric_normal, const Frame& shading_frame,
		BxDFType flags);

};			// !namespace renderbox2

#endif		// !MATERIAL_H
