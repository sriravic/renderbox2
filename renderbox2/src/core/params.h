
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

#ifndef PARAMS_H
#define PARAMS_H

// Application specific headers.
#include <core/bsdf.h>
#include <core/filter.h>

// Cuda specific headers.

// Standard c++ headers.
#include <cstdint>

namespace renderbox2
{
	//
	// This file contains all the parameter structures required to control application settings.
	//


	//
	// Application params.
	// Holds all data required to control application flow.
	//

	struct ApplicationParams
	{

	};

	
	//
	// Renderer Params.
	//

	
	//
	// Sampler Renderer Params.
	//

	enum class SamplerRendererMode
	{
		MODE_FINAL,
		MODE_PROGRESSIVE
	};

	struct SamplerRendererParams
	{
		uint32_t			m_spp;
		uint32_t			m_iterations;
		SamplerRendererMode m_mode;
	};

	
	//
	// Integrator Params.
	//


	//
	// Path Integrator Params.
	//

	enum class PathIntegratorMode
	{
		MODE_SINGLE_MEGA_KERNEL,							// A large single monolithic kernel with no ray regeneration and proceeds to completion.
		MODE_MULTI_KERNEL3									// Trace/Shade/Logic variant of path tracing.
	};

	struct PathIntegratorParams
	{
		uint32_t	m_max_depth;
		uint32_t	m_rrstart;
		float		m_rrprob;
		bool		m_use_max_depth;		
	};


	//
	// Bidirectional Path Integrator Params.
	//

	enum class BidirectionalPathIntegratorMode
	{
		MODE_BDPT_LVC_SK,
		MODE_BDPT_LVC_MK,
		MODE_BDPT_SORTED_LVC
	};

	struct BidirectionalPathIntegratorLvcParams
	{
		uint32_t m_num_prep_paths;
		uint32_t m_num_light_paths;
		uint32_t m_num_light_path_max_depth;
		uint32_t m_light_path_rrstart;
		uint32_t m_num_cam_path_max_depth;
		uint32_t m_cam_path_rrstart;
		float    m_cam_path_rrprob;
		uint32_t m_num_cam_path_sample_connections;
		uint32_t m_num_cam_path_connections;
		
		BidirectionalPathIntegratorMode m_mode;
	};


	//
	// Ambient Occlusion Integrator Params.
	//

	struct AmbientOcclusionIntegratorParams
	{
		uint32_t	m_samples;
		float		m_radius;
	};


	//
	// Ray Cast Integrator Params.
	//

	enum RayCastShade
	{
		SHADE_PRIMITIVE_ID,
		SHADE_MATERIAL_ID,
		SHADE_NORMALS,
		SHADE_UVS,
		SHADE_UNIFORM
	};

	struct RayCastIntegratorParams
	{
		RayCastShade m_shade;
	};

	
	//
	// Filter types supported.
	//

	enum class FilterType
	{
		FILTER_BOX,
		FILTER_TRIANGLE,
		FILTER_GAUSSIAN,
		FILTER_SINC,
		FILTER_MITCHELL,
		FILTER_UNKNOWN
	};


	//
	// Filter Params.
	//

	struct FilterParams
	{
		float		xwidth;
		float		ywidth;
		float		alpha;				// parameter used for gaussian filter.
		FilterType	m_type;
	};


	//
	// Microfacet Distributions.
	//

	enum class MicrofacetDistributions { BLINN, ANISOTROPIC };

	enum class FresnelType { FRESNEL_DIELECTRIC, FRESNEL_CONDUCTOR, FRESNEL_NOOP, FRESNEL_NONE };
	

	//
	// Various BSDF params.
	//

	struct LambertianBsdfParams
	{
		float4 color;
	};

	struct GlassBsdfParams
	{
		float etai;
		float etat;
		float4 color;
	};

	struct MirrorBsdfParams
	{
		float4 color;
	};

	struct DiffuseEmitterParams
	{
		float4 color;
	};

	struct OrenNayarBsdfParams
	{
		float4 color;
		float2 AB;
		float  sigma;
		float  pad;
	};

	struct FresnelBlendBsdfParams
	{

	};

	struct BlinnMicrofacetParams
	{
		float exponent;
	};

	struct MicrofacetBsdfParams
	{
		float4 R;
		float4 eta, k;
		float etai, etat;
		float exponent;						// BlinnMicrofacet bsdf exponent.
		FresnelType type;
	};	
};			// !namespace renderbox2

#endif		// !PARAMS_H
