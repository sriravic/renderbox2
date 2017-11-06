
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
#include <core/material.h>
#include <core/params.h>

// Cuda specific headers.

// Standard c++ headers.

namespace renderbox2
{

	//
	// Util functions
	//

	__inline__ __device__ __host__ bool matches_flags(BxDFType source_flag, BxDFType compare_flag)
	{
		return (source_flag & compare_flag) == source_flag;
	}

	__inline__ __device__ __host__ uint32_t num_components(const Material& material, BxDFType flags)
	{
		uint32_t num = 0;
		for (uint32_t i = 0; i < material.num_valid_layers; i++)
		{
			if (matches_flags(material.layer_bxdf_type[i], flags))
				num++;
		}
		return num;
	}


	//
	// Material sampling, pdf and brdf calculation routines.
	//

	__host__ __device__ float3 material_sample_f(
		MaterialBuffer material_buffer,
		uint32_t material_id,
		const float3& wo_world,
		const Frame& shading_frame,
		const float3& geometric_normal,
		const BsdfSample& sample,
		float3& wi_world,
		float& pdf,
		BxDFType flags,
		BxDFType& sampled_type)
	{
		const Material material = material_buffer.m_materials[material_id];
		if (material.num_valid_layers == 0) return make_float3(0.0f);

		// Choose Bsdf to sample
		uint32_t num_matching_components = num_components(material, flags);

		if (num_matching_components == 0)
		{
			sampled_type = BxDFType(0);
			pdf = 0.0f;
			return make_float3(0.0f);
		}

		uint32_t which = min(static_cast<uint32_t>(floorf(sample.fComponent * num_matching_components)), num_matching_components - 1);
		
		// Sample the chosen bsdf
		BsdfType sampled_bsdf_type = material.layer_bsdf_type[which];
		sampled_type = material.layer_bxdf_type[which];
		float3 wo_local = shading_frame.to_local(wo_world);
		float3 wi_local;

		float3 f = make_float3(0.0f);
		
		switch (sampled_bsdf_type)
		{
		case BSDF_LAMBERTIAN:
		{
			LambertianBsdfParams params = material_buffer.m_lambertian_bsdfs[material.layer_bsdf_id[which]];
			f = sample_lambertian(wo_local, sample, params, wi_local, pdf);
			sampled_type = BxDFType(BxDF_REFLECTION | BxDF_DIFFUSE);
		}
			break;
		case BSDF_GLASS:
		{
			GlassBsdfParams params = material_buffer.m_glass_bsdfs[material.layer_bsdf_id[which]];
			f = sample_glass(wo_local, sample, params, wi_local, pdf);
			sampled_type = BxDFType(BxDF_SPECULAR | BxDF_TRANSMISSION);
		}
			break;
		case BSDF_MIRROR:
		{
			MirrorBsdfParams params = material_buffer.m_mirror_bsdfs[material.layer_bsdf_id[which]];
			f = sample_mirror(wo_local, sample, params, wi_local, pdf);
			sampled_type = BxDFType(BxDF_SPECULAR | BxDF_REFLECTION);
		}
			break;
		case BSDF_BLINN_MICROFACET:
		{
			MicrofacetBsdfParams params = material_buffer.m_microfacet_bsdfs[material.layer_bsdf_id[which]];
			f = sample_microfacet(wo_local, wi_local, params, sample.m_dir1, sample.m_dir2, pdf);
			sampled_type = BxDFType(BxDF_REFLECTION | BxDF_GLOSSY);
		}
			break;
		case BSDF_OREN_NAYAR:
		{
			OrenNayarBsdfParams params = material_buffer.m_orennayar_bsdfs[material.layer_bsdf_id[which]];
			f = sample_orennayar(wo_local, sample, params, wi_local, pdf);
		}
			break;
		}

		if (pdf == 0.0f)
		{
			sampled_type = BxDFType(0);
			return make_float3(0.0f);
		}

		wi_world = shading_frame.to_world(wi_local);

		// Compute overall pdf.
		if (!(sampled_type & BxDF_SPECULAR) && num_matching_components > 1)
		{
			for (uint32_t i = 0; i < material.num_valid_layers; i++)
			{
				if (i != which && matches_flags(material.layer_bxdf_type[i], sampled_type))
				{
					// add the pdf
					BsdfType type = material.layer_bsdf_type[i];
					switch (type)
					{
					case BSDF_LAMBERTIAN:
						pdf += evaluate_lambertian_pdf(wi_local, wo_local);
						break;
					case BSDF_OREN_NAYAR:
						pdf += evaluate_orrennayar_pdf(wi_local, wo_local);
						break;
					case BSDF_MIRROR:
						pdf += evaluate_mirror_pdf(wi_local, wo_local);
						break;
					case BSDF_GLASS:
						pdf += evaluate_glass_pdf(wi_local, wo_local);
						break;
					case BSDF_BLINN_MICROFACET:
					{
						MicrofacetBsdfParams params = material_buffer.m_microfacet_bsdfs[material.layer_bsdf_id[i]];
						pdf += evaluate_microfacet_pdf(wi_local, wo_local, params.exponent);
					}
						break;
					}
				}
			}
		}

		// compute value of bsdf for chosen direction.
		if (!(sampled_type & BxDF_SPECULAR))
		{
			f = make_float3(0.0f);
			if (dot(wi_world, geometric_normal) * dot(wo_world, geometric_normal) > 0)
				sampled_type = BxDFType(sampled_type & ~BxDF_TRANSMISSION);
			else
				sampled_type = BxDFType(sampled_type & ~BxDF_REFLECTION);
			for (uint32_t i = 0; i < material.num_valid_layers; i++)
			{
				if (matches_flags(material.layer_bxdf_type[i], flags))
				{
					BsdfType type = material.layer_bsdf_type[i];
					switch (type)
					{
					case BSDF_LAMBERTIAN:
					{
						LambertianBsdfParams params = material_buffer.m_lambertian_bsdfs[material.layer_bsdf_id[i]];
						f += evaluate_lambertian(wi_local, wo_local, params);
					}
						break;
					case BSDF_OREN_NAYAR:
					{
						OrenNayarBsdfParams params = material_buffer.m_orennayar_bsdfs[material.layer_bsdf_id[i]];
						f += evaluate_orennayar(wi_local, wo_local, params);
					}
						break;
					case BSDF_MIRROR:
					{
						MirrorBsdfParams params = material_buffer.m_mirror_bsdfs[material.layer_bsdf_id[i]];
						f += evaluate_mirror(wi_local, wo_local, params);
					}
						break;
					case BSDF_GLASS:
					{
						GlassBsdfParams params = material_buffer.m_glass_bsdfs[material.layer_bsdf_id[i]];
						f += evaluate_glass(wi_local, wo_local, params);
					}
					case BSDF_BLINN_MICROFACET:
					{
						MicrofacetBsdfParams params = material_buffer.m_microfacet_bsdfs[material.layer_bsdf_id[i]];
						f += evaluate_microfacet(wi_local, wo_local, params);
					}
						break;
					}
				}
			}
		}

		return f;
	}

	__host__ __device__ float material_pdf(MaterialBuffer material_buffer, uint32_t material_id, const float3& wo_world, const float3& wi_world, const Frame& shading_frame, BxDFType flags)
	{
		const Material material = material_buffer.m_materials[material_id];
		if (material.num_valid_layers == 0) return 0.0f;
		float3 wi_local = shading_frame.to_local(wi_world);
		float3 wo_local = shading_frame.to_local(wo_world);

		float pdf = 0.0f;
		uint32_t n_matching_comps = 0;
		
		for (uint32_t i = 0; i < material.num_valid_layers; i++)
		{
			if (matches_flags(material.layer_bxdf_type[i], flags))
			{
				n_matching_comps++;
				BsdfType type = material.layer_bsdf_type[i];
				switch (type)
				{
				case BSDF_LAMBERTIAN:
					pdf += evaluate_lambertian_pdf(wi_local, wo_local);
					break;
				case BSDF_OREN_NAYAR:
					pdf += evaluate_orrennayar_pdf(wi_local, wo_local);
					break;
				case BSDF_MIRROR:
					pdf += evaluate_mirror_pdf(wi_local, wo_local);
					break;
				case BSDF_GLASS:
					pdf += evaluate_glass_pdf(wi_local, wo_local);
					break;
				case BSDF_BLINN_MICROFACET:
					MicrofacetBsdfParams params = material_buffer.m_microfacet_bsdfs[material.layer_bsdf_id[i]];
					pdf += evaluate_microfacet_pdf(wi_local, wo_local, params.exponent);
					break;
				}
			}
		}

		float v = n_matching_comps > 0 ? pdf / n_matching_comps : 0.0f;
		return v;
	}

	__host__ __device__ float3 material_f(
		const MaterialBuffer material_buffer,
		uint32_t material_id,
		const float3& wo_world,
		const float3& wi_world,
		const float3& geometric_normal,
		const Frame& shading_frame,
		BxDFType flags)
	{
		const Material& material = material_buffer.m_materials[material_id];
		if (material.num_valid_layers == 0) return make_float3(0.0f);
		float3 wi_local = shading_frame.to_local(wi_world);
		float3 wo_local = shading_frame.to_local(wo_world);

		if (dot(wi_world, geometric_normal) * dot(wo_world, geometric_normal) > 0)
			flags = BxDFType(flags & ~BxDF_TRANSMISSION);
		else
			flags = BxDFType(flags & ~BxDF_REFLECTION);

		float3 f = make_float3(0.0f);
		for (uint32_t i = 0; i < material.num_valid_layers; i++)
		{
			if (matches_flags(material.layer_bxdf_type[i], flags))
			{
				BsdfType type = material.layer_bsdf_type[i];
				switch (type)
				{
				case BSDF_LAMBERTIAN:
				{
					LambertianBsdfParams params = material_buffer.m_lambertian_bsdfs[material.layer_bsdf_id[i]];
					f += evaluate_lambertian(wi_local, wo_local, params);
				}
					break;
				case BSDF_OREN_NAYAR:
				{
					OrenNayarBsdfParams params = material_buffer.m_orennayar_bsdfs[material.layer_bsdf_id[i]];
					f += evaluate_orennayar(wi_local, wo_local, params);
				}
					break;
				case BSDF_MIRROR:
				{
					MirrorBsdfParams params = material_buffer.m_mirror_bsdfs[material.layer_bsdf_id[i]];
					f += evaluate_mirror(wi_local, wo_local, params);
				}
					break;
				case BSDF_GLASS:
				{
					GlassBsdfParams params = material_buffer.m_glass_bsdfs[material.layer_bsdf_id[i]];
					f += evaluate_glass(wi_local, wo_local, params);
				}
					break;
				case BSDF_BLINN_MICROFACET:
				{
					MicrofacetBsdfParams params = material_buffer.m_microfacet_bsdfs[material.layer_bsdf_id[i]];
					f += evaluate_microfacet(wi_local, wo_local, params);
				}
					break;
				}
			}	
		}
		return f;
	}
}
