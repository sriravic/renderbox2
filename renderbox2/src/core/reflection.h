
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

#ifndef REFLECTION_H
#define REFLECTION_H

// Application specific headers.
#include <core/buffer.h>
#include <core/params.h>

// Cuda specific headers.
#include <cuda.h>
#include <vector_types.h>
#include <helper_math.h>

// Standard c++ headers.

namespace renderbox2
{

	//
	// This file contains all code for all types of bsdfs supported by the application.
	// All the bsdf evaluation/pdf/sampling routines are monolithic device functions.
	// Any shader evaluation is to use these functions through kernels.
	//


	//
	// Sample structs.
	//

	struct BsdfSample
	{
		float m_dir1, m_dir2;
		float fComponent;
		float pad;
	};

	struct LightSample
	{
		float m_u0, m_u1;
		float fComponent;
		float pad;
	};


	//
	// Shading utility functions.
	// NOTE: All these functions expect the incoming and outgoing directions within the shading frame.
	//

	__device__ __host__ __inline__ float cos_theta(const float3& w)
	{
		return w.z;
	}

	__device__ __host__ __inline__ float abs_cos_theta(const float3& w)
	{
		return fabsf(w.z);
	}

	__device__ __host__ __inline__ float sin_theta2(const float3& w)
	{
		return fmaxf(0.0f, 1.0f - cos_theta(w) * cos_theta(w));
	}

	__device__ __host__ __inline__ float sin_theta(const float3& w)
	{
		return sqrtf(sin_theta2(w));
	}

	__device__ __host__ __inline__ float cos_phi(const float3& w)
	{
		float sintheta = sin_theta(w);
		if (sintheta == 0.0f) return 1.0f;
		else return clamp(w.x / sintheta, -1.0f, 1.0f);
	}

	__device__ __host__ __inline__ float sin_phi(const float3& w)
	{
		float sintheta = sin_theta(w);
		if (sintheta == 0.0f) return 0.0f;
		else return clamp(w.y / sintheta, -1.0f, 1.0f);
	}

	__device__ __host__ __inline__ bool same_hemisphere(const float3& wi, const float3& wo)
	{
		return wi.z * wo.z > 0.0f;
	}

	
	//
	// Utility optics based functions.
	//

	__device__ __host__ __inline__ float3 reflect(const float3& v, const float3& n)
	{
		return 2.0f * dot(n, v) * n - v;
	}

	__device__ __host__ __inline__ float3 refract(const float3& v, const float3& n, const float etai, const float etat)
	{
		bool entering = cos_theta(v) > 0.0f;
		float ei = etai;
		float et = etat;

		if (!entering)
		{
			float temp = ei;
			ei = et;
			et = temp;
		}

		float sini2 = sin_theta2(v);
		float eta = ei / et;
		float sint2 = eta * eta * sini2;

		if (sint2 >= 1.0f) return make_float3(0.0f);

		float cost = sqrtf(fmaxf(0.0f, 1.0f - sint2));
		if (entering) cost = -cost;
		float sintOverSini = eta;
		return make_float3(sintOverSini * -v.x, sintOverSini * -v.y, cost);
	}


	//
	// Fresnel functions.
	//

	__device__ __host__ __inline__ float3 fresnel_dielectric(const float cosi, const float3& etai, const float cost, const float3& etat)
	{
		float3 rparl = ((etat * cosi) - (etai * cost)) / ((etat * cosi) + (etai * cost));
		float3 rperp = ((etai * cosi) - (etat * cost)) / ((etai * cosi) + (etat * cost));
		return (rparl * rparl + rperp * rperp) / 2.0f;
	}

	__device__ __host__ __inline__ float3 fresnel_conductor(const float cosi, const float3& eta, const float3& k)
	{
		float3 temp = (eta * eta + k * k) * cosi * cosi;
		float3 rparl2 = (temp - (2.0f * eta * cosi) + make_float3(1.0f)) /
			(temp + (2.0f * eta * cosi) + make_float3(1.0f));
		
		float3 temp_f = eta * eta + k * k;
		float3 rperp2 = (temp_f - (2.0f * eta * cosi) + make_float3(cosi * cosi)) /
			(temp_f + (2.0f * eta * cosi) + make_float3(cosi * cosi));
		return (rparl2 + rperp2) / 2.0f;
	}


	//
	// Fresnel related structs.
	//

	struct Fresnel
	{
	public:
		__host__ __device__ Fresnel(FresnelType type)
		{
			m_type = type;
		}

		__host__ __device__ float3 evaluate(const float cosi) const
		{
			if (m_type == FresnelType::FRESNEL_CONDUCTOR)
			{
				return fresnel_conductor(cosi, m_eta, m_k);
			}
			else if (m_type == FresnelType::FRESNEL_DIELECTRIC)
			{
				float temp_cosi = clamp(cosi, -1.0f, 1.0f);

				// compute indices of refraction for dielectric
				bool entering = cosi > 0;
				float ei = m_etai, et = m_etat;
				if (!entering)
				{
					float temp = ei;
					ei = et;
					et = temp;
				}

				// compute sint 
				float sint = ei / et * sqrtf((fmaxf(0.0f, 1.0f - cosi * cosi)));
				if (sint >= 1.0f)
					return make_float3(1.0f);		// total internal reflection
				else
				{
					float cost = sqrtf(fmaxf(0.0f, 1.0f - sint * sint));
					return fresnel_dielectric(fabs(cosi), make_float3(ei), cost, make_float3(et));
				}
			}
			else if (m_type == FresnelType::FRESNEL_NOOP)
			{
				return make_float3(1.0f);
			}
			else
			{				
				return make_float3(0.0f);
			}
		}

		FresnelType m_type;
		float		m_etai, m_etat;
		float3		m_eta, m_k;
	};


	//
	// Lambertian Bsdfs.
	//

	__device__ __host__ float3 evaluate_lambertian(const float3& wi_local, const float3& wo_local, const LambertianBsdfParams& params);

	__device__ __host__ float evaluate_lambertian_pdf(const float3& wi_local, const float3& wo_local);

	__device__ __host__ float3 sample_lambertian(const float3& wo_local, const BsdfSample& sample, const LambertianBsdfParams& params, float3& wi_local, float& pdf);


	//
	// Mirror Bsdf.
	//

	__device__ __host__ float3 evaluate_mirror(const float3& wi_local, const float3& wo_local, const MirrorBsdfParams& params);

	__device__ __host__ float evaluate_mirror_pdf(const float3& wi_local, const float3& wo_local);

	__device__ __host__ float3 sample_mirror(const float3& wo_local, const BsdfSample& sample, const MirrorBsdfParams& params, float3& wi_local, float& pdf);


	//
	// SpecularTransmission.
	//

	__device__ __host__ float3 evaluate_glass(const float3& wi_local, const float3& wo_local, const GlassBsdfParams& params);

	__device__ __host__ float evaluate_glass_pdf(const float3& wi_local, const float3& wo_local);

	__device__ __host__ float3 sample_glass(const float3& wo_local, const BsdfSample& sample, const GlassBsdfParams& params, float3& wi_local, float& pdf);


	//
	// OrenNayar Bsdf.
	//

	__device__ __host__ float3 evaluate_orennayar(const float3& wi_local, const float3& wo_local, const OrenNayarBsdfParams& params);

	__device__ __host__ float evaluate_orrennayar_pdf(const float3& wi_local, const float3& wo_local);

	__device__ __host__ float3 sample_orennayar(const float3& wo_local, const BsdfSample& sample, const OrenNayarBsdfParams& params, float3& wi_local, float& pdf);


	//
	// Blinn Microfacet Distribution.
	//

	__device__ __host__ void sample_blinn_microfacet(float3& wi_local, float3& sampled_f, float u1, float u2, float& pdf);

	__device__ __host__ float evaluate_blinn_microfacet_pdf(const float3& wi_local, const float3& wo_local, float exponent);

	__device__ __host__ float blinn_microfacet_D(const float3& wh, float exponent);


	//
	// Power cosine microfacet distribution.
	//

	__device__ __host__ void sample_power_cosine();

	__device__ __host__ float evaluate_power_cosine_pdf();

	__device__ __host__ float evaluate_power_cosine();


	//
	// Anisotropic Microfacet Distribution.
	//

	__device__ __host__ float anisotropic_microfacet_D();
	

	//
	// Torrance Sparrow Microfacet Distribution Model.
	//

	__device__ __host__ float3 evaluate_microfacet(const float3& wi_local, const float3& wo_local, const MicrofacetBsdfParams& params);

	__device__ __host__ float evaluate_microfacet_pdf(const float3& wi_local, const float3& wo_local, float exponent);

	__device__ __host__ float3 sample_microfacet(const float3& wo_local, float3& wi_local, const MicrofacetBsdfParams& params, float u1, float u2, float& pdf);

	__device__ __host__ float microfacet_G(const float3& wi_local, const float3& wo_local, const float3& wh_local);

	//
	// Cook-Torrance Microfacet distribution.
	//

	__inline__ __device__ __host__ float chi_ggx(float v)
	{
		return v > 0.0f ? 1.0f : 0.0f;
	}

	__inline__ __device__ float ggx_distribution(const float3& normal, const float3& half, const float alpha);


	//
	// Fresnel Blend.
	//

}				// !namespace renderbox2

#endif			// !REFLECTION_H
