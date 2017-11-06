
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
#include <core/montecarlo.h>
#include <core/reflection.h>

// Cuda specific headers.

// Standard c++ headers.

namespace renderbox2
{

	//
	// Individual Brdfs supported in renderbox2.
	// NOTE: All functions expect directions in the local coordinate frame.
	//


	//
	// Lambertian Bsdf.
	//

	__device__ __host__ float3 evaluate_lambertian(const float3& wi_local, const float3& wo_local, const LambertianBsdfParams& params)
	{
		if (!same_hemisphere(wi_local, wo_local)) return make_float3(0.0f);
		float3 reflectance = make_float3(params.color.x, params.color.y, params.color.z);
		return reflectance * INV_PI;

	}

	__device__ __host__ float evaluate_lambertian_pdf(const float3& wi_local, const float3& wo_local)
	{
		if (!same_hemisphere(wi_local, wo_local)) return 0.0f;
		return abs_cos_theta(wi_local) * INV_PI;
	}

	__device__ __host__ float3 sample_lambertian(const float3& wo_local, const BsdfSample& sample, const LambertianBsdfParams& params, float3& wi_local, float& pdf)
	{
		wi_local = cosine_sample_hemisphere(sample.m_dir1, sample.m_dir2);

		// Flip directions if necessary.
		if (wo_local.z < 0.0f)
			wi_local.z *= -1.0f;

		wi_local = normalize(wi_local);
		
		pdf = evaluate_lambertian_pdf(wi_local, wo_local);

		return evaluate_lambertian(wi_local, wo_local, params);
	}


	//
	// Mirror Bsdf.
	//

	__device__ __host__ float3 evaluate_mirror(const float3& wi_local, const float3& wo_local, const MirrorBsdfParams& params)
	{
		return make_float3(0.0f);
	}

	__device__ __host__ float evaluate_mirror_pdf(const float3& wi_local, const float3& wo_local)
	{
		return 0.0f;
	}

	__device__ __host__ float3 sample_mirror(const float3& wo_local, const BsdfSample& sample, const MirrorBsdfParams& params, float3& wi_local, float& pdf)
	{
		wi_local = make_float3(-wo_local.x, -wo_local.y, wo_local.z);
		pdf = 1.0f;
		Fresnel fresnel(FresnelType::FRESNEL_NOOP);
		return fresnel.evaluate(cos_theta(wo_local)) * make_float3(params.color.x, params.color.y, params.color.z) / abs_cos_theta(wi_local);
	}

	
	//
	// Glass Bsdf.
	//

	__device__ __host__ float3 evaluate_glass(const float3& wi_local, const float3& wo_local, const GlassBsdfParams& params)
	{
		return make_float3(0.0f);
	}

	__device__ __host__ float evaluate_glass_pdf(const float3& wi_local, const float3& wo_local)
	{
		return 0.0f;
	}

	__device__ __host__ float3 sample_glass(const float3& wo_local, const BsdfSample& sample, const GlassBsdfParams& params, float3& wi_local, float& pdf)
	{
		
		Fresnel fresnel(FresnelType::FRESNEL_DIELECTRIC);

		bool entering = cos_theta(wo_local) > 0.0f;
		float ei = fresnel.m_etai = params.etai;
		float et = fresnel.m_etat = params.etat;

		if (!entering)
		{
			float temp = ei;
			ei = et;
			et = temp;
		}

		float sini2 = sin_theta2(wo_local);
		float eta = ei / et;
		float sint2 = eta * eta * sini2;

		if (sint2 >= 1.0f) return make_float3(0.0f);

		float cost = sqrtf(fmaxf(0.0f, 1.0f - sint2));
		if (entering) cost = -cost;
		float sintOverSini = eta;
		wi_local = make_float3(sintOverSini * -wo_local.x, sintOverSini * -wo_local.y, cost);
		pdf = 1.0f;
		
		float3 f = fresnel.evaluate(cos_theta(wo_local));
		return (et * et) / (ei * ei) * ((make_float3(1.0f) - f) * make_float3(params.color.x, params.color.y, params.color.z)) / abs_cos_theta(wi_local);
	}


	//
	// Oren Nayar Bsdf.
	//

	__device__ __host__ float3 evaluate_orennayar(const float3& wi_local, const float3& wo_local, const OrenNayarBsdfParams& params)
	{
		float sinthetai = sin_theta(wi_local);
		float sinthetao = sin_theta(wo_local);
		float maxcos = 0.0f;
		if(sinthetai > 1e-4f && sinthetao > 1e-4f)
		{
			float sinphii = sin_phi(wi_local), cosphii = cos_phi(wi_local);
			float sinphio = sin_phi(wo_local), cosphio = cos_phi(wo_local);
			float dcos = cosphii * cosphio + sinphii * sinphio;
			maxcos = fmaxf(0.0f, dcos);
		}

		float sinalpha, tanbeta;
		if (abs_cos_theta(wi_local) > abs_cos_theta(wo_local))
		{
			sinalpha = sinthetao;
			tanbeta = sinthetai / abs_cos_theta(wi_local);
		}
		else
		{
			sinalpha = sinthetai;
			tanbeta = sinthetao / abs_cos_theta(wo_local);
		}

		return make_float3(params.color.x, params.color.y, params.color.z) * INV_PI *(params.AB.x + params.AB.y * maxcos * sinalpha * tanbeta);
	}

	__device__ __host__ float evaluate_orrennayar_pdf(const float3& wi_local, const float3& wo_local)
	{
		if (!same_hemisphere(wi_local, wo_local)) return 0.0f;
		return abs_cos_theta(wi_local) * INV_PI;
	}

	__device__ __host__ float3 sample_orennayar(const float3& wo_local, const BsdfSample& sample, const OrenNayarBsdfParams& params, float3& wi_local, float& pdf)
	{
		wi_local = cosine_sample_hemisphere(sample.m_dir1, sample.m_dir2);

		// Flip directions if necessary.
		if (wo_local.z < 0.0f)
			wi_local.z *= -1.0f;

		wi_local = normalize(wi_local);

		pdf = evaluate_orrennayar_pdf(wi_local, wo_local);

		return evaluate_orennayar(wi_local, wo_local, params);
	}


	//
	// Blinn Microfacet distribution function.
	//

	__device__ __host__ __inline__ float3 spherical_direction(float sintheta, float costheta, float phi)
	{
		return make_float3(sintheta * cosf(phi), sintheta * sinf(phi), costheta);
	}

	__device__ __host__ void sample_blinn_microfacet(const float3& wo_local, float3& wi_local, float exponent, float u1, float u2, float& pdf)
	{
		float costheta = pow(u1, 1.0f / (exponent + 1));
		float sintheta = sqrtf(fmaxf(0.0f, 1.0f - costheta * costheta));
		float phi = u2 * 2.0f * M_PI;

		float3 wh_local = spherical_direction(sintheta, costheta, phi);
		if (!same_hemisphere(wo_local, wh_local)) wh_local = -wh_local;
		wi_local = make_float3(-wo_local.x, -wo_local.y, -wo_local.z) + 2.0f * dot(wo_local, wh_local) * wh_local;
		
		float blinn_pdf = ((exponent + 1.0f) * pow(costheta, exponent)) / (2.0f * M_PI * 4.0f * dot(wo_local, wh_local));
		
		if (dot(wo_local, wh_local) <= 0.0f) blinn_pdf = 0.0f;
		pdf = blinn_pdf;
	}

	__device__ __host__ float evaluate_blinn_microfacet_pdf(const float3& wi_local, const float3& wo_local, float exponent)
	{
		float3 wh_local = normalize(wo_local + wi_local);
		float costheta = abs_cos_theta(wh_local);
		float blinn_pdf = ((exponent + 1.0f) * pow(costheta, exponent)) / (2.0f * M_PI * 4.0f * dot(wo_local, wh_local));
		if (dot(wo_local, wh_local) <= 0.0f) blinn_pdf = 0.0f;
		return blinn_pdf;
	}

	__device__ __host__ float blinn_microfacet_D(const float3& wh, float exponent)
	{
		float costhetah = abs_cos_theta(wh);
		return (exponent + 2) * INV_TWOPI * pow(costhetah, exponent);
	}


	//
	// Torrance Sparrow Microfacet model.
	//

	__device__ __host__ float microfacet_G(const float3& wi_local, const float3& wo_local, const float3& wh_local)
	{
		float NdotWh = abs_cos_theta(wh_local);
		float NdotWo = abs_cos_theta(wo_local);
		float NdotWi = abs_cos_theta(wi_local);
		float WOdotWh = fabsf(dot(wo_local, wh_local));
		return fminf(1.0f, fminf((2.0f * NdotWh / WOdotWh), (2.0f * NdotWh * NdotWi / WOdotWh)));
	}

	__device__ __host__ float3 evaluate_microfacet(const float3& wi_local, const float3& wo_local, const MicrofacetBsdfParams& params)
	{
		float costhetaO = abs_cos_theta(wo_local);
		float costhetaI = abs_cos_theta(wi_local);

		if (costhetaI == 0.0f || costhetaO == 0.0f) return make_float3(0.0f);

		float3 wh_local = wi_local + wo_local;

		if (wh_local.x == 0.0f && wh_local.y == 0.0f && wh_local.z == 0.0f) return make_float3(0.0f);

		wh_local = normalize(wh_local);

		float costhetaH = dot(wi_local, wh_local);

		Fresnel fresnel(params.type);

		float3 F;

		switch (params.type)
		{
		case FresnelType::FRESNEL_CONDUCTOR:
			fresnel.m_k = make_float3(params.k.x, params.k.y, params.k.z);
			fresnel.m_eta = make_float3(params.eta.x, params.eta.y, params.eta.z);
			break;
		case FresnelType::FRESNEL_DIELECTRIC:
			fresnel.m_etai = params.etai;
			fresnel.m_etat = params.etat;
			break;
		case FresnelType::FRESNEL_NOOP:
			break;
		}

		F = fresnel.evaluate(costhetaH);
		return make_float3(params.R.x, params.R.y, params.R.z) * blinn_microfacet_D(wh_local, params.exponent) * microfacet_G(wi_local, wo_local, wh_local) * F / (4.0f * costhetaI * costhetaO);
	}

	__device__ __host__ float evaluate_microfacet_pdf(const float3& wi_local, const float3& wo_local, float exponent)
	{
		if (!same_hemisphere(wi_local, wo_local)) return 0.0f;
		float pdf = evaluate_blinn_microfacet_pdf(wi_local, wo_local, exponent);
		return pdf;
	}

	__device__ __host__ float3 sample_microfacet(const float3& wo_local, float3& wi_local, const MicrofacetBsdfParams& params, float u1, float u2, float& pdf)
	{
		sample_blinn_microfacet(wo_local, wi_local, params.exponent, u1, u2, pdf);
		if (!same_hemisphere(wi_local, wo_local)) return make_float3(0.0f);
		return evaluate_microfacet(wi_local, wo_local, params);
	}
}
