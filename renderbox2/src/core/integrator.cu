
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
#include <core/integrator.h>
#include <core/lights.h>

// Cuda specific headers.
#include <thrust/random.h>

// Standard c++ headers.

namespace renderbox2
{

	//
	// Direct Lighting computation method.
	//

	__device__ float3 compute_direct_lighting(SceneBuffer scene_buffer, MaterialBuffer material_buffer, BvhStruct bvh, Intersection& isect, uint32_t seed)
	{

		thrust::default_random_engine rng(seed);
		thrust::uniform_real_distribution<float> f01(0.0f, 1.0f);
		thrust::uniform_int_distribution<uint32_t> u01(0, scene_buffer.m_num_lights - 1);
		
		float3 Ld = make_float3(0.0f);

		// First get information about the light to be chosen.
		float light_pdf, bsdf_pdf;
		const uint32_t light_id = u01(rng);
		const uint32_t light_primitive_id = scene_buffer.m_light_ids_to_primitive_ids[light_id];
		float3 light_v0, light_v1, light_v2;

		get_light_vertices(scene_buffer, light_id, light_v0, light_v1, light_v2);

		const uint32_t material_id = scene_buffer.m_tri_material_ids[isect.m_primitive_id];
		const uint32_t light_material_id = scene_buffer.m_tri_material_ids[light_primitive_id];
		const Material& light_material = material_buffer.m_materials[light_material_id];

		const DiffuseEmitterParams& params = material_buffer.m_diffuse_emitter_bsdfs[light_material.layer_bsdf_id[0]];
		DiffuseAreaLight light(light_v0, light_v1, light_v2, make_float3(params.color.x, params.color.y, params.color.z));
		
		// Set the shading frame appropriately.
		Frame shading_frame;
		shading_frame.set_from_z(isect.m_Ns);

		// Sample light with MIS.
		{
			LightSample sample;
			float3 sampled_wi;
			VisibilityTesterElement e;

			sample.m_u0 = f01(rng);
			sample.m_u1 = f01(rng);

			float3 Li = light.sample_L(isect.m_position, isect.m_epsilon, sample, sampled_wi, light_pdf, e);
			if (!is_black(Li) && light_pdf > 0.0f)
			{
				Ray shadow_ray = e.m_visibility_ray;
				float3 f = material_f(material_buffer, material_id, isect.m_wi, shadow_ray.direction(), isect.m_Ns, shading_frame, BxDFType(BxDF_ALL & ~BxDF_SPECULAR));
				bool occluded = trace_ray(shadow_ray, true, nullptr, bvh.m_nodes, bvh.m_tris, bvh.m_tri_idx, scene_buffer);

				if (!is_black(f) && !occluded)
				{
					//bsdf_pdf = 1.0f;		// evaluate bsdf for material.
					bsdf_pdf = material_pdf(material_buffer, material_id, isect.m_wi, sampled_wi, shading_frame, BxDFType(BxDF_ALL & ~BxDF_SPECULAR));
					float weight = balanced_heuristic(1, light_pdf, 1, bsdf_pdf);

					// compute Ld.
					Ld += f * Li * (abs(dot(sampled_wi, isect.m_Ns)) * weight) / light_pdf;
				}
			}
		}

		// Sample bsdf with MIS
		{
			// Sample f
			BsdfSample sample = { f01(rng), f01(rng), f01(rng), 0.0f };
			float3 Li = make_float3(0.0f);

			float3 sampled_wi;
			BxDFType sampled_type;

			float3 f = material_sample_f(material_buffer, material_id, isect.m_wi, shading_frame, isect.m_Ns, sample, sampled_wi, bsdf_pdf, BxDFType(BxDF_ALL & ~BxDF_SPECULAR), sampled_type);

			bool flag = true;
			if (!is_black(f) && bsdf_pdf > 0.0f)
			{
				float weight = 1.0f;
				if (!(sampled_type & BxDF_SPECULAR))
				{
					// Compute light generating the same direction pdf from intersected point.
					light_pdf = light.pdf(isect.m_position, sampled_wi);
					if (light_pdf == 0.0f)
					{
						Ld += make_float3(0.0f);
						flag = false;
					}
					weight = balanced_heuristic(1, bsdf_pdf, 1, light_pdf);
				}

				if (flag)
				{
					// check for full intersection with all scenes.
					Intersection light_isect;
					Ray shadow_ray(isect.m_position, sampled_wi, isect.m_epsilon);

					bool intersected = trace_ray(shadow_ray, false, &light_isect, bvh.m_nodes, bvh.m_tris, bvh.m_tri_idx, scene_buffer);

					if (light_isect.m_intersected == 1 && (light_isect.m_primitive_id == light_primitive_id))
					{
						Li = light.L(-sampled_wi);
					}

					Ld += f * Li * (abs(dot(sampled_wi, isect.m_Ns)) * weight) / bsdf_pdf;
				}
			}
		}
		
		Ld *= scene_buffer.m_num_lights;
		return Ld;
	}


	//
	// Some utility methods that all integrators will use frequently.
	//

	__device__ void get_triangle_vertices(SceneBuffer scene_buffer, GIndexType id, float3& v0, float3& v1, float3& v2)
	{
		const GIndexVec3Type idx = make_uint3(scene_buffer.m_vtx_indices[0][id], scene_buffer.m_vtx_indices[1][id], scene_buffer.m_vtx_indices[2][id]);
		v0 = make_float3(scene_buffer.m_vertices[0][idx.x], scene_buffer.m_vertices[1][idx.x], scene_buffer.m_vertices[2][idx.x]);
		v1 = make_float3(scene_buffer.m_vertices[0][idx.y], scene_buffer.m_vertices[1][idx.y], scene_buffer.m_vertices[2][idx.y]);
		v2 = make_float3(scene_buffer.m_vertices[0][idx.z], scene_buffer.m_vertices[1][idx.z], scene_buffer.m_vertices[2][idx.z]);
	}

	__device__ void get_light_vertices(SceneBuffer scene_buffer, uint32_t light_id, float3& v0, float3& v1, float3& v2)
	{
		const GIndexVec3Type light_idx = make_uint3(scene_buffer.m_light_indices[0][light_id], scene_buffer.m_light_indices[1][light_id], scene_buffer.m_light_indices[2][light_id]);
		v0 = make_float3(scene_buffer.m_vertices[0][light_idx.x], scene_buffer.m_vertices[1][light_idx.x], scene_buffer.m_vertices[2][light_idx.x]);
		v1 = make_float3(scene_buffer.m_vertices[0][light_idx.y], scene_buffer.m_vertices[1][light_idx.y], scene_buffer.m_vertices[2][light_idx.y]);
		v2 = make_float3(scene_buffer.m_vertices[0][light_idx.z], scene_buffer.m_vertices[1][light_idx.z], scene_buffer.m_vertices[2][light_idx.z]);
	}
}
