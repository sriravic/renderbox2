
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
#include <accelerators/sbvh/cudatracerkernels.h>
#include <accelerators/sbvh/cudatracer.h>
#include <core/buffer.h>
#include <core/intersection.h>
#include <core/lights.h>
#include <core/montecarlo.h>
#include <core/util.h>
#include <integrators/integrator_path.h>
#include <util/cudatimer.h>

// Cuda specific headers.
#include <thrust/random.h>

// Standard c++ headers.

namespace renderbox2
{

	//
	// Path Tracing Single Monolith Kernel.
	//

	__global__ void kernel_pathtracer_single_monolithic(
		IntersectionBuffer	ib,
		CameraSampleBuffer	csb,
		RayBuffer			rb,
		BvhStruct			bvh,
		SceneBuffer			scene_buffer,
		MaterialBuffer		material_buffer,
		uint32_t			rrstart,
		uint32_t			maxbounce,
		float				rrprob,
		uint32_t			iteration
		)
	{
		for (GIndexType tidx = threadIdx.x + blockIdx.x * blockDim.x; tidx < csb.m_size; tidx += gridDim.x * blockDim.x)
		{
			float4 color = make_float4(0.0f);
			if (ib.m_intersected[tidx] == 1)
			{
				bool alive = true;
				uint32_t seed = simplehash(tidx + iteration);
				thrust::default_random_engine rng(seed);
				thrust::uniform_int_distribution<uint32_t> u01(0, scene_buffer.m_num_lights - 1);
				thrust::uniform_real_distribution<float> f01(0.0f, 1.0f);

				float4& path_throughput = csb.m_throughput[tidx];
				float4 L = make_float4(0.0f);
				bool specular_bounce = false;

				Ray current_ray = rb.m_data[tidx];
				
				Intersection isect;
				isect.m_primitive_id = ib.m_primitive_id[tidx];
				isect.m_position = make_float3(ib.m_position[0][tidx], ib.m_position[1][tidx], ib.m_position[2][tidx]);
				isect.m_Ng = make_float3(ib.m_geomteric_normal[0][tidx], ib.m_geomteric_normal[1][tidx], ib.m_geomteric_normal[2][tidx]);
				isect.m_Ns = make_float3(ib.m_shading_normal[0][tidx], ib.m_shading_normal[1][tidx], ib.m_shading_normal[2][tidx]);
				isect.m_wi = -current_ray.direction();
				isect.m_epsilon = ib.m_epsilon[tidx];

				Frame shading_frame;
				uint32_t bounce = 0;
				
				do
				{
					assert(isect.m_primitive_id < scene_buffer.m_size);

					// First get the material of interseced element
					const uint32_t material_id = scene_buffer.m_tri_material_ids[isect.m_primitive_id];
					const Material& material = material_buffer.m_materials[material_id];
					float3 Ld = make_float3(0.0f);
					
					// Possibly add illumination at the object itself.
					if ((material.m_emitter == 1 && bounce == 0) || (material.m_emitter == 1 && specular_bounce))
					{
						// Add contribution and exit.
						const uint32_t diffuse_emitter_id = material.layer_bsdf_id[0];
						const DiffuseEmitterParams& params = material_buffer.m_diffuse_emitter_bsdfs[diffuse_emitter_id];
						
						float3 light_v0, light_v1, light_v2;
						get_triangle_vertices(scene_buffer, isect.m_primitive_id, light_v0, light_v1, light_v2);
						
						DiffuseAreaLight light(light_v0, light_v1, light_v2, make_float3(params.color.x, params.color.y, params.color.z));

						L += path_throughput * make_float4(light.L(-current_ray.direction()), 0.0f);

						// Once hitting an object that emits light, we don't proceed further. We quit.
						break;
					}

					// compute direct lighting
					{
						thrust::default_random_engine rng1(seed);
						thrust::uniform_int_distribution<uint32_t> u011(0, scene_buffer.m_num_lights - 1);
						thrust::uniform_real_distribution<float> f011(0.0f, 1.0f);
						
						// Initialize direct lighting to zero.
						Ld = make_float3(0.0f);

						// First get information about the light to be chosen.
						float light_pdf, bsdf_pdf;
						const uint32_t light_id = u011(rng1);
						const uint32_t light_primitive_id = scene_buffer.m_light_ids_to_primitive_ids[light_id];
						float3 light_v0, light_v1, light_v2;

						get_light_vertices(scene_buffer, light_id, light_v0, light_v1, light_v2);

						const uint32_t light_material_id = scene_buffer.m_tri_material_ids[light_primitive_id];
						const Material& light_material = material_buffer.m_materials[light_material_id];

						const DiffuseEmitterParams& params = material_buffer.m_diffuse_emitter_bsdfs[light_material.layer_bsdf_id[0]];
						DiffuseAreaLight light(light_v0, light_v1, light_v2, make_float3(params.color.x, params.color.y, params.color.z));

						// Set the shading frame appropriately.
						shading_frame.set_from_z(isect.m_Ns);

						// Sample light with MIS.
						{
							LightSample sample;
							float3 sampled_wi;
							VisibilityTesterElement e;

							sample.m_u0 = f011(rng1);
							sample.m_u1 = f011(rng1);
							
							float3 Li = light.sample_L(isect.m_position, isect.m_epsilon, sample, sampled_wi, light_pdf, e);
							if (!is_black(Li) && light_pdf > 0.0f)
							{
								Ray shadow_ray = e.m_visibility_ray;
								float3 f = material_f(material_buffer, material_id, -current_ray.direction(), shadow_ray.direction(), isect.m_Ns, shading_frame, BxDFType(BxDF_ALL & ~BxDF_SPECULAR));
								bool occluded = trace_ray(shadow_ray, true, nullptr, bvh.m_nodes, bvh.m_tris, bvh.m_tri_idx, scene_buffer);

								if (!is_black(f) && !occluded)
								{
									//bsdf_pdf = 1.0f;		// evaluate bsdf for material.
									bsdf_pdf = material_pdf(material_buffer, material_id, -current_ray.direction(), sampled_wi, shading_frame, BxDFType(BxDF_ALL & ~BxDF_SPECULAR));
									float weight = balanced_heuristic(1, light_pdf, 1, bsdf_pdf);

									// compute Ld.
									Ld += f * Li * (abs(dot(sampled_wi, isect.m_Ns)) * weight) / light_pdf;
								}
							}
						}

						// Sample bsdf with MIS
						{
							// Sample f
							BsdfSample sample = { f011(rng1), f011(rng1), f011(rng1), 0.0f };
							float3 Li = make_float3(0.0f);

							float3 sampled_wi;
							BxDFType sampled_type;

							float3 f = material_sample_f(material_buffer, material_id, -current_ray.direction(), shading_frame, isect.m_Ns, sample, sampled_wi, bsdf_pdf, BxDFType(BxDF_ALL & ~BxDF_SPECULAR), sampled_type);

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
					}

					L += path_throughput * make_float4(Ld, 0.0f);

					// Sample the bsdf to get the next bounce.
					BsdfSample sample = { f01(rng), f01(rng), f01(rng), 0.0f };

					float3 wi;
					float pdf;
					BxDFType flags;

					float3 f = material_sample_f(material_buffer, material_id, -current_ray.direction(), shading_frame, isect.m_Ns, sample, wi, pdf, BxDF_ALL, flags);

					if (is_black(f) || pdf == 0.0f)
						break;

					specular_bounce = (flags & BxDF_SPECULAR) != 0;
					path_throughput *= make_float4(f * abs(dot(wi, isect.m_Ns)) / pdf, 0.0f);

					current_ray = Ray(isect.m_position, wi, isect.m_epsilon);
					
					// Russian roulette termination.
					if (bounce > rrstart)
					{
						float continue_probability = min(rrprob, luminance(make_float3(path_throughput.x, path_throughput.y, path_throughput.z)));
						if (f01(rng) > continue_probability)
						{
							alive = 0;
						}
						else
						{
							path_throughput /= (1.0f - continue_probability);
						}
					}
					if (bounce == maxbounce)
						break;
					
					// compute intersection and update current ray and current intersection.
					bool next_bounce_intersect = trace_ray(current_ray, false, &isect, bvh.m_nodes, bvh.m_tris, bvh.m_tri_idx, scene_buffer);
					
					if (!next_bounce_intersect)
						break;
					bounce++;
				} while (alive);
				
				color += L;
			}
			
			csb.m_contribution[tidx] = color;
		}
	}

	void IntegratorPath::compute(CameraSampleBuffer* csb, RayBuffer* rb, uint32_t iteration)
	{
		IntersectionBufferClass ibc(m_allocator);
		ibc.allocate(csb->m_size);

		IntersectionBuffer ib = ibc.get_buffer();
		SceneBuffer scene_data = m_scene->gpu_get_buffer();
		BvhStruct bvh = m_tracer->get_bvh();
		MaterialBuffer mb = m_scene->gpu_get_material_buffer();

		m_tracer->trace(*rb, ib, scene_data, nullptr, false);

		dim3 grid_size(256, 1, 1);
		dim3 block_size(256, 1, 1);

		// Depending upon the type of path tracer implementation, we can decide to go ahead with compaction and regeneration.
		CudaTimer t1("path tracing timer");
		t1.start();
		kernel_pathtracer_single_monolithic<<<grid_size, block_size>>>(ib, *csb, *rb, bvh, scene_data, mb, m_params.m_rrstart, m_params.m_max_depth, m_params.m_rrprob, iteration);
		t1.stop();
	}
}
