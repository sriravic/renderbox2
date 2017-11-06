
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
#include <accelerators/sbvh/cudatracer.h>
#include <core/buffer.h>
#include <core/intersection.h>
#include <core/lights.h>
#include <core/montecarlo.h>
#include <core/params.h>
#include <core/util.h>
#include <integrators/integrator_bdpt_lvc.h>
#include <util/cudatimer.h>

// Cuda specific headers.
#include <thrust/device_ptr.h>
#include <thrust/random.h>
#include <thrust/reduce.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>

// Standard c++ headers.

namespace renderbox2
{

	//
	// Single kernel implementations.
	//

	__global__ void kernel_sk_light_trace(
		SceneBuffer			scene_buffer,
		MaterialBuffer		material_buffer,
		BvhStruct			bvh,
		IntersectionBuffer	lvc,
		uint32_t			rrstart,
		uint32_t			max_bounce,
		bool				store_lvc,					// if this flag is enabled we store the per intersection data in the lvc.
		uint32_t*			depths,						// per ray max depth to compute average path length (NOTE: can be null in case of actual light pass).
		uint32_t			total_samples,
		uint32_t*			current_lvc_size,			// atomic increment of lvc counter to keep track of vertices.
		uint32_t			total_lvc_size				// if an lvc is allocated, its total size.
		)
	{
		for (GIndexType tidx = threadIdx.x + blockIdx.x * blockDim.x; tidx < total_samples; tidx += gridDim.x * blockDim.x)
		{
			// Choose a light.
			uint32_t seed = simplehash(tidx);
			thrust::default_random_engine rng(seed);
			thrust::uniform_int_distribution<uint32_t> i01(0, scene_buffer.m_num_lights - 1);
			thrust::uniform_real_distribution<float> f01(0.0f, 1.0f);

			uint32_t light_id = i01(rng);
			GIndexType primitive_id = scene_buffer.m_light_ids_to_primitive_ids[light_id];
			uint32_t material_id = scene_buffer.m_tri_material_ids[primitive_id];
			Material material = material_buffer.m_materials[material_id];

			const DiffuseEmitterParams params = material_buffer.m_diffuse_emitter_bsdfs[material.layer_bsdf_id[0]];

			float3 v0, v1, v2;
			float3 position, direction;
			float pdf;

			get_light_vertices(scene_buffer, light_id, v0, v1, v2);
			DiffuseAreaLight l(v0, v1, v2, make_float3(params.color.x, params.color.y, params.color.z));
			LightSample sample = { f01(rng), f01(rng), f01(rng) };
			float3 Le = l.Le(sample, make_float2(f01(rng), f01(rng)), position, direction, pdf);

			float4 contribution = make_float4(Le, 0.0f) / pdf;

			// Start tracing the rays through the scene.
			Ray current_ray = Ray(position, direction);
			bool alive = true;
			Intersection isect;
			uint32_t depth = 0;
			Frame shading_frame;
			do
			{
				bool intersected = trace_ray(current_ray, false, &isect, bvh.m_nodes, bvh.m_tris, bvh.m_tri_idx, scene_buffer);

				if (!intersected)
					break;

				// Store lvc data.
				if (store_lvc)
				{
					uint32_t write_index = atomicAdd(current_lvc_size, 1);
					if (write_index < total_lvc_size)
					{
						lvc.m_position[0][write_index] = isect.m_position.x;
						lvc.m_position[1][write_index] = isect.m_position.y;
						lvc.m_position[2][write_index] = isect.m_position.z;
						lvc.m_contribution[write_index] = contribution;
						lvc.m_depth[write_index] = depth  + 1;
						lvc.m_epsilon[write_index] = isect.m_epsilon;
						lvc.m_geomteric_normal[0][write_index] = isect.m_Ng.x;
						lvc.m_geomteric_normal[1][write_index] = isect.m_Ng.y;
						lvc.m_geomteric_normal[2][write_index] = isect.m_Ng.z;
						lvc.m_shading_normal[0][write_index] = isect.m_Ns.x;
						lvc.m_shading_normal[1][write_index] = isect.m_Ns.y;
						lvc.m_shading_normal[2][write_index] = isect.m_Ns.z;
						lvc.m_incoming_direction[0][write_index] = -current_ray.direction().x;
						lvc.m_incoming_direction[1][write_index] = -current_ray.direction().y;
						lvc.m_incoming_direction[2][write_index] = -current_ray.direction().z;
						lvc.m_primitive_id[write_index] = isect.m_primitive_id;
						lvc.m_uv[write_index] = isect.m_uv;
					}
					else
					{
						// We've filled the buffer. No point in continuing this ray any further.
						atomicSub(current_lvc_size, 1);
						alive = false;
					}
				}
				else
				{
					// We store only the depth.
					depths[tidx] = depth + 1;
				}

				material_id = scene_buffer.m_tri_material_ids[isect.m_primitive_id];
				shading_frame.set_from_z(isect.m_Ns);

				// Sample the bsdf and get next direction
				BsdfSample sample = { f01(rng), f01(rng), f01(rng) };
				float3 wi;
				BxDFType sampled_type;

				float3 f = material_sample_f(material_buffer, material_id, -current_ray.direction(), shading_frame, isect.m_Ng, sample, wi, pdf, BxDF_ALL, sampled_type);

				if (is_black(f) || pdf == 0.0f)
					break;

				// update the throughtput.
				float3 temp = f * abs(dot(isect.m_Ns, wi)) / pdf;
				
				// If path throughtput falls below a particular level kill the sample.
				float continue_probability = min(1.0f, luminance(temp));
				if (f01(rng) > continue_probability)
				{
					break;
				}

				contribution *= make_float4(temp, 0.0f) / continue_probability;
				
				// Create next bounce ray.
				current_ray = Ray(isect.m_position, wi, isect.m_epsilon);
				depth++;

			} while (alive && !is_black(contribution));
		}
	}

	__global__ void kernel_sk_camera_trace(
		IntersectionBuffer	ib,
		IntersectionBuffer	lvc,
		CameraSampleBuffer	csb,
		RayBuffer			rb,
		BvhStruct			bvh,
		SceneBuffer			scene_buffer,
		MaterialBuffer		material_buffer,
		uint32_t			rrstart,
		uint32_t			maxbounce,
		float				rrprob,
		uint32_t			current_lvc_size,
		uint32_t			num_connections_per_cam_vertex,
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
				thrust::uniform_int_distribution<uint32_t> u02(0, current_lvc_size - 1);
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
					shading_frame.set_from_z(isect.m_Ns);
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
								float3 f = material_f(material_buffer, material_id, -current_ray.direction(), shadow_ray.direction(), isect.m_Ng, shading_frame, BxDFType(BxDF_ALL & ~BxDF_SPECULAR));
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

							float3 f = material_sample_f(material_buffer, material_id, -current_ray.direction(), shading_frame, isect.m_Ng, sample, sampled_wi, bsdf_pdf, BxDFType(BxDF_ALL & ~BxDF_SPECULAR), sampled_type);

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

					if (current_lvc_size != 0)
					{
						for (uint32_t i = 0; i < num_connections_per_cam_vertex; i++)
						{
							// choose one lvc
							uint32_t lvc_index = u02(rng);

							float3 lvc_position = make_float3(lvc.m_position[0][lvc_index], lvc.m_position[1][lvc_index], lvc.m_position[2][lvc_index]);
							VisibilityTesterElement e;
							e.set_segment(isect.m_position, isect.m_epsilon, lvc_position, 1e-3f);

							bool occluded = trace_ray(e.m_visibility_ray, true, nullptr, bvh.m_nodes, bvh.m_tris, bvh.m_tri_idx, scene_buffer);

							if (!occluded)
							{
								float4 light_path_contrib = lvc.m_contribution[lvc_index];

								const uint32_t light_primitive_id = lvc.m_primitive_id[lvc_index];
								const uint32_t lvc_material_id = scene_buffer.m_tri_material_ids[light_primitive_id];
								const float3 lvc_wo = make_float3(lvc.m_incoming_direction[0][lvc_index], lvc.m_incoming_direction[1][lvc_index], lvc.m_incoming_direction[2][lvc_index]);
								const float3 lvc_ng = make_float3(lvc.m_geomteric_normal[0][lvc_index], lvc.m_geomteric_normal[1][lvc_index], lvc.m_geomteric_normal[2][lvc_index]);
								const float3 lvc_ns = make_float3(lvc.m_shading_normal[0][lvc_index], lvc.m_shading_normal[1][lvc_index], lvc.m_shading_normal[2][lvc_index]);

								Frame lvc_shading_frame;
								lvc_shading_frame.set_from_z(lvc_ns);

								// compute light bsdf
								// NOTE: Even specular connections work correctly here as the f would return a value of zero. no matter what.
								float3 lvc_f = material_f(material_buffer, lvc_material_id, lvc_wo, -e.m_visibility_ray.direction(), lvc_ng, lvc_shading_frame, BxDF_ALL);
								float3 f = material_f(material_buffer, material_id, -current_ray.direction(), e.m_visibility_ray.direction(), isect.m_Ng, shading_frame, BxDF_ALL);

								// compute G.
								float G = abs(dot(isect.m_Ns, e.m_visibility_ray.direction())) * abs(dot(lvc_ns, -e.m_visibility_ray.direction())) / distance2(isect.m_position, lvc_position);

								float weight = 1.0f / ((bounce + 1) + lvc.m_depth[lvc_index]);

								// appropriately weight them and add to the pool.
								L += path_throughput * light_path_contrib * make_float4(f, 0.0f) * G * make_float4(lvc_f, 0.0f) * weight;
							}
						}
					}

					// Sample the bsdf to get the next bounce.
					BsdfSample sample = { f01(rng), f01(rng), f01(rng), 0.0f };

					float3 wi;
					float pdf;
					BxDFType flags;

					float3 f = material_sample_f(material_buffer, material_id, -current_ray.direction(), shading_frame, isect.m_Ng, sample, wi, pdf, BxDF_ALL, flags);

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

	void IntegratorBdptLvc::method_sk_light_trace(CameraSampleBuffer* csb, RayBuffer* rb, uint32_t iteration)
	{
		IntersectionBufferClass dummy(m_allocator);
		dummy.allocate(0);
		IntersectionBuffer lvc = dummy.get_buffer();

		SceneBuffer scene_data = m_scene->gpu_get_buffer();
		BvhStruct bvh = m_tracer->get_bvh();
		MaterialBuffer mb = m_scene->gpu_get_material_buffer();

		// Allocate depth device pointer for storing path lengths.
		DevicePointer depths = m_allocator.allocate(sizeof(uint32_t) * m_params.m_num_prep_paths);

		dim3 grid_size(256, 1, 1);
		dim3 block_size(256, 1, 1);

		uint32_t* filled_lvc_vertices = nullptr;
		checkCuda(cudaMalloc((void**)&filled_lvc_vertices, sizeof(uint32_t)));
		checkCuda(cudaMemset(filled_lvc_vertices, 0, sizeof(uint32_t)));

		// Prep phase
		cout << "Preparation Phase" << endl;
		CudaTimer t1("Light Trace");
		t1.start();
		kernel_sk_light_trace<<<grid_size, block_size>>>(scene_data, mb, bvh, lvc, m_params.m_light_path_rrstart, m_params.m_num_light_path_max_depth,
			false, static_cast<uint32_t*>(depths.m_ptr), m_params.m_num_prep_paths, nullptr, 0);
		t1.stop();

		// compute average path length and allocate buffer.
		uint32_t total_bounces = thrust::reduce(thrust::device_ptr<uint32_t>(static_cast<uint32_t*>(depths.m_ptr)),
			thrust::device_ptr<uint32_t>(static_cast<uint32_t*>(depths.m_ptr)) + m_params.m_num_prep_paths);

		float avg_path_length = total_bounces / static_cast<float>(m_params.m_num_prep_paths);
		uint32_t total_lvc_allocated = (uint32_t)ceil(1.1f * avg_path_length * m_params.m_num_light_paths);

		cout << "Average path length : " << avg_path_length << endl;
		cout << "Total allocated : " << total_lvc_allocated << endl;

		m_lvc_class.allocate(total_lvc_allocated);
		lvc = m_lvc_class.get_buffer();

		cout << "Light Trace" << endl;

		t1.start();
		kernel_sk_light_trace<<<grid_size, block_size >>>(scene_data, mb, bvh, lvc, m_params.m_light_path_rrstart, m_params.m_num_light_path_max_depth,
			true, nullptr, m_params.m_num_light_paths, filled_lvc_vertices, total_lvc_allocated);
		t1.stop();

		checkCuda(cudaMemcpy(&m_num_filled_vertices, filled_lvc_vertices, sizeof(uint32_t), cudaMemcpyDeviceToHost));

		cout << "Filled lvc vertices count : " << m_num_filled_vertices << endl;
	}

	void IntegratorBdptLvc::method_sk_camera_trace(CameraSampleBuffer* csb, RayBuffer* rb, uint32_t iteration)
	{
		IntersectionBufferClass ibc(m_allocator);
		ibc.allocate(csb->m_size);
		IntersectionBuffer ib = ibc.get_buffer();
		IntersectionBuffer lvc = m_lvc_class.get_buffer();

		SceneBuffer scene_data = m_scene->gpu_get_buffer();
		BvhStruct bvh = m_tracer->get_bvh();
		MaterialBuffer mb = m_scene->gpu_get_material_buffer();

		dim3 grid_size(256, 1, 1);
		dim3 block_size(256, 1, 1);

		// Trace primary rays.
		m_tracer->trace(*rb, ib, scene_data, nullptr, false);

		cout << "Camera Trace" << endl;
		// Camera trace
		CudaTimer t1("camera timer");
		t1.start();
		kernel_sk_camera_trace <<<grid_size, block_size>>>(ib, lvc, *csb, *rb, bvh, scene_data, mb, m_params.m_cam_path_rrstart, m_params.m_num_cam_path_max_depth,
			m_params.m_cam_path_rrprob, m_num_filled_vertices, m_params.m_num_cam_path_connections, iteration);
		t1.stop();
	}
}
