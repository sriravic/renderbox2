
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
#include <curand_kernel.h>
#include <helper_math.h>
#include <thrust/device_ptr.h>
#include <thrust/random.h>
#include <thrust/reduce.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>

// Standard c++ headers.
#include <random>

namespace renderbox2
{
	
	//
	// BDPT - LVC Multiple Kernel implementations.
	//


	//
	// Util functions.
	//

	template<typename T>
	__device__ __inline__ T get_random(T start, T end, float rand_value)
	{
		return static_cast<T>(floorf(lerp(static_cast<T>(start), static_cast<T>(end), rand_value)));
	}


	// This kernel creates the light samples and starts with a trace of the created light rays.
	__global__ void kernel_mk_light_pass_create_light_samples(
		SceneBuffer scene_buffer,
		MaterialBuffer material_buffer,
		curandState* rng_states,
		RayBuffer rays,
		float4* contributions
		)
	{
		for (GIndexType tidx = threadIdx.x + blockIdx.x * blockDim.x; tidx < rays.m_size; tidx += gridDim.x * blockDim.x)
		{
			// initialize the random number generators.
			uint32_t seed = simplehash(tidx);
			curand_init(1234, tidx, 0, &rng_states[tidx]);
			
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

			contributions[tidx] = make_float4(Le, 0.0f) / pdf;

			// Start tracing the rays through the scene.
			Ray current_ray = Ray(position, direction);
			rays.m_data[tidx] = current_ray;
		}
	}

	// This kernel estimates the bsdf at the current intersection point, updates current partial contribution and computes the next bounce.
	__global__ void kernel_mk_light_pass_evaluate_material_next_bounce(
		SceneBuffer			scene_buffer,
		MaterialBuffer		material_buffer,
		curandState*		rng_states,
		RayBuffer			rays,
		IntersectionBuffer	ib,
		IntersectionBuffer	lvc,
		uint8_t*			alives,
		uint32_t			rrstart,
		uint32_t			max_bounce,
		bool				store_lvc,
		uint32_t			total_samples,
		uint32_t*			current_lvc_size,			// atomic increment of lvc counter to keep track of vertices.
		uint32_t			total_lvc_size,				// if an lvc is allocated, its total size.
		uint32_t            salt						// add some salt to the random number generator.
		)
	{
		for (GIndexType tidx = threadIdx.x + blockIdx.x * blockDim.x; tidx < rays.m_size; tidx += gridDim.x * blockDim.x)
		{
			if (alives[tidx] == 1 && ib.m_intersected[tidx] == 1)
			{
				// Get all the details.
				const GIndexType primitive_id = ib.m_primitive_id[tidx];
				const uint32_t material_id = ib.m_primitive_material_id[tidx];
				const float3 position = make_float3(ib.m_position[0][tidx], ib.m_position[1][tidx], ib.m_position[2][tidx]);
				float4& contribution = ib.m_contribution[tidx];
				const float3 wo_world = make_float3(ib.m_incoming_direction[0][tidx], ib.m_incoming_direction[1][tidx], ib.m_incoming_direction[2][tidx]);
				const float3 Ns = make_float3(ib.m_shading_normal[0][tidx], ib.m_shading_normal[1][tidx], ib.m_shading_normal[2][tidx]);
				const float3 Ng = make_float3(ib.m_geomteric_normal[0][tidx], ib.m_geomteric_normal[1][tidx], ib.m_geomteric_normal[2][tidx]);
				const float2 uv = ib.m_uv[tidx];
				uint32_t& depth = ib.m_depth[tidx];
				const float epsilon = ib.m_epsilon[tidx];

				if (store_lvc)
				{
					// Store all the intermediate data in lvc buffer.
					uint32_t write_index = atomicAdd(current_lvc_size, 1);
					if (write_index < total_lvc_size)
					{
						lvc.m_position[0][write_index] = position.x;
						lvc.m_position[1][write_index] = position.y;
						lvc.m_position[2][write_index] = position.z;
						lvc.m_contribution[write_index] = contribution;
						lvc.m_depth[write_index] = depth + 1;
						lvc.m_epsilon[write_index] = epsilon;
						lvc.m_geomteric_normal[0][write_index] = Ng.x;
						lvc.m_geomteric_normal[1][write_index] = Ng.y;
						lvc.m_geomteric_normal[2][write_index] = Ng.z;
						lvc.m_shading_normal[0][write_index] = Ns.x;
						lvc.m_shading_normal[1][write_index] = Ns.y;
						lvc.m_shading_normal[2][write_index] = Ns.z;
						lvc.m_incoming_direction[0][write_index] = wo_world.x;
						lvc.m_incoming_direction[1][write_index] = wo_world.y;
						lvc.m_incoming_direction[2][write_index] = wo_world.z;
						lvc.m_primitive_id[write_index] = primitive_id;
						lvc.m_uv[write_index] = uv;
					}
					else
					{
						// We've filled the buffer. No point in carrying over the ray.
						alives[tidx] = 0;
						atomicSub(current_lvc_size, 1);
					}
				}

				// Sample current vertex BSDF.
				curandState& local_state = rng_states[tidx];
								
				Frame shading_frame;
				shading_frame.set_from_z(Ns);

				BsdfSample sample = { curand_uniform(&local_state), curand_uniform(&local_state), curand_uniform(&local_state), 0.0f };

				float3 sampled_wi;
				float sampled_pdf;
				BxDFType sampled_type;

				float3 sampled_f = material_sample_f(material_buffer, material_id, wo_world, shading_frame, Ng, sample, sampled_wi, sampled_pdf, BxDF_ALL, sampled_type);

				if (is_black(sampled_f) || sampled_pdf == 0.0f)
				{
					alives[tidx] = 0;
					break;
				}

				// Update contribution.
				float3 temp = sampled_f * abs(dot(Ns, sampled_wi)) / sampled_pdf;

				// If path throughtput falls below a particular level kill the sample.
				float continue_probability = min(1.0f, luminance(temp));
				if (curand_uniform(&local_state) > continue_probability)
				{
					alives[tidx] = 0;
					break;
				}

				contribution *= make_float4(temp, 0.0f) / continue_probability;

				if (is_black(contribution))
				{
					alives[tidx] = 0;
					break;
				}

				// Write the new ray back to the buffer.
				rays.m_data[tidx] = Ray(position, sampled_wi, epsilon);
				depth++;
			}
			else
			{
				alives[tidx] = 0;
			}
		}
	}

	__global__ void kernel_mk_camera_trace(
		IntersectionBuffer	ib,
		IntersectionBuffer	lvc,
		CameraSampleBuffer	csb,
		curandState*		rng_states,
		RayBuffer			rb,
		BvhStruct			bvh,
		SceneBuffer			scene_buffer,
		MaterialBuffer		material_buffer,
		uint8_t*			specular_bounces,
		uint8_t*			alives,
		uint32_t			rrstart,
		uint32_t			maxbounce,
		float				rrprob,
		uint32_t			current_lvc_size,
		uint32_t			num_connections_per_cam_vertex,
		uint32_t			iteration,
		uint32_t			salt								// to be added to the random number generator
		)
	{
		for (GIndexType tidx = threadIdx.x + blockIdx.x * blockDim.x; tidx < csb.m_size; tidx += gridDim.x * blockDim.x)
		{
			float4& contribution = csb.m_contribution[tidx];
			float4 L = make_float4(0.0f);
			
			if (alives[tidx] == 1 && ib.m_intersected[tidx] == 1)
			{
				// Get the intersection data first.
				Ray& current_ray = rb.m_data[tidx];
				const GIndexType primitive_id = ib.m_primitive_id[tidx];
				const uint32_t material_id = ib.m_primitive_material_id[tidx];
				const Material material = material_buffer.m_materials[material_id];
				const float3 position = make_float3(ib.m_position[0][tidx], ib.m_position[1][tidx], ib.m_position[2][tidx]);
				float4& path_throughput = csb.m_throughput[tidx];
				const float3 wo_world = make_float3(ib.m_incoming_direction[0][tidx], ib.m_incoming_direction[1][tidx], ib.m_incoming_direction[2][tidx]);
				const float3 Ns = make_float3(ib.m_shading_normal[0][tidx], ib.m_shading_normal[1][tidx], ib.m_shading_normal[2][tidx]);
				const float3 Ng = make_float3(ib.m_geomteric_normal[0][tidx], ib.m_geomteric_normal[1][tidx], ib.m_geomteric_normal[2][tidx]);
				uint32_t& depth = ib.m_depth[tidx];
				const float epsilon = ib.m_epsilon[tidx];

				// compute connections
				uint32_t seed = simplehash(tidx + iteration + salt);
				curandState& local_state = rng_states[tidx];

				if ((specular_bounces[tidx] == 1 && material.m_emitter == 1) || (depth == 0 && material.m_emitter == 1))
				{
					// Add contribution and exit.
					const uint32_t diffuse_emitter_id = material.layer_bsdf_id[0];
					const DiffuseEmitterParams& params = material_buffer.m_diffuse_emitter_bsdfs[diffuse_emitter_id];

					float3 light_v0, light_v1, light_v2;
					get_triangle_vertices(scene_buffer, primitive_id, light_v0, light_v1, light_v2);

					DiffuseAreaLight light(light_v0, light_v1, light_v2, make_float3(params.color.x, params.color.y, params.color.z));

					L += path_throughput * make_float4(light.L(-current_ray.direction()), 0.0f);

					alives[tidx] = 0;
					contribution += L;
					break;
				}

				float3 Ld = make_float3(0.0f);
				Frame shading_frame;
				shading_frame.set_from_z(Ns);

				// Compute direct lighting rays
				{
					// Initialize direct lighting to zero.
					Ld = make_float3(0.0f);

					// First get information about the light to be chosen.
					float light_pdf, bsdf_pdf;
					const uint32_t light_id = get_random<uint32_t>(0, scene_buffer.m_num_lights - 1, curand_uniform(&local_state));
					const uint32_t light_primitive_id = scene_buffer.m_light_ids_to_primitive_ids[light_id];
					float3 light_v0, light_v1, light_v2;

					get_light_vertices(scene_buffer, light_id, light_v0, light_v1, light_v2);
					
					const uint32_t light_material_id = scene_buffer.m_tri_material_ids[light_primitive_id];
					const Material& light_material = material_buffer.m_materials[light_material_id];

					const DiffuseEmitterParams& params = material_buffer.m_diffuse_emitter_bsdfs[light_material.layer_bsdf_id[0]];
					DiffuseAreaLight light(light_v0, light_v1, light_v2, make_float3(params.color.x, params.color.y, params.color.z));

					// Set the shading frame appropriately.
					shading_frame.set_from_z(Ns);

					// Sample light with MIS.
					{
						LightSample sample;
						float3 sampled_wi;
						VisibilityTesterElement e;

						sample.m_u0 = curand_uniform(&local_state);
						sample.m_u1 = curand_uniform(&local_state);

						float3 Li = light.sample_L(position, epsilon, sample, sampled_wi, light_pdf, e);
						if (!is_black(Li) && light_pdf > 0.0f)
						{
							Ray shadow_ray = e.m_visibility_ray;
							float3 f = material_f(material_buffer, material_id, -current_ray.direction(), shadow_ray.direction(), Ng, shading_frame, BxDFType(BxDF_ALL & ~BxDF_SPECULAR));
							bool occluded = trace_ray(shadow_ray, true, nullptr, bvh.m_nodes, bvh.m_tris, bvh.m_tri_idx, scene_buffer);

							if (!is_black(f) && !occluded)
							{
								//bsdf_pdf = 1.0f;		// evaluate bsdf for material.
								bsdf_pdf = material_pdf(material_buffer, material_id, -current_ray.direction(), sampled_wi, shading_frame, BxDFType(BxDF_ALL & ~BxDF_SPECULAR));
								float weight = balanced_heuristic(1, light_pdf, 1, bsdf_pdf);

								// compute Ld.
								Ld += f * Li * (abs(dot(sampled_wi, Ns)) * weight) / light_pdf;
							}
						}
					}

					// Sample bsdf with MIS
					{
						// Sample f
						BsdfSample sample = { curand_uniform(&local_state), curand_uniform(&local_state), curand_uniform(&local_state), 0.0f };
						float3 Li = make_float3(0.0f);

						float3 sampled_wi;
						BxDFType sampled_type;

						float3 f = material_sample_f(material_buffer, material_id, -current_ray.direction(), shading_frame, Ng, sample, sampled_wi, bsdf_pdf, BxDFType(BxDF_ALL & ~BxDF_SPECULAR), sampled_type);

						bool flag = true;
						if (!is_black(f) && bsdf_pdf > 0.0f)
						{
							float weight = 1.0f;
							if (!(sampled_type & BxDF_SPECULAR))
							{
								// Compute light generating the same direction pdf from intersected point.
								light_pdf = light.pdf(position, sampled_wi);
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
								Ray shadow_ray(position, sampled_wi, epsilon);

								bool intersected = trace_ray(shadow_ray, false, &light_isect, bvh.m_nodes, bvh.m_tris, bvh.m_tri_idx, scene_buffer);

								if (light_isect.m_intersected == 1 && (light_isect.m_primitive_id == light_primitive_id))
								{
									Li = light.L(-sampled_wi);
								}

								Ld += f * Li * (abs(dot(sampled_wi, Ns)) * weight) / bsdf_pdf;
							}
						}
					}

					Ld *= scene_buffer.m_num_lights;
				}

				L += path_throughput * make_float4(Ld, 0.0f);

				for (uint32_t i = 0; i < num_connections_per_cam_vertex; i++)
				{
					// Choose a particular number of lvcs.
					uint32_t lvc_index = get_random<uint32_t>(0, current_lvc_size - 1, curand_uniform(&local_state));

					float3 lvc_position = make_float3(lvc.m_position[0][lvc_index], lvc.m_position[1][lvc_index], lvc.m_position[2][lvc_index]);
					VisibilityTesterElement e;
					e.set_segment(position, epsilon, lvc_position, 1e-3f);

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
						float3 f = material_f(material_buffer, material_id, -current_ray.direction(), e.m_visibility_ray.direction(), Ng, shading_frame, BxDF_ALL);

						// compute G.
						float G = abs(dot(Ns, e.m_visibility_ray.direction())) * abs(dot(lvc_ns, -e.m_visibility_ray.direction())) / distance2(position, lvc_position);

						float weight = 1.0f / ((depth + 1) + lvc.m_depth[lvc_index]);

						// appropriately weight them and add to the pool.
						L += path_throughput * light_path_contrib * make_float4(f, 0.0f) * G * make_float4(lvc_f, 0.0f) * weight;
					}
				}

				// Compute next bounce.
				BsdfSample sample = { curand_uniform(&local_state), curand_uniform(&local_state), curand_uniform(&local_state), 0.0f };

				float3 wi;
				float pdf;
				BxDFType flags;

				float3 f = material_sample_f(material_buffer, material_id, -current_ray.direction(), shading_frame, Ns, sample, wi, pdf, BxDF_ALL, flags);

				if (is_black(f) || pdf == 0.0f)
				{
					alives[tidx] = 0;
					contribution += L;
					break;
				}

				bool specular_bounce = (flags & BxDF_SPECULAR) != 0;
				specular_bounces[tidx] = specular_bounce ? 1 : 0;

				path_throughput *= make_float4(f * abs(dot(wi, Ns)) / pdf, 0.0f);

				// Russian roulette termination.
				if (depth > rrstart)
				{
					float continue_probability = min(rrprob, luminance(make_float3(path_throughput.x, path_throughput.y, path_throughput.z)));
					if (curand_uniform(&local_state) > continue_probability)
					{
						alives[tidx] = 0;
						contribution += L;
						break;
					}
					else
					{
						path_throughput /= (1.0f - continue_probability);
					}
				}
				if (depth == maxbounce)
				{
					alives[tidx] = 0;
					contribution += L;
					break;
				}

				// Write the next ray, update the bounce, contribution.
				current_ray = Ray(position, wi, epsilon);
				depth++;
			}
			else
			{
				// Kill the sample.
				alives[tidx] = 0;
			}
			contribution += L;
		}
	}

	__global__ void kernel_mk_initialize_rng_states(curandState* rng_states, GIndexType size)
	{
		for (GIndexType tidx = threadIdx.x + blockIdx.x * blockDim.x; tidx < size; tidx += gridDim.x * blockDim.x)
		{
			uint32_t seed = simplehash(tidx);
			curand_init(1234, tidx, 0, &rng_states[tidx]);
		}
	}

	void IntegratorBdptLvc::method_mk_light_preprocess(CameraSampleBuffer* csb, RayBuffer* rb, uint32_t iteration)
	{
		uint32_t num_alives = m_params.m_num_prep_paths;
		DevicePointer alives = m_allocator.allocate(sizeof(uint8_t) * m_params.m_num_prep_paths);
		
		DevicePointer rng_states = m_allocator.allocate(sizeof(curandState) * m_params.m_num_prep_paths);

		RayBufferClass rbc(m_allocator);
		IntersectionBufferClass ibc(m_allocator);
		IntersectionBufferClass dummy(m_allocator);

		rbc.allocate(m_params.m_num_prep_paths);
		ibc.allocate(m_params.m_num_prep_paths);
		dummy.allocate(0);

		RayBuffer lrb = rbc.get_buffer();
		IntersectionBuffer ib = ibc.get_buffer();
		IntersectionBuffer lvc = dummy.get_buffer();
		SceneBuffer scene_data = m_scene->gpu_get_buffer();
		MaterialBuffer mb = m_scene->gpu_get_material_buffer();

		// we need to add salt to the default random number generator.
		std::default_random_engine rng;
		std::uniform_int_distribution<uint32_t> u01(0, UINT32_MAX);

		thrust::fill(thrust::device_ptr<uint8_t>(static_cast<uint8_t*>(alives.m_ptr)), thrust::device_ptr<uint8_t>(static_cast<uint8_t*>(alives.m_ptr)) + m_params.m_num_prep_paths, 1);
		thrust::fill(thrust::device_ptr<uint32_t>(static_cast<uint32_t*>(ib.m_depth)), thrust::device_ptr<uint32_t>(static_cast<uint32_t*>(ib.m_depth)) + m_params.m_num_prep_paths, 0);

		dim3 grid_size(256, 1, 1);
		dim3 block_size(256, 1, 1);

		CudaTimer t1("timer");
		
		// Create light samples and trace the first batch.
		t1.start();
		kernel_mk_light_pass_create_light_samples<<<grid_size, block_size>>>(scene_data, mb, static_cast<curandState*>(rng_states.m_ptr), lrb, static_cast<float4*>(ib.m_contribution));
		t1.stop();
		uint32_t deep = 0;
		while (num_alives > 0)
		{
			// Trace.
			m_tracer->trace(lrb, ib, scene_data, static_cast<uint8_t*>(alives.m_ptr), false);

			// Compute BRDFs and next bounce.
			t1.start();
			kernel_mk_light_pass_evaluate_material_next_bounce<<<grid_size, block_size>>>(scene_data, mb, static_cast<curandState*>(rng_states.m_ptr), lrb, ib, lvc, static_cast<uint8_t*>(alives.m_ptr),
				m_params.m_light_path_rrstart, m_params.m_num_light_path_max_depth, false, m_params.m_num_prep_paths, nullptr, 0, u01(rng));
			t1.stop();
			
			// Get if any path is still alive.
			num_alives = thrust::count(thrust::device_ptr<uint8_t>(static_cast<uint8_t*>(alives.m_ptr)), thrust::device_ptr<uint8_t>(static_cast<uint8_t*>(alives.m_ptr)) + m_params.m_num_prep_paths, 1);
			
			deep++;
		}
		
		// Compute average path length.
		uint32_t total_bounces = thrust::reduce(thrust::device_ptr<uint32_t>(ib.m_depth), thrust::device_ptr<uint32_t>(ib.m_depth) + m_params.m_num_prep_paths);

		float avg_path_length = total_bounces / static_cast<float>(m_params.m_num_prep_paths);
		m_num_lvc_vertices = (uint32_t)ceil(1.1f * avg_path_length * m_params.m_num_light_paths);

		cout << "Average path length : " << avg_path_length << endl;
		cout << "Total allocated : " << m_num_lvc_vertices << endl;

		m_lvc_class.allocate(m_num_lvc_vertices);

		// Free all allocated memory.
		m_allocator.free(alives);
		m_allocator.free(rng_states);
	}

	void IntegratorBdptLvc::method_mk_light_trace(CameraSampleBuffer* csb, RayBuffer* rb, uint32_t iteration)
	{
		uint32_t num_alives = m_params.m_num_light_paths;
		DevicePointer alives = m_allocator.allocate(sizeof(uint8_t) * m_params.m_num_light_paths);
		DevicePointer current_lvc_size = m_allocator.allocate(sizeof(uint32_t));
		DevicePointer rng_states = m_allocator.allocate(sizeof(curandState) * m_params.m_num_light_paths);

		RayBufferClass rbc(m_allocator);
		IntersectionBufferClass ibc(m_allocator);
		
		rbc.allocate(m_params.m_num_light_paths);
		ibc.allocate(m_params.m_num_light_paths);
		m_lvc_class.allocate(m_num_lvc_vertices);

		RayBuffer lrb = rbc.get_buffer();
		IntersectionBuffer ib = ibc.get_buffer();
		IntersectionBuffer lvc = m_lvc_class.get_buffer();
		SceneBuffer scene_data = m_scene->gpu_get_buffer();
		MaterialBuffer mb = m_scene->gpu_get_material_buffer();

		thrust::fill(thrust::device_ptr<uint8_t>(static_cast<uint8_t*>(alives.m_ptr)), thrust::device_ptr<uint8_t>(static_cast<uint8_t*>(alives.m_ptr)) + m_params.m_num_light_paths, 1);
		thrust::fill(thrust::device_ptr<uint32_t>(static_cast<uint32_t*>(ib.m_depth)), thrust::device_ptr<uint32_t>(static_cast<uint32_t*>(ib.m_depth)) + m_params.m_num_light_paths, 0);

		// we need to add salt to the default random number generator.
		std::default_random_engine rng;
		std::uniform_int_distribution<uint32_t> u01(0, UINT32_MAX);

		dim3 grid_size(256, 1, 1);
		dim3 block_size(256, 1, 1);

		CudaTimer t1("timer");

		// Initialize the rng states.
		t1.start();
		kernel_mk_initialize_rng_states<<<grid_size, block_size>>>(static_cast<curandState*>(rng_states.m_ptr), m_params.m_num_light_paths);
		t1.stop();
		
		// Create light samples and trace the first batch.
		t1.start();
		kernel_mk_light_pass_create_light_samples<<<grid_size, block_size>>>(scene_data, mb, static_cast<curandState*>(rng_states.m_ptr), lrb, static_cast<float4*>(ib.m_contribution));
		t1.stop();
		
		while (num_alives > 0)
		{
			// Trace.
			m_tracer->trace(lrb, ib, scene_data, static_cast<uint8_t*>(alives.m_ptr), false);

			// Compute BRDFs and next bounce.
			t1.start();
			kernel_mk_light_pass_evaluate_material_next_bounce<<<grid_size, block_size >>>(scene_data, mb, static_cast<curandState*>(rng_states.m_ptr), lrb, ib, lvc, static_cast<uint8_t*>(alives.m_ptr),
				m_params.m_light_path_rrstart, m_params.m_num_light_path_max_depth, true, m_params.m_num_light_paths, static_cast<uint32_t*>(current_lvc_size.m_ptr), m_num_lvc_vertices, u01(rng));
			t1.stop();

			// Get if any path is still alive.
			num_alives = thrust::count(thrust::device_ptr<uint8_t>(static_cast<uint8_t*>(alives.m_ptr)), thrust::device_ptr<uint8_t>(static_cast<uint8_t*>(alives.m_ptr)) + m_params.m_num_light_paths, 1);
		}
		
		checkCuda(cudaMemcpy(&m_num_filled_vertices, current_lvc_size.m_ptr, sizeof(uint32_t), cudaMemcpyDeviceToHost));

		// Free all allocated memory.
		m_allocator.free(current_lvc_size);
		m_allocator.free(alives);
	}

	void IntegratorBdptLvc::method_mk_camera_trace(CameraSampleBuffer* csb, RayBuffer* rb, uint32_t iteration)
	{
		cout << "Camera Trace" << endl;

		IntersectionBufferClass ibc(m_allocator);
		ibc.allocate(csb->m_size);
		IntersectionBuffer ib = ibc.get_buffer();
		IntersectionBuffer lvc = m_lvc_class.get_buffer();

		SceneBuffer scene_data = m_scene->gpu_get_buffer();
		BvhStruct bvh = m_tracer->get_bvh();
		MaterialBuffer mb = m_scene->gpu_get_material_buffer();

		DevicePointer specular_bounces = m_allocator.allocate(sizeof(uint8_t) * csb->m_size);
		DevicePointer alives = m_allocator.allocate(sizeof(uint8_t) * csb->m_size);
		DevicePointer rng_states = m_allocator.allocate(sizeof(curandState) * csb->m_size);

		thrust::fill(thrust::device_ptr<uint8_t>(static_cast<uint8_t*>(alives.m_ptr)), thrust::device_ptr<uint8_t>(static_cast<uint8_t*>(alives.m_ptr)) + csb->m_size, 1);
		thrust::fill(thrust::device_ptr<uint32_t>(static_cast<uint32_t*>(ib.m_depth)), thrust::device_ptr<uint32_t>(static_cast<uint32_t*>(ib.m_depth)) + csb->m_size, 0);
		thrust::fill(thrust::device_ptr<uint8_t>(static_cast<uint8_t*>(specular_bounces.m_ptr)), thrust::device_ptr<uint8_t>(static_cast<uint8_t*>(specular_bounces.m_ptr) + csb->m_size), 0);

		dim3 grid_size(256, 1, 1);
		dim3 block_size(256, 1, 1);

		// initialize the random number generators.
		CudaTimer rng_timer("Rng Init Timer");
		rng_timer.start();
		kernel_mk_initialize_rng_states<<<grid_size, block_size>>>(static_cast<curandState*>(rng_states.m_ptr), csb->m_size);
		rng_timer.stop();

		// Create salt for random number generator
		GIndexType num_alives = csb->m_size;
		std::default_random_engine rng;
		std::uniform_int_distribution<uint32_t> u01(0, UINT32_MAX);

		while (num_alives > 0)
		{
			// Trace primary rays.
			m_tracer->trace(*rb, ib, scene_data, static_cast<uint8_t*>(alives.m_ptr), false);

			// Camera trace
			CudaTimer t1("camera timer");
			t1.start();
			kernel_mk_camera_trace<<<grid_size, block_size>>>(ib, lvc, *csb, static_cast<curandState*>(rng_states.m_ptr), *rb, bvh, scene_data, mb, static_cast<uint8_t*>(specular_bounces.m_ptr),
				static_cast<uint8_t*>(alives.m_ptr), m_params.m_cam_path_rrstart, m_params.m_num_cam_path_max_depth, m_params.m_cam_path_rrprob, m_num_filled_vertices,
				m_params.m_num_cam_path_connections, iteration, 0);
			t1.stop();

			num_alives = thrust::count(thrust::device_ptr<uint8_t>(static_cast<uint8_t*>(alives.m_ptr)), thrust::device_ptr<uint8_t>(static_cast<uint8_t*>(alives.m_ptr)) + csb->m_size, 1);
		}

		m_allocator.free(rng_states);
		m_allocator.free(specular_bounces);
		m_allocator.free(alives);
	}
}
