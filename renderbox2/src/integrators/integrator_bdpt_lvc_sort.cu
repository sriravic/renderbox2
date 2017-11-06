
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
#include <random>

namespace renderbox2
{

	//
	// BDPT - LVC Sort based implementation.
	//


	//
	// Custom Functors used by all the kernels.
	//

	struct IsIntersected
	{
		__device__ __host__ bool operator() (const uint8_t intersected)
		{
			return intersected == 1;
		}
	};

	struct IsAlive
	{
		__device__ __host__ bool operator()(const uint8_t alive)
		{
			return alive == 1;
		}
	};


	//
	// Kernels used by the application.
	//


	__global__ void kernel_sort_light_pass_create_light_samples(
		SceneBuffer scene_buffer,
		MaterialBuffer material_buffer,
		RayBuffer rays,
		float4* contributions
		)
	{
		for (GIndexType tidx = threadIdx.x + blockIdx.x * blockDim.x; tidx < rays.m_size; tidx += gridDim.x * blockDim.x)
		{
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

			contributions[tidx] = make_float4(Le, 0.0f) / pdf;

			// Start tracing the rays through the scene.
			Ray current_ray = Ray(position, direction);
			rays.m_data[tidx] = current_ray;
		}
	}

	__global__ void kernel_sort_light_pass_evaluate_material_next_bounce(
		SceneBuffer			scene_buffer,
		MaterialBuffer		material_buffer,
		RayBuffer			rays,
		IntersectionBuffer	ib,
		IntersectionBuffer	lvc,
		uint8_t*			alives,
		GIndexType*			rayids,
		uint32_t			rrstart,
		uint32_t			max_bounce,
		bool				store_lvc,
		GIndexType			num_rays,
		uint32_t			total_samples,
		uint32_t*			current_lvc_size,			// atomic increment of lvc counter to keep track of vertices.
		uint32_t			total_lvc_size,				// if an lvc is allocated, its total size.
		uint32_t            salt						// add some salt to the random number generator.
		)
	{
		for (GIndexType tidx = threadIdx.x + blockIdx.x * blockDim.x; tidx < num_rays; tidx += gridDim.x * blockDim.x)
		{
			// Get intersection data
			GIndexType rayid = rayids[tidx];
			const GIndexType primitive_id = ib.m_primitive_id[rayid];
			const uint32_t material_id = ib.m_primitive_material_id[rayid];
			const float3 position = make_float3(ib.m_position[0][rayid], ib.m_position[1][rayid], ib.m_position[2][rayid]);
			float4& contribution = ib.m_contribution[rayid];
			const float3 wo_world = make_float3(ib.m_incoming_direction[0][rayid], ib.m_incoming_direction[1][rayid], ib.m_incoming_direction[2][rayid]);
			const float3 Ns = make_float3(ib.m_shading_normal[0][rayid], ib.m_shading_normal[1][rayid], ib.m_shading_normal[2][rayid]);
			const float3 Ng = make_float3(ib.m_geomteric_normal[0][rayid], ib.m_geomteric_normal[1][rayid], ib.m_geomteric_normal[2][rayid]);
			const float2 uv = ib.m_uv[rayid];
			uint32_t& depth = ib.m_depth[rayid];
			const float epsilon = ib.m_epsilon[rayid];

			// Evaluate bsdfs
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
					alives[rayid] = 0;
					ib.m_intersected[rayid] = 0;
					atomicSub(current_lvc_size, 1);
				}
			}

			// Sample current vertex BSDF.
			uint32_t seed = simplehash(rayid + salt);
			thrust::default_random_engine rng(seed);
			thrust::uniform_real_distribution<float> f01(0.0f, 1.0f);

			Frame shading_frame;
			shading_frame.set_from_z(Ns);

			BsdfSample sample = { f01(rng), f01(rng), f01(rng), 0.0f };

			float3 sampled_wi;
			float sampled_pdf;
			BxDFType sampled_type;

			float3 sampled_f = material_sample_f(material_buffer, material_id, wo_world, shading_frame, Ng, sample, sampled_wi, sampled_pdf, BxDF_ALL, sampled_type);

			if (is_black(sampled_f) || sampled_pdf == 0.0f)
			{
				alives[rayid] = 0;
				ib.m_intersected[rayid] = 0;
				break;
			}

			// Update contribution.
			float3 temp = sampled_f * abs(dot(Ns, sampled_wi)) / sampled_pdf;

			// If path throughtput falls below a particular level kill the sample.
			float continue_probability = min(1.0f, luminance(temp));
			if (f01(rng) > continue_probability)
			{
				alives[rayid] = 0;
				ib.m_intersected[rayid] = 0;
				break;
			}

			contribution *= make_float4(temp, 0.0f) / continue_probability;

			if (is_black(contribution))
			{
				alives[rayid] = 0;
				ib.m_intersected[rayid] = 0;
				break;
			}

			// Write the new ray back to the buffer.
			rays.m_data[rayid] = Ray(position, sampled_wi, epsilon);
			depth++;
		}
	}


	__global__ void kernel_compute_material_ranges(
		uint32_t* material_ids,
		uint2*  ranges,
		GIndexType N
		)
	{
		for (GIndexType tidx = threadIdx.x + blockIdx.x * blockDim.x; tidx < N; tidx += gridDim.x * blockDim.x)
		{
			uint32_t current_material_id = material_ids[tidx];
			if (tidx != 0 && tidx != N - 1)
			{
				const uint32_t next_material_id = material_ids[tidx + 1];
				const uint32_t prev_material_id = material_ids[tidx - 1];

				if (current_material_id != prev_material_id)
				{
					ranges[current_material_id].x = tidx;
				}
				if (current_material_id != next_material_id)
				{
					ranges[current_material_id].y = tidx;
				}
			}

			// Only the start and end of the entire range.
			if (tidx == 0)
			{
				ranges[current_material_id].x = tidx;
			}
			if (tidx == N - 1)
			{
				ranges[current_material_id].y = tidx;
			}
		}
	}

	__global__ void kernel_evaluate_materials(
		SceneBuffer		scene_buffer,
		MaterialBuffer	material_buffer,
		GIndexType		offset,
		GIndexType		total_size
		)
	{
		for (GIndexType tidx = threadIdx.x + blockIdx.x * blockDim.x; tidx < total_size; tidx += gridDim.x * blockDim.x)
		{

		}
	}

	void IntegratorBdptLvc::method_sort_light_preprocess(CameraSampleBuffer* csb, RayBuffer* rb, uint32_t iteration)
	{
		cout << "Preprocess Light Trace step";

		DevicePointer alives = m_allocator.allocate(sizeof(uint8_t) * m_params.m_num_prep_paths);
		DevicePointer rayids1 = m_allocator.allocate(sizeof(GIndexType) * m_params.m_num_prep_paths);
		DevicePointer rayids2 = m_allocator.allocate(sizeof(GIndexType) * m_params.m_num_prep_paths);
		DevicePointer matids = m_allocator.allocate(sizeof(uint32_t) * m_params.m_num_prep_paths);
		uint32_t num_alives = m_params.m_num_prep_paths;
		
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

		thrust::sequence(thrust::device_ptr<GIndexType>(static_cast<GIndexType*>(rayids1.m_ptr)), thrust::device_ptr<GIndexType>(static_cast<GIndexType*>(rayids1.m_ptr)) + m_params.m_num_prep_paths, 0);
		thrust::fill(thrust::device_ptr<GIndexType>(static_cast<GIndexType*>(rayids2.m_ptr)), thrust::device_ptr<GIndexType>(static_cast<GIndexType*>(rayids2.m_ptr)) + m_params.m_num_prep_paths, 0);
		thrust::fill(thrust::device_ptr<uint8_t>(static_cast<uint8_t*>(alives.m_ptr)), thrust::device_ptr<uint8_t>(static_cast<uint8_t*>(alives.m_ptr)) + m_params.m_num_prep_paths, 1);
		thrust::fill(thrust::device_ptr<uint8_t>(static_cast<uint8_t*>(ib.m_intersected)), thrust::device_ptr<uint8_t>(static_cast<uint8_t*>(ib.m_intersected)) + m_params.m_num_prep_paths, 0);
		
		dim3 grid_size(256, 1, 1);
		dim3 block_size(256, 1, 1);

		CudaTimer t1("timer");

		std::default_random_engine rng;
		std::uniform_int_distribution<uint32_t> u01(0, UINT32_MAX);

		// Create light samples and trace the first batch.
		t1.start();
		kernel_sort_light_pass_create_light_samples<<<grid_size, block_size>>>(scene_data, mb, lrb, static_cast<float4*>(ib.m_contribution));
		t1.stop();

		while (num_alives > 0)
		{
			// Trace.
			m_tracer->trace(lrb, ib, scene_data, static_cast<uint8_t*>(alives.m_ptr), false);

			// compact into the rayids2 whichever is alive
			thrust::transform(thrust::device_ptr<uint8_t>(static_cast<uint8_t*>(ib.m_intersected)), thrust::device_ptr<uint8_t>(static_cast<uint8_t*>(ib.m_intersected)) + m_params.m_num_prep_paths,
				thrust::device_ptr<uint8_t>(static_cast<uint8_t*>(alives.m_ptr)), IsIntersected());

			auto end = thrust::copy_if(thrust::device_ptr<GIndexType>(static_cast<GIndexType*>(rayids1.m_ptr)), thrust::device_ptr<GIndexType>(static_cast<GIndexType*>(rayids1.m_ptr)) + num_alives,
									   thrust::device_ptr<uint8_t>(static_cast<uint8_t*>(alives.m_ptr)), thrust::device_ptr<GIndexType>(static_cast<GIndexType*>(rayids2.m_ptr)), IsAlive());

			//num_alives = end - thrust::device_ptr<GIndexType>(static_cast<GIndexType*>(rayids2.m_ptr));

			// Sort rayids by material ids.
			thrust::sort_by_key(thrust::device_ptr<uint32_t>(static_cast<uint32_t*>(matids.m_ptr)), thrust::device_ptr<uint32_t>(static_cast<uint32_t*>(matids.m_ptr)) + num_alives,
				thrust::device_ptr<GIndexType>(static_cast<GIndexType*>(rayids2.m_ptr)));
			
			// Compute BRDFs and next bounce.
			kernel_sort_light_pass_evaluate_material_next_bounce<<<grid_size, block_size>>>(scene_data, mb, lrb, ib, lvc, static_cast<uint8_t*>(alives.m_ptr), static_cast<GIndexType*>(rayids2.m_ptr),
				m_params.m_cam_path_rrstart, m_params.m_num_light_path_max_depth, false, num_alives, m_params.m_num_prep_paths, nullptr, 0, 0);

			// Get if any path is still alive.
			num_alives = thrust::count(thrust::device_ptr<uint8_t>(static_cast<uint8_t*>(alives.m_ptr)), thrust::device_ptr<uint8_t>(static_cast<uint8_t*>(alives.m_ptr)) + m_params.m_num_prep_paths, 1);
		}

		// Free all allocated memory.
		m_allocator.free(alives);
		m_allocator.free(rayids1);
		m_allocator.free(matids);
		m_allocator.free(rayids2);
	}

	void IntegratorBdptLvc::method_sort_light_trace(CameraSampleBuffer* csb, RayBuffer* rb, uint32_t iteration)
	{
		// Preprocess step
		cout << "Preprocess Light Trace Step" << endl;
		// Create light samples

		DevicePointer d_ranges = m_allocator.allocate(sizeof(uint2) * m_scene->get_num_materials());
		uint2* h_ranges = new uint2[sizeof(uint2) * m_scene->get_num_materials()];

		bool some_path_alive = true;

		while (some_path_alive)
		{
			// trace
			// sort samples based on material complexity

			// compute sizes of each material
			checkCuda(cudaMemcpy(h_ranges, d_ranges.m_ptr, sizeof(uint2) * m_scene->get_num_materials(), cudaMemcpyHostToDevice));

			CudaTimer t1("material eval timer");

			// launch kernels - to evaluate and compute next bounce

		}
	}

	void IntegratorBdptLvc::method_sort_camera_trace(CameraSampleBuffer* csb, RayBuffer* rb, uint32_t iteration)
	{
		// Input camera samples are provided
		// #1. Trace
		// #2. Sort intersections based on material id
		// #3. Choose M vertices for both
		// #4. Evaluate f*g
		// #5. Select top N for shadow rays
		// #6. Compute results
		// Redo till all rays die.

		IntersectionBufferClass ibc(m_allocator);
		ibc.allocate(rb->m_size);

		IntersectionBuffer isect = ibc.get_buffer();
		SceneBuffer sb = m_scene->gpu_get_buffer();
		MaterialBuffer mb = m_scene->gpu_get_material_buffer();
		const uint32_t* material_ids = m_scene->gpu_get_tri_material_ids();

		m_tracer->trace(*rb, isect, sb, NULL, false);

		dim3 grid_size(256, 1, 1);
		dim3 block_size(256, 1, 1);

		CudaTimer t1("update kernel timer");
		t1.start();
		t1.stop();
	}
}