
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
#include <core/montecarlo.h>
#include <core/util.h>
#include <integrators/integrator_ao.h>
#include <util/cudatimer.h>

// Cuda specific headers.
#include <thrust/random.h>

// Standard c++ headers.

namespace renderbox2
{

	//
	// Ambient Occlusion Integrator kernels
	//

	__global__ void kernel_ambient_occlusion(
		IntersectionBuffer	isects,
		CameraSampleBuffer	csamples,
		BvhStruct			bvh,
		SceneBuffer			scene_data,
		float				ao_radius,
		uint32_t			num_ao_samples
		)
	{
		for (GIndexType tidx = threadIdx.x + blockIdx.x * blockDim.x; tidx < csamples.m_size; tidx += gridDim.x * blockDim.x)
		{
			// Get the current intersection point and normal.
			float3 position = make_float3(isects.m_position[0][tidx], isects.m_position[1][tidx], isects.m_position[2][tidx]);
			float3 normal = make_float3(isects.m_shading_normal[0][tidx], isects.m_shading_normal[1][tidx], isects.m_shading_normal[2][tidx]);
			float epsilon = isects.m_epsilon[tidx];
			float4 m_contribution = make_float4(0.0f);
			bool intersected = (isects.m_intersected[tidx] == 1);

			if (intersected)
			{
				int32_t seed = simplehash(tidx);
				thrust::uniform_real_distribution<float> u01(0.0f, 1.0f);
				thrust::default_random_engine rng(seed);


				for (uint32_t i = 0; i < num_ao_samples; i++)
				{
					// Sample a ray in the hemisphere.
					float3 direction = cosine_sample_hemisphere_normal(u01(rng), u01(rng), normal);
					Ray ao_ray(position, direction, epsilon, ao_radius);

					// Trace the ray with max radius and if so accumulate.
					bool occluded = trace_ray(ao_ray, true, nullptr, bvh.m_nodes, bvh.m_tris, bvh.m_tri_idx, scene_data);
					if (!occluded)
					{
						m_contribution += make_float4(1.0f, 1.0f, 1.0f, 0.0f);
					}
				}
			}

			// Finally divide to give value.
			csamples.m_contribution[tidx] = m_contribution / num_ao_samples;
		}
	}


	//
	// Ambient Occlusion Integrators compute method shoots a bunch of rays and computes occlusion for each primary ray.
	//

	void IntegratorAO::compute(CameraSampleBuffer* csb, RayBuffer* rb)
	{
		IntersectionBufferClass ibc(m_allocator);
		ibc.allocate(csb->m_size);

		IntersectionBuffer ib = ibc.get_buffer();
		SceneBuffer scene_data = m_scene->gpu_get_buffer();
		BvhStruct bvh = m_tracer->get_bvh();

		m_tracer->trace(*rb, ib, scene_data, nullptr, false);

		dim3 grid_size(256, 1, 1);
		dim3 block_size(256, 1, 1);

		CudaTimer t1("ao timer");
		t1.start();
		kernel_ambient_occlusion<<<grid_size, block_size>>>(ib, *csb, bvh, scene_data, m_params.m_radius, m_params.m_samples);
		t1.stop();
	}
}
