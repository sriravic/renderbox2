
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
#include <core/intersection.h>
#include <integrators/integrator_raycast.h>
#include <util/cudatimer.h>

// Cuda specific headers.

// Standard c++ headers.

namespace renderbox2
{

	//
	// Ray Cast Integrator update kernel.
	//

	__global__ void kernel_ray_cast_update(
		CameraSampleBuffer	sample_buffer,
		IntersectionBuffer	isect_buffer,
		MaterialBuffer		material_buffer,
		const uint32_t*		material_ids,
		RayCastShade		shade_mode
		)
	{
		for (GIndexType tidx = threadIdx.x + blockIdx.x * blockDim.x; tidx < sample_buffer.m_size; tidx += gridDim.x * blockDim.x)
		{
			float4 color = make_float4(0.0f);

			float4 colors[15] =	{ make_float4(1.0f, 0.0f, 0.0f, 0.0f),					// red
								  make_float4(0.0f, 1.0f, 0.0f, 0.0f),					// green
								  make_float4(0.0f, 0.0f, 1.0f, 0.0f),					// blue
								  make_float4(0.0f, 1.0f, 1.0f, 0.0f),					// yellow
								  make_float4(1.0f, 0.0f, 1.0f, 0.0f),					// magenta
								  make_float4(1.0f, 1.0f, 0.0f, 0.0f),					// cyan
								  make_float4(0.1f, 0.1f, 0.1f, 0.0f),					// dull gray
								  make_float4(0.5f, 0.5f, 0.5f, 0.0f),					// medium gray
								  make_float4(1.0f, 1.0f, 1.0f, 0.0f),					// full white
								  make_float4(0.0f, 0.25f, 0.5f, 0.0f),					// dark greenish
								  make_float4(1.0f, 0.5f, 0.0f, 0.0f),					// orangeish
								  make_float4(0.5f, 0.25f, 0.0f, 0.0f),					// brown
								  make_float4(0.2f, 0.9f, 1.0f, 0.0f),					
							      make_float4(0.7f, 0.3f, 0.1f, 0.0f),
								  make_float4(0.34f, 0.89f, 0.45f, 0.0f)};

			if (shade_mode == RayCastShade::SHADE_PRIMITIVE_ID)
			{
				if (isect_buffer.m_intersected[tidx] == 1)
					color = colors[isect_buffer.m_primitive_id[tidx] % 15];
				else
					color = make_float4(0.0f);
			}
			else if (shade_mode == RayCastShade::SHADE_MATERIAL_ID)
			{
				if (isect_buffer.m_intersected[tidx] == 1)
				{
					const uint32_t primitive_id = isect_buffer.m_primitive_id[tidx];
					const uint32_t mid = material_ids[primitive_id];
					const Material& material = material_buffer.m_materials[mid];
					const BsdfType type = material.layer_bsdf_type[0];								// assuming zeroth layer color only.
					const uint32_t bsdf_param_index = material.layer_bsdf_id[0];
					const uint32_t emitter = material.m_emitter;
					
					if (type == BSDF_LAMBERTIAN && emitter != 1)
					{
						const LambertianBsdfParams& params = material_buffer.m_lambertian_bsdfs[bsdf_param_index];
						color = params.color;
					}
					else if (type == BSDF_LAMBERTIAN && emitter == 1)
					{
						const DiffuseEmitterParams& params = material_buffer.m_diffuse_emitter_bsdfs[bsdf_param_index];
						color = params.color;
					}
					else if (type == BSDF_GLASS)
					{
						const GlassBsdfParams& params = material_buffer.m_glass_bsdfs[bsdf_param_index];
						color = params.color;
					}
					else if (type == BSDF_MIRROR)
					{
						const MirrorBsdfParams& params = material_buffer.m_mirror_bsdfs[bsdf_param_index];
						color = params.color;
					}
					else if (type == BSDF_BLINN_MICROFACET)
					{
						const MicrofacetBsdfParams& params = material_buffer.m_microfacet_bsdfs[bsdf_param_index];
						color = params.R;
					}
				}
			}
			else if (shade_mode == RayCastShade::SHADE_NORMALS)
			{
				if (isect_buffer.m_intersected[tidx] == 1)
					color = make_float4(isect_buffer.m_shading_normal[0][tidx], isect_buffer.m_shading_normal[1][tidx], isect_buffer.m_shading_normal[2][tidx], 0.0f);
				else
					color = make_float4(0.0f);
			}
			else if (shade_mode == RayCastShade::SHADE_UVS)
			{

			}
			else if (shade_mode == RayCastShade::SHADE_UNIFORM)
			{
				color = isect_buffer.m_intersected[tidx] == 1 ? make_float4(1.0f) : make_float4(0.0f);
			}

			sample_buffer.m_contribution[tidx] = color;
		}
	}


	//
	// Ray Cast Integrator's compute method calls the tracer kernels and computes all the results.
	// NOTE: Number of elements in the camera sample buffer and ray buffer are the same. One to One correspondence.
	//

	void IntegratorRayCast::compute(CameraSampleBuffer* csb, RayBuffer* rb)
	{

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
		kernel_ray_cast_update<<<grid_size, block_size>>>(*csb, isect, mb, material_ids, m_params.m_shade);
		t1.stop();
	}
}
