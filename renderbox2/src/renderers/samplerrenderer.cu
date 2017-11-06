
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
#include <core/camera.h>
#include <core/film.h>
#include <core/filter.h>
#include <core/integrator.h>
#include <core/scene.h>
#include <core/util.h>
#include <renderers/samplerrenderer.h>
#include <util/cudatimer.h>

// Cuda specific headers.
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_occupancy.h>

#define NDEBUG 1

#include <thrust/random.h>

// Standard c++ headers.

namespace renderbox2
{

	//
	// Global kernels used by the sampler renderer.
	//
	
	__global__ void kernel_generate_perspective_camera_samples(
		CameraSampleBuffer sample_buffer,
		RayBuffer		   ray_buffer,
		PerspectiveCamera  camera,
		uint32_t		   spp,
		thrust::default_random_engine* generators,
		uint32_t		   iteration,
		uint4			   window)
	{
		for (GIndexType tidx = threadIdx.x + blockIdx.x * blockDim.x; tidx < camera.get_total_pixels(); tidx += gridDim.x * blockDim.x)
		{
			thrust::default_random_engine rng;
			thrust::uniform_real_distribution<float> u01(0, 1);
			
			if (iteration == 0)
			{
				uint32_t seed = simplehash(tidx + iteration);
				rng = thrust::default_random_engine(seed);
			}
			else
			{
				rng = generators[tidx];
			}
			
			uint32_t pixel_x = tidx % camera.get_film_width();
			uint32_t pixel_y = tidx / camera.get_film_height();
			
			for (uint32_t sample = 0; sample < spp; sample++)
			{
				int32_t idx = tidx * spp + sample;

				float randx = u01(rng);
				float randy = u01(rng);

				float2 pixel = make_float2(static_cast<float>(pixel_x)+randx, static_cast<float>(pixel_y)+randy);

				sample_buffer.m_alives[idx] = 1;
				sample_buffer.m_continue_probability[idx] = 1.0f;
				sample_buffer.m_contribution[idx] = make_float4(0.0f);
				sample_buffer.m_ids[idx] = idx;
				sample_buffer.m_pixel_coords[idx] = pixel;
				sample_buffer.m_throughput[idx] = make_float4(1.0f);

				ray_buffer.m_data[idx] = camera.generate_ray(pixel.x, pixel.y);
			}

			generators[tidx] = rng;
		}
	}


	//
	// Update film kernels.
	//

	__global__ void kernel_update_film(
		CameraSampleBuffer	sample_buffer,
		Pixel*				pixels,
		uint2				film_size,
		float*				filter_table_coeffs,
		float2				filter_width,
		float2				filter_invwidth,
		uint2				pixel_start
		)
	{
		for (GIndexType tidx = threadIdx.x + blockIdx.x * blockDim.x; tidx < sample_buffer.m_size; tidx += gridDim.x + blockDim.x)
		{
			// NOTE: for now we assume only one sample per pixel.
			float4 contribution = sample_buffer.m_contribution[tidx];
			
			float2 pixel = sample_buffer.m_pixel_coords[tidx];

			float dimageX = pixel.x - 0.5f;
			float dimageY = pixel.y - 0.5f;

			// compute window of pixels that this sample affects
			int x0 = static_cast<int>(ceilf(dimageX - filter_width.x));
			int x1 = static_cast<int>(floorf(dimageX + filter_width.x));
			int y0 = static_cast<int>(ceilf(dimageY - filter_width.y));
			int y1 = static_cast<int>(floorf(dimageY + filter_width.y));

			x0 = fmaxf(x0, pixel_start.x);
			x1 = fminf(x1, pixel_start.x + film_size.x - 1);
			y0 = fmaxf(y0, pixel_start.y);
			y1 = fminf(y1, pixel_start.y + film_size.y - 1);

			// Precompute indices.
			// NOTE: This is a DANGEROUS ASSUMPTION.!
			int ifx[16];
			int ify[16];

			for (int x = x0; x <= x1; x++)
			{
				float fx = fabsf((x - dimageX) * filter_invwidth.x * FILTER_TABLE_SIZE);
				ifx[x - x0] = min(static_cast<int>(floorf(fx)), FILTER_TABLE_SIZE - 1);
			}

			for (int y = y0; y <= y1; y++)
			{
				float fy = fabsf((y - dimageY) * filter_invwidth.y * FILTER_TABLE_SIZE);
				ify[y - y0] = min(static_cast<int>(floorf(fy)), FILTER_TABLE_SIZE - 1);
			}


			for (int y = y0; y <= y1; y++)
			{
				for (int x = x0; x <= x1; x++)
				{
					int offset = ify[y - y0] * FILTER_TABLE_SIZE + ifx[x - x0];
					float filter_weight = filter_table_coeffs[offset];

					// Update in atomic fashion only.
					Pixel& pixel = pixels[(y - pixel_start.y) * film_size.x + (x - pixel_start.x)];
					
					atomicAdd(&(pixel.Lrgb[0]), filter_weight * contribution.x);
					atomicAdd(&(pixel.Lrgb[1]), filter_weight * contribution.y);
					atomicAdd(&(pixel.Lrgb[2]), filter_weight * contribution.z);
					atomicAdd(&pixel.weightsum, filter_weight);
				}
			}
		}
	}


	//
	// Integrator private methods.
	//

	void SamplerRenderer::compute(uint32_t iteration)
	{
		// NOTE: For now we are assuming samples can be allocated in memory as such without needing bucketing.
		
		// Create the samples.
		const uint2 film_dimensions = make_uint2(m_scene->get_camera()->get_film_width(), m_scene->get_camera()->get_film_height());
		const uint32_t num_samples = film_dimensions.x * film_dimensions.y * m_params.m_spp;

		if (iteration == 0)
		{
			alloc_rng_states(film_dimensions.x * film_dimensions.y);
		}

		CameraSampleBufferClass csamples(m_allocator);
		csamples.allocate(num_samples);
		
		CameraSampleBuffer csb = csamples.get_buffer();

		// Create camera rays.
		RayBufferClass primary_rays(m_allocator);
		primary_rays.allocate(num_samples);

		// call a primary ray generation kernel.
		RayBuffer rb = primary_rays.get_buffer();
		
		dim3 grid_size(256, 1, 1);
		dim3 block_size(256, 1, 1);
		
		CudaTimer t1("primary rays timer");
		
		thrust::default_random_engine* rng = static_cast<thrust::default_random_engine*>(m_rng_generators.m_ptr);

		t1.start();
		kernel_generate_perspective_camera_samples<<<grid_size, block_size>>>(csb, rb, *(m_scene->get_camera()), m_params.m_spp, rng, iteration, make_uint4(0));
		t1.stop();

		// Call trace functionality.
		void* data[] = { &csb, &rb, &iteration };
		
		m_integrator->render(m_scene, m_tracer, data, 2);
		
		// Collect results and update.
		Film* film = m_scene->get_output_film();
		Filter* filter = film->get_filter();

		CudaTimer t2("update timer");
		t2.start();
		kernel_update_film <<<grid_size, block_size >> >(csb, static_cast<Pixel*>(film->get_pixels().m_ptr), make_uint2(film->get_width(), film->get_height()),
			static_cast<float*>(film->get_filter_table().m_ptr), make_float2(filter->m_xwidth, filter->m_ywidth), make_float2(filter->m_inv_xwidth, filter->m_inv_ywidth), make_uint2(0, 0));

		t2.stop();
		float ms = t2.get_ms();
		std::cout << "Filtering : " << ms << std::endl;
	}

	void SamplerRenderer::alloc_rng_states(uint32_t samples)
	{
		m_rng_generators = m_allocator.allocate(sizeof(thrust::default_random_engine) * samples);
	}
}
