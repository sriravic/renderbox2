
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
#include <core/camera.h>
#include <core/film.h>
#include <core/integrator.h>
#include <core/scene.h>
#include <integrators/integrator_ao.h>
#include <integrators/integrator_bdpt_lvc.h>
#include <integrators/integrator_path.h>
#include <integrators/integrator_raycast.h>
#include <renderers/samplerrenderer.h>
#include <util/configuration.h>

// Cuda specific headers.

// Standard c++ headers.
#include <iostream>
#include <omp.h>

namespace renderbox2
{

	//
	// SamplerRenderer class implementation.
	// 

	SamplerRenderer::SamplerRenderer(const SamplerRendererParams& params, MemoryAllocator& allocator) : Renderer(allocator)
	{
		m_integrator = nullptr;
		m_params = params;
	}

	SamplerRenderer::~SamplerRenderer()
	{
		// Free the integrator.
		SAFE_RELEASE(m_integrator);
	}

	void SamplerRenderer::render(Scene* scene, CudaTracer* tracer, const Configuration* configuration)
	{

		// Create the integrator.
		m_scene = scene;
		m_tracer = tracer;

		IntegratorType integrator_type = configuration->get_integrator_type();

		switch (integrator_type)
		{
			case IntegratorType::INTEGRATOR_PATH:
			{
				PathIntegratorParams params = configuration->get_path_integrator_params();
				m_integrator = new IntegratorPath(params, m_allocator);
			}
			break;
			case IntegratorType::INTEGRATOR_BDPT_LVC:
			{
				BidirectionalPathIntegratorLvcParams params = configuration->get_bdpt_lvc_integrator_params();
				m_integrator = new IntegratorBdptLvc(params, m_allocator);
			}
			break;
			case IntegratorType::INTEGRATOR_AO:
			{
				AmbientOcclusionIntegratorParams params = configuration->get_ao_integrator_params();
				m_integrator = new IntegratorAO(params, m_allocator);
			}
			break;
			case IntegratorType::INTEGRATOR_RAYCAST:
			{
				RayCastIntegratorParams params = configuration->get_raycast_integrator_params();
				m_integrator = new IntegratorRayCast(params, m_allocator);
			}
			break;
		};

		// Determine if we can go for a fully interactive mode or go for tiled rendering.
		if (m_params.m_mode == SamplerRendererMode::MODE_FINAL)
		{
			for (uint32_t run = 0; run < m_params.m_iterations; run++)
			{
				compute(run);
			}
		}
		else if (m_params.m_mode == SamplerRendererMode::MODE_PROGRESSIVE)
		{
			std::cout << "Mode not implemented yet";
		}		
	}
}
