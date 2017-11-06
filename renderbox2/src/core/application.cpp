
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
#include <accelerators/sbvh/cudabvh.h>
#include <accelerators/sbvh/cudatracer.h>
#include <core/application.h>
#include <core/defs.h>
#include <core/camera.h>
#include <core/film.h>
#include <core/filter.h>
#include <core/renderer.h>
#include <core/scene.h>
#include <io/customfilereader.h>
#include <memory/memoryallocator.h>
#include <renderers/samplerrenderer.h>
#include <util/configuration.h>

// Cuda specific headers.

// Standard c++ headers.

namespace renderbox2
{

	//
	// Application class definitions.
	//

	Application::Application()
	{

		cout << "Application starting up" << endl;

		m_gui = false;

		m_bvh = nullptr;
		m_camera = nullptr;
		m_output_film = nullptr;
		m_configuration = nullptr;
		m_reader = nullptr;
		m_scene = nullptr;
		m_allocator = nullptr;
		m_renderer = nullptr;
		m_tracer = nullptr;
		m_filter = nullptr;
	}

	Application::~Application()
	{
		SAFE_RELEASE(m_bvh);
		SAFE_RELEASE(m_camera);
		SAFE_RELEASE(m_output_film);
		SAFE_RELEASE(m_configuration);
		SAFE_RELEASE(m_reader);
		SAFE_RELEASE(m_scene);
		SAFE_RELEASE(m_tracer);
		SAFE_RELEASE(m_renderer);
		SAFE_RELEASE(m_filter);
		SAFE_RELEASE(m_allocator);
	}

	bool Application::init(const string& config_file)
	{
		cout << "Application initializing.." << endl;

		m_configuration = new Configuration();
		m_allocator = new MemoryAllocator();

		if (!m_configuration->load_config_file(config_file))
		{
			cout << "Error loading configuration file. Application Exiting" << endl;
			return false;
		}

		if (m_configuration->get_application_flags().m_gui)
		{
			m_gui = true;
		}

		return true;
	}

	bool Application::prepare()
	{
		cout << "Preparing Scene Data.\nThis might take some time.\nHave a cookie" << endl;

		// First load the scene.
		m_scene = new Scene(*m_allocator);

		m_reader = new CustomFileReader(m_configuration->get_scene_file_path(),
			m_configuration->get_scene_file_name(),
			*m_scene);

		if (!m_reader->all_ok())
		{
			cerr << "Not all is okay with the custom file reader" << endl;
			return false;
		}
		
		// Next build the bvh.
		bool to_build;
		string bvh_file = m_configuration->get_bvh_file(to_build);

		BVHLayout required_layout;

#ifdef USE_KERNEL_FERMI
		required_layout = BVHLayout::BVHLayout_Compact;
#elif defined USE_KERNEL_KEPLER
		required_layout = BVHLayout::BVHLayout_Compact2;
#endif
		if (to_build)
		{
			cout << "Constructing BVH over scene." << endl;
			Platform platform;
			BVH::BuildParams params;
			BVH::Stats stats;		

			params.m_enable_prints = false;
			params.m_stats = &stats;
			BVH sbvh(m_scene, platform, params);			

			m_bvh = new CudaBvh(sbvh, required_layout);
		}
		else
		{
			cout << "Loading Prebuilt BVH File :" << bvh_file << endl;
			bool is_loaded_properly;

			m_bvh = new CudaBvh(bvh_file, is_loaded_properly);

			if (!is_loaded_properly)
			{
				cerr << "Error loading the BVH file" << endl;
				return false;
			}

			if (is_loaded_properly)
			{
				if (required_layout != m_bvh->get_layout())
				{
					cerr << "Loaded SBVH has a different layout than the one required by compiled kernels";
					return false;
				}
			}
		}

		// Get camera, filter and film.
		uint32_t width, height;
		m_configuration->get_output_dims(width, height);

		FilterParams filter_params = m_configuration->get_filter_params();

		assert(filter_params.m_type != FilterType::FILTER_UNKNOWN);

		switch (filter_params.m_type)
		{
		case FilterType::FILTER_BOX:
			m_filter = new FilterBox(filter_params.xwidth, filter_params.ywidth);
			break;
		case FilterType::FILTER_GAUSSIAN:
			m_filter = new FilterGaussian(filter_params.xwidth, filter_params.ywidth, filter_params.alpha);
			break;
		case FilterType::FILTER_TRIANGLE:
			m_filter = new FilterTriangle(filter_params.xwidth, filter_params.ywidth);
			break;
		}

		m_output_film = new Film(width, height, m_filter, *m_allocator);

		m_reader->get_camera(&m_camera, width, height);

		// Set all the scene variables.
		m_scene->set_camera(m_camera);
		m_scene->set_output_film(m_output_film);
		m_scene->set_bvh(m_bvh);
		m_scene->gpu_prepare();
		
		// Report stats on scene data.
		cout << "Total GPU memory usage : " << get_mb(m_allocator->total_size()) << " mb"<< endl;

		// create the ray tracer.
		m_tracer = new CudaTracer(*m_bvh, *m_allocator);

		// Create the renderer.
		RendererType m_type = m_configuration->get_renderer_type();
		
		if (m_type == RendererType::RENDERER_SAMPLER)
		{
			SamplerRendererParams m_params = m_configuration->get_sampler_renderer_params();
			m_renderer = new SamplerRenderer(m_params, *m_allocator);
		}
		else
		{
			cout << "Unsupported Renderer type" << endl;
			return false;
		}

		return true;
	}

	bool Application::run()
	{
		cout << "Application Running" << endl;

		if (!m_configuration->get_application_flags().m_gui)
		{
			// Render the scene.
			m_renderer->render(m_scene, m_tracer, m_configuration);

			// Write the output.
			m_output_film->write_image(m_configuration->get_output_filename());
		}
		
		return true;
	}

	bool Application::shutdown()
	{
		cout << "Application shutting down" << endl;
		return true;
	}
}
