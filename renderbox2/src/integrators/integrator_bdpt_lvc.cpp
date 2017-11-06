
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
#include <integrators/integrator_bdpt_lvc.h>

// Cuda specific headers.

// Standard c++ headers.
#include <omp.h>
#include <iostream>

using namespace std;

namespace renderbox2
{
	
	//
	// Bidirectional Path LVC Integrator.
	//

	IntegratorBdptLvc::IntegratorBdptLvc(const BidirectionalPathIntegratorLvcParams& params, MemoryAllocator& allocator)
		: Integrator(allocator)
		, m_lvc_class(allocator)
		, m_params(params)
	{
		m_num_filled_vertices = 0;
		m_num_lvc_vertices = 0;
	}

	void IntegratorBdptLvc::render(Scene* scene, CudaTracer* tracer, void** pdata, const uint32_t n_data)
	{
		m_scene = scene;
		m_tracer = tracer;

		CameraSampleBuffer* csb = static_cast<CameraSampleBuffer*>(pdata[0]);
		RayBuffer* rb = static_cast<RayBuffer*>(pdata[1]);
		uint32_t* iteration = static_cast<uint32_t*>(pdata[2]);

		assert(csb != nullptr);
		assert(rb != nullptr);

		compute(csb, rb, *iteration);
	}

	void IntegratorBdptLvc::compute(CameraSampleBuffer* csb, RayBuffer* rb, uint32_t iteration)
	{
		switch (m_params.m_mode)
		{
		case BidirectionalPathIntegratorMode::MODE_BDPT_LVC_SK:
		{
			double start = omp_get_wtime();
			method_sk_light_trace(csb, rb, iteration);
			double end = omp_get_wtime();
			cout << "Light Trace : " << (end - start) * 1000 << endl;
			method_sk_camera_trace(csb, rb, iteration);
		}
		break;
		case BidirectionalPathIntegratorMode::MODE_BDPT_LVC_MK:
		{
			double start = omp_get_wtime();
			method_mk_light_preprocess(csb, rb, iteration);
			method_mk_light_trace(csb, rb, iteration);
			double end = omp_get_wtime();
			cout << "Light Trace : " << (end - start) * 1000 << endl;

			method_mk_camera_trace(csb, rb, iteration);
		}
		break;
		case BidirectionalPathIntegratorMode::MODE_BDPT_SORTED_LVC:
		{
			method_sort_light_trace(csb, rb, iteration);
			method_sort_camera_trace(csb, rb, iteration);
		}
		break;
		}
	}
};
