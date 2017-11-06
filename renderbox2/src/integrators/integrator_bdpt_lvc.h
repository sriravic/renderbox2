
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

#ifndef INTEGRATOR_BDPT_LVC_H
#define INTEGRATOR_BDPT_LVC_H

// Application specific headers.
#include <core/integrator.h>
#include <core/params.h>

// Cuda specific headers.

// Standard c++ headers.

namespace renderbox2
{

	//
	// Bidirectional Path Tracing Integrator with LVC method.
	//

	class IntegratorBdptLvc : public Integrator
	{
	public:
		
		IntegratorBdptLvc(const BidirectionalPathIntegratorLvcParams& params, MemoryAllocator& allocator);

		void render(Scene* scene, CudaTracer* tracer, void** data, const uint32_t n_data);

	private:

		void compute(CameraSampleBuffer* csb, RayBuffer* rb, uint32_t iteration);

		//  Single monolithic kernel for the both the light and camera trace.
		void method_sk_light_trace(CameraSampleBuffer* csb, RayBuffer* rb, uint32_t iteration);
		void method_sk_camera_trace(CameraSampleBuffer* csb, RayBuffer* rb, uint32_t iteration);

		// Multiple kernels - generate/eval_trace/shadow (from paper)
		void method_mk_light_preprocess(CameraSampleBuffer* csb, RayBuffer* rb, uint32_t iteration);
		void method_mk_light_trace(CameraSampleBuffer* csb, RayBuffer* rb, uint32_t iteration);
		void method_mk_camera_trace(CameraSampleBuffer* csb, RayBuffer* rb, uint32_t iteration);

		// Our method.
		void method_sort_light_preprocess(CameraSampleBuffer* csb, RayBuffer* rb, uint32_t iteration);
		void method_sort_light_trace(CameraSampleBuffer* csb, RayBuffer* rb, uint32_t iteration);
		void method_sort_camera_trace(CameraSampleBuffer* csb, RayBuffer* rb, uint32_t iteration);

		BidirectionalPathIntegratorLvcParams m_params;

		IntersectionBufferClass m_lvc_class;
		uint32_t	m_num_filled_vertices;
		uint32_t	m_num_lvc_vertices;
	};

};				// !namespace renderbox2

#endif			// !INTEGRATOR_BDPT_LVC_H
