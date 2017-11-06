
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
#include <core/scene.h>
#include <integrators/integrator_ao.h>

// Cuda specific headers.

// Standard c++ headers.
#include <cassert>

namespace renderbox2
{

	//
	// Ambient Occlusion Integrator implementation.
	//

	IntegratorAO::IntegratorAO(const AmbientOcclusionIntegratorParams& params, MemoryAllocator& allocator)
		: Integrator(allocator)
		, m_params(params)
	{

	}

	void IntegratorAO::render(Scene* scene, CudaTracer* tracer, void** pdata, const uint32_t n_data)
	{
		m_scene = scene;
		m_tracer = tracer;

		CameraSampleBuffer* csb = static_cast<CameraSampleBuffer*>(pdata[0]);
		RayBuffer* rb = static_cast<RayBuffer*>(pdata[1]);

		assert(csb != nullptr);
		assert(rb != nullptr);

		compute(csb, rb);
	}
};
