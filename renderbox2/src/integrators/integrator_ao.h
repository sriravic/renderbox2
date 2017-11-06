
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

#ifndef INTEGRATOR_AO_H
#define INTEGRATOR_AO_H

// Application specific headers.
#include <core/integrator.h>
#include <core/params.h>

// Cuda specific headers.

// Standard c++ headers.

namespace renderbox2
{

	//
	// Ambient Occlusion Integrator.
	//

	class IntegratorAO : public Integrator
	{
	public:

		IntegratorAO(const AmbientOcclusionIntegratorParams& params, MemoryAllocator& allocator);

		void render(Scene* scene, CudaTracer* tracer, void** data, const uint32_t n_data);

	private:

		void compute(CameraSampleBuffer* csb, RayBuffer* rb);

		AmbientOcclusionIntegratorParams m_params;
	};

};				// !namespace renderbox2

#endif			// !INTEGRATOR_AO_H
