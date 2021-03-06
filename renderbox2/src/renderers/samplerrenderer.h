
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

#ifndef SAMPLER_RENDERER_H
#define SAMPLER_RENDERER_H

// Application specific headers.
#include <core/defs.h>
#include <core/params.h>
#include <core/renderer.h>

// Cuda specific headers.

// Standard c++ headers.
#include <cstdint>

// Forward declarations.
namespace renderbox2 { class Integrator; }

namespace renderbox2
{

	class SamplerRenderer : public Renderer
	{
	public:

		SamplerRenderer(const SamplerRendererParams& params, MemoryAllocator& allocator);

		~SamplerRenderer();

		void render(Scene* scene, CudaTracer* tracer, const Configuration* config);

	private:

		// Integrator methods supported by the application have separate logic and hence in their own methods.
		// Keeps the code clean and manageable.
		void compute(uint32_t iteration);
		void alloc_rng_states(uint32_t num_samples);

		DevicePointer				m_rng_generators;

		Integrator*					m_integrator;
		SamplerRendererParams		m_params;
	};

};				// !namespace renderbox2

#endif			// !SAMPLER_RENDERER_H
