
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

#ifndef RENDERER_H
#define RENDERER_H

// Application specific headers.
#include <memory/memoryallocator.h>
#include <util/configuration.h>

// Cuda specific headers.

// Standard c++ headers.

// Forward declarations.
namespace renderbox2 { class Scene; }
namespace renderbox2 { class Configuration; }
namespace renderbox2 { class CudaTracer; }

namespace renderbox2
{

	//
	// Renderer is the abstract base class for all renderers available within the system.
	//

	class Renderer
	{
	public:

		Renderer(MemoryAllocator& allocator) : m_allocator(allocator)
		{
			// Empty constructor.
		}

		virtual void render(Scene* scene, CudaTracer* tracer, const Configuration* config) = 0;

	protected:
		
		Scene*					m_scene;
		CudaTracer*				m_tracer;
		MemoryAllocator&		m_allocator;
	};

};				// !namespace renderbox2

#endif			// !RENDERER_H
