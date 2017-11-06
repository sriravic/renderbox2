
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

#ifndef APPLICATION_H
#define APPLICATION_H

// Application specific headers.

// Cuda specific headers.

// Standard c++ headers.
#include <string>

// Forward declarations
class CudaBvh;

namespace renderbox2 { class Configuration; }
namespace renderbox2 { class CudaTracer; }
namespace renderbox2 { class CustomFileReader; }
namespace renderbox2 { class Film; }
namespace renderbox2 { class MemoryAllocator; }
namespace renderbox2 { class PerspectiveCamera; }
namespace renderbox2 { class Renderer; }
namespace renderbox2 { class Scene; }
namespace renderbox2 { struct Filter; }

namespace renderbox2
{
	class Application
	{
	public:
		
		Application();
		
		~Application();

		//
		// Application core methods.
		//

		bool init(const std::string& config_file = "..//src//config.xml");

		bool prepare();

		bool run();

		bool shutdown();

		bool gui() const { return m_gui; }
	
	private:

		//
		// Private data members.
		//

		PerspectiveCamera*	m_camera;
		Configuration*		m_configuration;
		CustomFileReader*   m_reader;
		CudaBvh*			m_bvh;
		CudaTracer*         m_tracer;
		Film*               m_output_film;
		Filter*				m_filter;
		Scene*				m_scene;
		MemoryAllocator*    m_allocator;
		Renderer*			m_renderer;

		bool				m_gui;
	};

}			// !namespace renderbox2

#endif		// !APPLICATION_H
