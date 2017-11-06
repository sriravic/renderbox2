
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

#ifndef CUSTOM_FILE_READER_H
#define CUSTOM_FILE_READER_H

// Application specific headers.
#include <core/scene.h>

// Cuda specific headers.
#include <vector_types.h>

// Standard c++ headers.
#include <string>

// Forward declarations.
namespace renderbox2 { class PerspectiveCamera; }

namespace renderbox2
{

	//
	// Custom File Reader class - reads the custom xml file format for the application.
	//

	class CustomFileReader
	{
	public:

		CustomFileReader()
		{
			// Empty constructor.
		}

		CustomFileReader(const std::string& scene_file_path, const std::string& scene_file_name, Scene& scene) : m_scene_file_path(scene_file_path), m_scene_file_name(scene_file_name)
		{
			m_all_ok = process(scene);
		}

		bool all_ok() const { return m_all_ok; }

		bool get_camera(PerspectiveCamera** camera, const uint32_t image_width, const uint32_t image_height) const;

	private:

		bool process(Scene& scene);

		// Utility methods.

		// Gets a float3 structure from a {x, y, z} tuple in the scene file.
		float3 get_float3(const std::string& str) const;

		std::string m_scene_file_path;
		std::string m_scene_file_name;

		bool		m_all_ok;
	};

}			// !namespace renderbox2

#endif		// !CUSTOM_FILE_READER
