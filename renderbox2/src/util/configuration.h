
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

#ifndef CONFIGURATION_H
#define CONFIGURATION_H

// Application specific headers.
#include <core/params.h>
#include <3rdparty/tinyxml2/tinyxml2.h>

// Cuda specific headers.

// Standard c++ headers.
#include <cstdint>
#include <fstream>
#include <iostream>

using namespace tinyxml2;

namespace renderbox2
{

	//
	// Application flags.
	//

	struct ApplicationFlags
	{
		bool		m_render;			// should we render an output?
		bool        m_gui;				// if the application should start the gui.
	};


	//
	// Supported Renderer Types.
	//

	enum class RendererType
	{
		RENDERER_SAMPLER,
		RENDERER_UNKNOWN
	};

	
	//
	// Supported Integrator Types.
	//

	enum class IntegratorType
	{
		INTEGRATOR_PATH,
		INTEGRATOR_BDPT_LVC,
		INTEGRATOR_RAYCAST,
		INTEGRATOR_AO,
		INTEGRATOR_UNKNOWN
	};


	//
	// Configuration determines the overall application.
	//

	class Configuration
	{
	public:

		bool load_config_file(const std::string& file);


		//
		// Get methods for various values regards to application data.
		//

		// Scene files are stored in a particular directory.
		std::string	get_scene_file_path() const;
		
		// Get a scene file from the scene file path.
		std::string get_scene_file_name() const;
		
		// If a prebuilt bvh is to be used load this file.
		std::string get_bvh_file(bool& build) const;		
		
		// Get output file to write results to.
		std::string get_output_filename() const;

		// Get output file dimensions
		void get_output_dims(uint32_t& width, uint32_t& height) const;

		// Get Application flags.
		const ApplicationFlags& get_application_flags() const { return m_flags; }

		// Get Renderer Type.
		RendererType get_renderer_type() const;

		// Get SamplerRenderer Params.
		SamplerRendererParams get_sampler_renderer_params() const;

		// Get Integrator Type.
		IntegratorType get_integrator_type() const;

		// Get Various Integrator Params.
		PathIntegratorParams get_path_integrator_params() const;
		AmbientOcclusionIntegratorParams get_ao_integrator_params() const;
		RayCastIntegratorParams get_raycast_integrator_params() const;
		BidirectionalPathIntegratorLvcParams get_bdpt_lvc_integrator_params() const;

		// Get Filter Params.
		FilterParams get_filter_params() const;

	private:

		void                process_flags();

		XMLDocument			m_config_file;
		XMLDocument			m_integrator_file;
		ApplicationFlags    m_flags;

	};

}				// !namespace renderbox2

#endif			// !CONFIGURATION_H