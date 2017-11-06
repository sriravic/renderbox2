
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
#include <util/configuration.h>

// Cuda specific headers.

// Standard c++ headers.

using namespace tinyxml2;

namespace renderbox2
{
	
	//
	// Configuration class implementation.
	//

	bool Configuration::load_config_file(const std::string& file)
	{
		m_config_file.LoadFile(file.c_str());
		XMLError error = m_config_file.ErrorID();
		if (error != XML_SUCCESS) {
			printf("SEVERE : %s\n", m_config_file.GetErrorStr1());
			return false;
		}

		// Process the integrator file.
		// They are present in the /src/integrators folder
		const std::string integrator_path = std::string("..//src//integrators//") +
			std::string(m_config_file.FirstChildElement("config")->FirstChildElement("integrator")->FirstChildElement("file")->GetText());

		m_integrator_file.LoadFile(integrator_path.c_str());

		error = m_integrator_file.ErrorID();

		if (error != XML_SUCCESS)
		{
			printf("SEVERE : %s\n", m_integrator_file.GetErrorStr1());
			return false;
		}

		// Proces the application flags.
		process_flags();
		return true;
	}

	void Configuration::process_flags()
	{
		XMLElement* root = m_config_file.FirstChildElement("config");
		XMLElement* e = root->FirstChildElement("flags");

		// Render any output image?.
		XMLElement* render_flags = e->FirstChildElement("render");
		std::string rf_string(render_flags->GetText());
		if (rf_string == "yes") m_flags.m_render = true;
		else m_flags.m_render = false;

		// Start the application in gui mode?
		XMLElement* gui = e->FirstChildElement("gui");
		std::string gui_string(gui->GetText());
		if (gui_string == "yes") m_flags.m_gui = true;
		else m_flags.m_gui = false;
	}

	std::string Configuration::get_scene_file_name() const
	{
		const XMLElement* root = m_config_file.FirstChildElement("config");
		const std::string name = std::string(root->FirstChildElement("scene")->FirstChildElement("file")->GetText());
		return name;
	}
	
	std::string Configuration::get_bvh_file(bool& build) const
	{
		const XMLElement* root = m_config_file.FirstChildElement("config");		
		const XMLElement* dir = root->FirstChildElement("directories")->FirstChildElement("scenefilepath");
		const XMLElement* e = root->FirstChildElement("scene")->FirstChildElement("cudabvh");
		
		const std::string file(e->FirstChildElement("file")->GetText());
		const std::string sbuild(e->FirstChildElement("build")->GetText());
		build = (sbuild == "yes");
		return std::string(dir->GetText()) + file;
	}

	std::string Configuration::get_scene_file_path() const
	{
		const XMLElement* root = m_config_file.FirstChildElement("config");
		const std::string scene_file_path = std::string(root->FirstChildElement("directories")->FirstChildElement("scenefilepath")->GetText());
		return scene_file_path;
	}

	std::string Configuration::get_output_filename() const
	{
		const std::string output_dir(m_config_file.FirstChildElement("config")->FirstChildElement("directories")->FirstChildElement("output")->GetText());
		const std::string output_file(m_config_file.FirstChildElement("config")->FirstChildElement("output")->FirstChildElement("file")->GetText());
		return output_dir + output_file;
	}

	void Configuration::get_output_dims(uint32_t& width, uint32_t& height) const
	{
		const XMLElement* root = m_config_file.FirstChildElement("config")->FirstChildElement("output");
		width = static_cast<uint32_t>(atoi(root->FirstChildElement("width")->GetText()));
		height = static_cast<uint32_t>(atoi(root->FirstChildElement("height")->GetText()));
	}

	RendererType Configuration::get_renderer_type() const
	{
		const std::string type(m_config_file.FirstChildElement("config")->FirstChildElement("renderer")->FirstChildElement("type")->GetText());

		if (type == "samplerrenderer")
		{
			return RendererType::RENDERER_SAMPLER;
		}
		else
		{
			std::cout << "SEVERE WARNING : Unknown Renderer Type!" << std::endl;
			return RendererType::RENDERER_UNKNOWN;
		}
	}

	IntegratorType Configuration::get_integrator_type() const
	{
		const std::string type(m_integrator_file.FirstChildElement("integrator")->FirstChildElement("type")->GetText());
		if (type == "pt")
		{
			return IntegratorType::INTEGRATOR_PATH;
		}
		else if (type == "bdpt_lvc")
		{
			return IntegratorType::INTEGRATOR_BDPT_LVC;
		}
		else if (type == "ao")
		{
			return IntegratorType::INTEGRATOR_AO;
		}
		else if (type == "raycast")
		{
			return IntegratorType::INTEGRATOR_RAYCAST;
		}
		else
		{
			return IntegratorType::INTEGRATOR_UNKNOWN;
		}
	}

	SamplerRendererParams Configuration::get_sampler_renderer_params() const
	{
		SamplerRendererParams ret;
		
		const XMLElement* params = m_config_file.FirstChildElement("config")->FirstChildElement("renderer")->FirstChildElement("params");
		const std::string mode(params->FirstChildElement("mode")->GetText());
		const std::string spp(params->FirstChildElement("spp")->GetText());
		const std::string iterations(params->FirstChildElement("iterations")->GetText());
		
		if (mode == "final")
		{
			ret.m_mode = SamplerRendererMode::MODE_FINAL;
		}
		else if (mode == "progresive")
		{
			ret.m_mode = SamplerRendererMode::MODE_PROGRESSIVE;
		}

		ret.m_spp = static_cast<uint32_t>(atoi(spp.c_str()));
		ret.m_iterations = static_cast<uint32_t>(atoi(iterations.c_str()));
		return ret;
	}

	PathIntegratorParams Configuration::get_path_integrator_params() const
	{
		PathIntegratorParams ret;

		const XMLElement* params = m_integrator_file.FirstChildElement("integrator")->FirstChildElement("params");
		
		ret.m_max_depth = static_cast<uint32_t>(atoi(params->FirstChildElement("maxdepth")->GetText()));
		ret.m_rrstart = static_cast<uint32_t>(atoi(params->FirstChildElement("rrstart")->GetText()));
		ret.m_rrprob = static_cast<float>(atof(params->FirstChildElement("rrprob")->GetText()));

		const std::string use_max_depth(params->FirstChildElement("usemaxdepth")->GetText());

		if (use_max_depth == "yes")
		{
			ret.m_use_max_depth = true;
		}
		else
		{
			ret.m_use_max_depth = false;
		}

		return ret;
	}

	AmbientOcclusionIntegratorParams Configuration::get_ao_integrator_params() const
	{
		AmbientOcclusionIntegratorParams ret;

		const XMLElement* params = m_integrator_file.FirstChildElement("integrator")->FirstChildElement("params");

		ret.m_samples = static_cast<uint32_t>(atoi(params->FirstChildElement("samples")->GetText()));
		ret.m_radius = static_cast<float>(atof(params->FirstChildElement("radius")->GetText()));

		return ret;
	}

	RayCastIntegratorParams Configuration::get_raycast_integrator_params() const
	{
		RayCastIntegratorParams ret;

		const XMLElement* params = m_integrator_file.FirstChildElement("integrator")->FirstChildElement("params");

		const std::string shade(params->FirstChildElement("shade")->GetText());

		if (shade == "primitive_id")
		{
			ret.m_shade = SHADE_PRIMITIVE_ID;
		}
		else if (shade == "material_id")
		{
			ret.m_shade = SHADE_MATERIAL_ID;
		}
		else if (shade == "normals")
		{
			ret.m_shade = SHADE_NORMALS;
		}
		else
		{
			// Shade one uniform random color.
			ret.m_shade = SHADE_UNIFORM;
		}

		return ret;
	}

	BidirectionalPathIntegratorLvcParams Configuration::get_bdpt_lvc_integrator_params() const
	{
		BidirectionalPathIntegratorLvcParams ret;

		const XMLElement* params = m_integrator_file.FirstChildElement("integrator")->FirstChildElement("params");

		// Get mode.
		const std::string mode(params->FirstChildElement("mode")->GetText());

		if (mode == "sk")
		{
			ret.m_mode = BidirectionalPathIntegratorMode::MODE_BDPT_LVC_SK;
		}
		else if (mode == "mk")
		{
			ret.m_mode = BidirectionalPathIntegratorMode::MODE_BDPT_LVC_MK;
		}
		else if (mode == "sort")
		{
			ret.m_mode = BidirectionalPathIntegratorMode::MODE_BDPT_SORTED_LVC;
		}
		else
		{
			std::cout << "Unsupported Bidirectional Path Integrator mode selected. Switching to default integrator mode of single kernel.";
			ret.m_mode = BidirectionalPathIntegratorMode::MODE_BDPT_LVC_SK;
		}

		// Get preparation phase params.
		ret.m_num_prep_paths = static_cast<uint32_t>(atoi(params->FirstChildElement("preparation")->FirstChildElement("numpaths")->GetText()));
		
		// Get light trace params.
		ret.m_light_path_rrstart = static_cast<uint32_t>(atoi(params->FirstChildElement("lighttrace")->FirstChildElement("rrstart")->GetText()));
		ret.m_num_light_path_max_depth = static_cast<uint32_t>(atoi(params->FirstChildElement("lighttrace")->FirstChildElement("maxdepth")->GetText()));
		ret.m_num_light_paths = static_cast<uint32_t>(atoi(params->FirstChildElement("lighttrace")->FirstChildElement("numpaths")->GetText()));
		
		// Get camera trace params.
		ret.m_cam_path_rrprob = static_cast<float>(atof(params->FirstChildElement("cameratrace")->FirstChildElement("rrprob")->GetText()));
		ret.m_cam_path_rrstart = static_cast<uint32_t>(atoi(params->FirstChildElement("cameratrace")->FirstChildElement("rrstart")->GetText()));
		ret.m_num_cam_path_max_depth = static_cast<uint32_t>(atoi(params->FirstChildElement("cameratrace")->FirstChildElement("maxdepth")->GetText()));
		
		// Get additional two paramets.
		ret.m_num_cam_path_connections = static_cast<uint32_t>(atoi(params->FirstChildElement("cameratrace")->FirstChildElement("sample_N")->GetText()));
		ret.m_num_cam_path_sample_connections = static_cast<uint32_t>(atoi(params->FirstChildElement("cameratrace")->FirstChildElement("sample_M")->GetText()));

		return ret;
	}

	FilterParams Configuration::get_filter_params() const
	{
		FilterParams ret;

		const XMLElement* fparams = m_config_file.FirstChildElement("config")->FirstChildElement("renderer")->FirstChildElement("params")->FirstChildElement("filter")->FirstChildElement("params");
		
		const std::string filter_type(fparams->FirstChildElement("type")->GetText());

		ret.xwidth = static_cast<float>(atof(fparams->FirstChildElement("xwidth")->GetText()));
		ret.ywidth = static_cast<float>(atof(fparams->FirstChildElement("ywidth")->GetText()));

		if (filter_type == "box")
		{
			ret.m_type = FilterType::FILTER_BOX;
			return ret;
		}
		else if (filter_type == "triangle")
		{
			ret.m_type = FilterType::FILTER_TRIANGLE;
			return ret;
		}
		else if (filter_type == "gaussian")
		{
			ret.alpha = static_cast<float>(atof(fparams->FirstChildElement("alpha")->GetText()));
			ret.m_type = FilterType::FILTER_GAUSSIAN;
			return ret;
		}
		else if (filter_type == "mitchell")
		{
			ret.m_type = FilterType::FILTER_MITCHELL;
			return ret;
		}
		else if (filter_type == "sinc")
		{
			ret.m_type = FilterType::FILTER_SINC;
			return ret;
		}
		else
		{
			ret.m_type = FilterType::FILTER_UNKNOWN;
			return ret;
		}
	}
}
