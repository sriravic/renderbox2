
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
#include <3rdparty/tinyxml2/tinyxml2.h>
#include <core/camera.h>
#include <core/util.h>
#include <io/customfilereader.h>
#include <io/objfilereader.h>

// Cuda specific headers.

// Standard c++ headers.
#include <cassert>
#include <cstdint>
#include <iostream>
#include <map>
#include <omp.h>
#include <vector>

using namespace tinyxml2;
using namespace std;

namespace renderbox2
{

	//
	// Custom File reader class implementation.
	//

	bool CustomFileReader::process(Scene& scene)
	{
		cout << "Processing Scene File. This might take a few minutes." << endl;
		cout << "-----------------------------------------------------" << endl;

		double start = omp_get_wtime();

		tinyxml2::XMLDocument scenefile;
		string s = m_scene_file_path + m_scene_file_name;
		scenefile.LoadFile(s.c_str());
		XMLError error = scenefile.ErrorID();
		if (error != XMLError::XML_SUCCESS) {
			printf("SEVERE : %s\n", scenefile.GetErrorStr1());
			return false;
		}

		XMLElement* root = scenefile.FirstChildElement("scene");
		XMLElement* count = root->FirstChildElement("files")->FirstChildElement("count");
		XMLElement* file = root->FirstChildElement("files")->FirstChildElement("file");
		uint32_t num_obj_files_to_load = atoi(count->GetText());								// as of now we only support 1 obj file loading with all the objects in it.

		assert(num_obj_files_to_load == 1);

		vector<Mesh> meshes;

		cout << "Reading mesh file." << endl;
		// As of now we process only obj files.
		ObjFileReader objfilereader;
		if (!objfilereader.read_file(m_scene_file_path + string(file->GetText()), meshes))
		{
			cerr << "Error reading obj file" << endl;
			return false;
		}

		// Now process all materials and assign them to the meshes.
		uint32_t material_id = 0;
		map<string, uint32_t> material_map;
		vector<pair<string, uint32_t>> light_list;

		// We need global counters to keep track of the indices of each material's layers bsdf to the global list.
		
		uint32_t diffuse_bsdf_index = 0;
		uint32_t orennayar_bsdf_index = 0;
		uint32_t glass_bsdf_index = 0;
		uint32_t mirror_bsdf_index = 0;
		uint32_t microfacet_bsdf_index = 0;

		cout << "Reading material list." << endl;
		for (XMLElement* element = root->FirstChildElement("material"); element != nullptr; element = element->NextSiblingElement("material"))
		{
			// Create the material first.
			Material material;

			XMLElement* material_name = element->FirstChildElement("name");
			material_map.insert(make_pair(string(material_name->GetText()), material_id));
			
			XMLElement* bsdf = element->FirstChildElement("bsdf");
			XMLElement* layers = bsdf->FirstChildElement("layers");
			uint32_t num_layers = static_cast<uint32_t>(atoi(layers->GetText()));

			assert(num_layers < MAX_LAYERS);
			uint32_t layer_index = 0;
			
			material.num_valid_layers = num_layers;
			material.m_emitter = 0;

			for (XMLElement* bxdf = bsdf->FirstChildElement("bxdf"); bxdf != nullptr; bxdf = bxdf->NextSiblingElement("bxdf"))
			{
				XMLElement* bxdf_type = bxdf->FirstChildElement("type");
				if (string(bxdf_type->GetText()) == "lambertian")
				{
					material.layer_bxdf_type[layer_index] = BxDFType(BxDF_REFLECTION | BxDF_DIFFUSE);
					material.layer_bsdf_type[layer_index] = BSDF_LAMBERTIAN;
					material.layer_bsdf_id[layer_index] = diffuse_bsdf_index;
					
					LambertianBsdfParams params;
					params.color = make_float4(get_float3(string(bxdf->FirstChildElement("reflectance")->GetText())), 0.0f);
					
					// Push in the scene diffuse list.
					scene.add_diffuse_bsdf(params);

					diffuse_bsdf_index++;
				}
				else if (string(bxdf_type->GetText()) == "orennayar")
				{
					material.layer_bxdf_type[layer_index] = BxDFType(BxDF_REFLECTION | BxDF_DIFFUSE);
					material.layer_bsdf_type[layer_index] = BSDF_OREN_NAYAR;
					material.layer_bsdf_id[layer_index] = orennayar_bsdf_index;

					OrenNayarBsdfParams params;
					params.color = make_float4(get_float3(string(bxdf->FirstChildElement("reflectance")->GetText())), 0.0f);
					params.sigma = static_cast<float>(atof(bxdf->FirstChildElement("sigma")->GetText()));
					params.sigma = clamp(params.sigma, 0.0f, 90.0f);

					// compute A and B
					params.sigma = to_radians(params.sigma);
					float sigma2 = params.sigma * params.sigma;
					params.AB.x = 1.0f - (sigma2 / (2.0f * (sigma2 + 0.33f)));
					params.AB.y = 0.45f * sigma2 / (sigma2 + 0.09f);

					scene.add_orennayar_bsdf(params);

					orennayar_bsdf_index++;
				}
				else if (string(bxdf_type->GetText()) == "mirror")
				{
					material.layer_bxdf_type[layer_index] = BxDFType(BxDF_REFLECTION | BxDF_SPECULAR);
					material.layer_bsdf_type[layer_index] = BSDF_MIRROR;
					material.layer_bsdf_id[layer_index] = mirror_bsdf_index;

					MirrorBsdfParams params;
					params.color = make_float4(get_float3(string(bxdf->FirstChildElement("reflectance")->GetText())), 0.0f);

					scene.add_mirror_bsdf(params);

					mirror_bsdf_index++;
				}
				else if (string(bxdf_type->GetText()) == "glass")
				{

					material.layer_bsdf_id[layer_index] = glass_bsdf_index;
					material.layer_bsdf_type[layer_index] = BSDF_GLASS;
					material.layer_bxdf_type[layer_index] = BxDFType(BxDF_REFLECTION | BxDF_SPECULAR | BxDF_TRANSMISSION);

					GlassBsdfParams params;

					params.color = make_float4(get_float3(string(bxdf->FirstChildElement("reflectance")->GetText())), 0.0f);
					params.etai = static_cast<float>(atof(bxdf->FirstChildElement("etai")->GetText()));
					params.etat = static_cast<float>(atof(bxdf->FirstChildElement("etat")->GetText()));

					scene.add_glass_bsdf(params);

					glass_bsdf_index++;
				}
				else if (string(bxdf_type->GetText()) == "microfacet")
				{
					material.layer_bsdf_id[layer_index] = microfacet_bsdf_index;
					material.layer_bsdf_type[layer_index] = BSDF_BLINN_MICROFACET;
					material.layer_bxdf_type[layer_index] = BxDFType(BxDF_REFLECTION | BxDF_GLOSSY);

					MicrofacetBsdfParams params;

					params.R = make_float4(get_float3(string(bxdf->FirstChildElement("reflectance")->GetText())), 0.0f);

					string distribution(bxdf->FirstChildElement("distribution")->GetText());
					
					if (distribution == "blinn")
					{
						params.exponent = static_cast<float>(atof(bxdf->FirstChildElement("exponent")->GetText()));
					}

					string type(bxdf->FirstChildElement("fresnel_type")->GetText());

					if (type == "fresnel_conductor")
					{
						params.type = FresnelType::FRESNEL_CONDUCTOR;
						params.eta = make_float4(get_float3(string(bxdf->FirstChildElement("eta")->GetText())), 0.0f);
						params.k = make_float4(get_float3(string(bxdf->FirstChildElement("k")->GetText())), 0.0f);
					}
					else if (type == "fresnel_dielectric")
					{
						params.type = FresnelType::FRESNEL_DIELECTRIC;
						params.etai = static_cast<float>(atof(bxdf->FirstChildElement("etai")->GetText()));
						params.etat = static_cast<float>(atof(bxdf->FirstChildElement("etat")->GetText()));
					}
					else if (type == "fresnel_noop")
					{
						params.type = FresnelType::FRESNEL_NOOP;
					}

					scene.add_microfacet_bsdf(params);

					microfacet_bsdf_index++;
				}				
				layer_index++;
			}
			
			// add the material to the scene.
			scene.add_material(material);
			material_id++;
		}

		// Add materials to the objects.
		cout << "Assigning materials to objects." << endl;
			
		for (XMLElement* object = root->FirstChildElement("object"); object != nullptr; object = object->NextSiblingElement("object"))
		{
			XMLElement* name = object->FirstChildElement("name");
			XMLElement* material = object->FirstChildElement("material");

			uint32_t mat_id = material_map.find(string(material->GetText()))->second;

			for (size_t i = 0; i < meshes.size(); i++)
			{
				if (meshes[i].m_name == string(name->GetText()))
				{
					set_material_id(meshes[i], mat_id);
					break;
				}
			}
		}

		// Process lights within the scene.
		// As of now we support only diffuse emitters.
		cout << "Processing lights." << endl;

		uint32_t nlights = 0;
		for (XMLElement* light = root->FirstChildElement("lightobjects")->FirstChildElement("light"); light != nullptr; light = light->NextSiblingElement("light"))
		{

			Material material;

			XMLElement* name = light->FirstChildElement("name");
			XMLElement* type = light->FirstChildElement("type");
			XMLElement* Le = light->FirstChildElement("Le");

			// Create a material with the emitter flag set and add to the global material list.

			DiffuseEmitterParams params;
			params.color = make_float4(get_float3(string(Le->GetText())), 0.0f);
			
			light_list.push_back(make_pair(name->GetText(), material_id));
			
			material.m_emitter = 1;
			material.num_valid_layers = 1;
			material.layer_bxdf_type[0] = BxDF_DIFFUSE;
			material.layer_bsdf_type[0] = BSDF_LAMBERTIAN;
			material.layer_bsdf_id[0] = nlights;

			scene.add_diffuse_light_bsdf(params);
			scene.add_material(material);

			nlights++;
			material_id++;
		}

		// Add all meshes to the scene.
		cout << "Adding meshes to the scene." << endl;

		for (size_t i = 0; i < meshes.size(); i++)
		{
			bool is_emitter = false;

			for (size_t j = 0; j < light_list.size(); j++)
			{
				string name = light_list[j].first;
				if (meshes[i].m_name == name)
				{
					is_emitter = true;
					set_material_id(meshes[i], light_list[j].second);
					break;
				}

			}
			scene.add_mesh(meshes[i], is_emitter);
		}

		double end = omp_get_wtime();
		
		cout << "Processing scene completed in : " << (end - start) * 1000.0f << "ms" << endl;
		cout << "-----------------------------------------------------" << endl;
		return true;
	}

	bool CustomFileReader::get_camera(PerspectiveCamera** camera, const uint32_t image_width, const uint32_t image_height) const
	{
		tinyxml2::XMLDocument scenefile;
		string s = m_scene_file_path + m_scene_file_name;
		scenefile.LoadFile(s.c_str());
		XMLError error = scenefile.ErrorID();
		if (error != XMLError::XML_SUCCESS) {
			printf("SEVERE : %s\n", scenefile.GetErrorStr1());
			return false;
		}

		XMLElement* root = scenefile.FirstChildElement("scene");
		XMLElement* e = root->FirstChildElement("camera");

		XMLElement* eye = e->FirstChildElement("position");
		XMLElement* lookat = e->FirstChildElement("lookat");
		XMLElement* up = e->FirstChildElement("up");
		XMLElement* fov = e->FirstChildElement("fov");
		XMLElement* fnear = e->FirstChildElement("fnear");
		XMLElement* ffar = e->FirstChildElement("ffar");
		XMLElement* sensor = e->FirstChildElement("sensor");

		*camera = new PerspectiveCamera(get_float3(eye->GetText()),
										get_float3(lookat->GetText()),
										get_float3(up->GetText()),
										make_float2(static_cast<float>(image_width), static_cast<float>(image_height)),
										90.0f);										// NOTE: Hardcoded value for fov. Have to get it from scene file.

		return true;
	}

	float3 CustomFileReader::get_float3(const string& str) const
	{
		char delimiters[] = " {,}";
		char* tempc = new char[str.length()];
		char* val = nullptr;
		char* next_token = nullptr;
		float x, y, z;
		str._Copy_s(tempc, str.length(), str.length());
		val = strtok_s(tempc, delimiters, &next_token);
		if (val != nullptr)
		{
			x = (float)atof(val);
			val = strtok_s(nullptr, delimiters, &next_token);
		}

		if (val != nullptr)
		{
			y = (float)atof(val);
			val = strtok_s(nullptr, delimiters, &next_token);
		}

		if (val != nullptr)
		{
			z = (float)atof(val);
		}

		delete[] tempc;
		return make_float3(x, y, z);
	}


}
