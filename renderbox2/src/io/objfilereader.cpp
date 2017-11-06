
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
#include <io/objfilereader.h>

// Cuda specific headers.

// Standard c++ headers.
#include <cassert>
#include <fstream>
#include <iostream>

namespace renderbox2
{

	bool ObjFileReader::is_comment(const std::string& line) const
	{
		size_t position = line.find("#");
		if (position == string::npos) return false;
		else return true;
	}
	
	Tokens ObjFileReader::tokenize(const std::string& line) const
	{
		Tokens ret;
		char* delimiters = " ";
		char* str = const_cast<char*>(line.c_str());
		char * pch = nullptr;
		char* next_token = nullptr;

		pch = strtok_s(str, delimiters, &next_token);
		while (pch != nullptr)
		{
			ret.m_tokens.push_back(string(pch));
			pch = strtok_s(nullptr, delimiters, &next_token);
		}
		return ret;
	}

	GIndexVec3Type ObjFileReader::parse_face(const string& line, bool& has_vtx, bool& has_normal, bool& has_tex) const
	{
		int ret[3] = { -1, -1, -1 };		// default indices
		int times = 0;
		
		char* delimiters = "/";
		char* pch = nullptr;
		char* next_token = nullptr;
		char* str = const_cast<char*>(line.c_str());
		
		pch = strtok_s(str, delimiters, &next_token);
		
		while (pch != nullptr)
		{
			ret[times] = atoi(pch);
			pch = strtok_s(nullptr, delimiters, &next_token);
			times++;
		}
		// check for the existence of a // string in the line
		// if so, its a v//vn format. 
		if (times == 1)
		{
			has_vtx = true;
			has_normal = false;
			has_tex = false;
		}
		else if (times == 2)
		{
			size_t position = line.find("//");
			if (position == string::npos)
			{
				// we have v//vn format
				has_vtx = true;
				has_normal = true;
				has_tex = false;
				swap(ret[2], ret[1]);		// swap the results
			}
			else
			{
				// else we have v/vt format
				has_vtx = true;
				has_normal = false;
				has_tex = true;
			}
		}
		else
		{
			has_vtx = true;
			has_normal = true;
			has_tex = true;
		}
		
		GIndexVec3Type ret_val = { ret[0], ret[1], ret[2] };
		return ret_val;
	}

	//
	// Note: filename has to be in absolute path.
	//

	bool ObjFileReader::read_file(const std::string& filename, std::vector<Mesh>& meshes) const
	{
		std::ifstream file(filename);
		
		try
		{
			if (file.fail())
			{
				throw runtime_error((string("Unable to open file :") + filename).c_str());
			}

			// Storage and delimter data.
			
			char line[1024];
			char* delimiters = " ";
			float fx, fy, fz;		// floats required
			char* val = nullptr;

			vector<Mesh>::pointer current_mesh = nullptr;

			while (file.peek() != EOF)
			{
				file.getline(line, sizeof(line), '\n');
				string inputString(line);
				if (!is_comment(inputString) && inputString.length() > 1)
				{
					// Now the line might be the start of an object, or might contain data for the object.
					// If it contains an object 'o' then create a new object.
					Tokens tokens = tokenize(inputString);
					
					if (is_object(tokens))
					{
						// add a new mesh
						Mesh mesh;
						mesh.m_name = tokens.m_tokens[1];		// the second token is the mesh name
						meshes.push_back(mesh);
						
						// Set the iterator to point to the current mesh.
						current_mesh = &meshes.back();
					}
					
					// Process vertices.
					else if (is_vertex(tokens))
					{
						assert(tokens.m_tokens.size() == 4);
						fx = (float)atof(tokens.m_tokens[1].c_str());
						fy = (float)atof(tokens.m_tokens[2].c_str());
						fz = (float)atof(tokens.m_tokens[3].c_str());
						float3 vtx = make_float3(fx, fy, fz);
						current_mesh->m_bounds.insert(vtx);
						current_mesh->m_vertices.push_back(vtx);
					}
					
					// Process normals.
					else if (is_normal(tokens))
					{
						assert(tokens.m_tokens.size() == 4);
						fx = (float)atof(tokens.m_tokens[1].c_str());
						fy = (float)atof(tokens.m_tokens[2].c_str());
						fz = (float)atof(tokens.m_tokens[3].c_str());
						current_mesh->m_normals.push_back(make_float3(fx, fy, fz));
					}
					
					// Process texture coordinates.
					else if (is_tex(tokens))
					{
						assert(tokens.m_tokens.size() == 3);
						fx = (float)atof(tokens.m_tokens[1].c_str());
						fy = (float)atof(tokens.m_tokens[2].c_str());
						current_mesh->m_texcoords.push_back(make_float2(fx, fy));
					}
					
					// Process faces.
					else if (is_face(tokens))
					{
						bool hasVtx, hasNor, hasTex;
						GIndexVec3Type v1 = parse_face(tokens.m_tokens[1], hasVtx, hasNor, hasTex);
						GIndexVec3Type v2 = parse_face(tokens.m_tokens[2], hasVtx, hasNor, hasTex);
						GIndexVec3Type v3 = parse_face(tokens.m_tokens[3], hasVtx, hasNor, hasTex);

						if (hasVtx)
							current_mesh->m_vtx_indices.push_back(make_uint3(v1.x - 1, v2.x - 1, v3.x - 1));
						if (hasTex)
							current_mesh->m_tex_indices.push_back(make_uint3(v1.y - 1, v2.y - 1, v3.y - 1));
						if (hasNor)
							current_mesh->m_nor_indices.push_back(make_uint3(v1.z - 1, v2.z - 1, v3.z - 1));
					}
				}
			}
		}
		catch (exception e)
		{
			std::cerr << e.what() << std::endl;
			file.close();
			return false;						// return immediately.
		}

		// Close the file and return.
		file.close();
		return true;
	}

}
