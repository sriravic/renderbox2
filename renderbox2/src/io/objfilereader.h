
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

#ifndef OBJ_FILE_READER_H
#define OBJ_FILE_READER_H

// Application specific headers.
#include <core/mesh.h>

// Cuda specific headers.

// Standard c++ headers.
#include <string>

namespace renderbox2
{

	//
	// Tokens struct.
	// 

	struct Tokens
	{
		std::vector<std::string> m_tokens;
	};

	
	//
	// Obj File Reader class
	//

	class ObjFileReader
	{
	public:
		
		//
		// read_file method reads an obj file and returns all the submeshes in the file in a vector.
		// if all the objects in a scene are defined as one object only then only one submesh is returned.
		//

		bool read_file(const std::string& filename, std::vector<Mesh>& meshes) const;


		//
		// Returns if a given line is comment line starting with #.
		//

		bool is_comment(const std::string& line) const;


		//
		// Returns a variety of mesh attributes from the line parsed in the file.
		//
		
		bool is_face(const Tokens& tokens) const   { if (!strcmp(tokens.m_tokens[0].c_str(), "f"))   return true; else return false; }
		
		bool is_vertex(const Tokens& tokens) const { if (!strcmp(tokens.m_tokens[0].c_str(), "v"))   return true; else return false; }
		
		bool is_normal(const Tokens& tokens) const { if (!strcmp(tokens.m_tokens[0].c_str(), "vn"))  return true; else return false; }
		
		bool is_tex(const Tokens& tokens) const    { if (!strcmp(tokens.m_tokens[0].c_str(), "vt")) return true; else return false; }
		
		bool is_object(const Tokens&tokens) const  { if (!strcmp(tokens.m_tokens[0].c_str(), "o"))   return true; else return false; }


		//
		// Returns all the individual tokens of a line.
		//

		Tokens tokenize(const std::string& line) const;


		//
		// Parse a triangle face from obj file.
		// Note: As of now we support only triangular meshes in obj files.
		//

		GIndexVec3Type parse_face(const string& line, bool& has_vtx, bool& has_normal, bool& has_tex) const;

	};

}			// !namespace renderbox2

#endif		// !OBJ_FILE_READER_H