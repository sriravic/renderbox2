
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

#ifndef MESH_H
#define MESH_H

// Application specific headers.
#include <core/globaltypes.h>
#include <core/primitives.h>

// Cuda specific headers.
#include <vector_types.h>

// Standard c++ headers.
#include <cstdint>
#include <string>
#include <vector>

using namespace std;

namespace renderbox2
{

	//
	// Mesh struct is used to store all the vtx/nor/tex data of a submesh within the larger scene mesh.
	// Particularly useful when we are reading hierarchical mesh data/submesh data.
	//

	struct Mesh
	{
		string					m_name;
		AABB					m_bounds;
		vector<float3>			m_vertices;
		vector<float3>			m_normals;
		vector<float2>			m_texcoords;

		vector<GIndexVec3Type>	m_vtx_indices;
		vector<GIndexVec3Type>	m_nor_indices;
		vector<GIndexVec3Type>	m_tex_indices;

		vector<GIndexVec3Type>	m_vtx_indices_local;
		vector<GIndexVec3Type>	m_nor_indices_local;
		vector<GIndexVec2Type>	m_tex_indices_local;

		vector<uint32_t>		m_mat_ids;						// We will never have more materials than this ever!.
	};

	
	//
	// Helper methods for certain mesh operations.
	//

	AABB compute_mesh_bounds(const Mesh& m);


	//
	// Function to set the material id for a submesh from the global list of materials.
	// This function resizes the mat_ids buffer of the mesh equal to the number of triangles within the mesh.
	//

	void set_material_id(Mesh& m, uint32_t mesh_id);

}				// !namespace renderbox2

#endif			// !MESH_H
