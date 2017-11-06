
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
#include <core/mesh.h>

// Cuda specific headers.

// Standard c++ headers.

namespace renderbox2
{

	//
	// Utility methods for various mesh related functions.
	//


	// Compute the bounding box enclosing the mesh.
	
	AABB compute_mesh_bounds(const Mesh& m)
	{
		AABB ret;
		for (size_t i = 0; i < m.m_vertices.size(); i++)
			ret.insert(m.m_vertices[i]);
		return ret;
	}


	// Set material id for the mesh.

	void set_material_id(Mesh& m, uint32_t mesh_id)
	{
		m.m_mat_ids.resize(m.m_vtx_indices.size());
		fill(m.m_mat_ids.begin(), m.m_mat_ids.end(), mesh_id);
	}
}
