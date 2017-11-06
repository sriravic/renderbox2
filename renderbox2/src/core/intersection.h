
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

#ifndef INTERSECTION_H
#define INTERSECTION_H

// Application specific headers.
#include <core/globaltypes.h>
#include <memory/memoryallocator.h>

// Cuda specific headers.
#include <vector_types.h>

// Standard c++ headers.

namespace renderbox2
{

	//
	// Intersection structure that is computed during ray traversal.
	// Total size = 89 bytes.
	// pbrt and other rendering engines store pointers to the underlying bsdf also. 
	// renderbox2 is designed with complex materials in hand and hence any access to materials goes through primitive id.
	//

	struct Intersection
	{
		float3		m_position;					// intersection position
		float3		m_Ng;						// gemoetric normal
		float3		m_Ns;						// shading normal
		float3		m_tangent;					// tangent
		float3		m_bitangent;				// bitangent
		float3		m_wi;						// incoming direction of the ray that generated this intersection.
		float2		m_uv;						// barycentric coordinates
		GIndexType	m_primitive_id;				// intersected primitive id
		uint32_t	m_primitive_material_id;	// Material id of the intersected primitive
		uint8_t		m_front;					// front face or back face of the primitive intersected.
		uint8_t		m_intersected;				// flag to indicate whether the current intersection is valid
		float		m_epsilon;					// small epsilon value used in intersections.
	};

	
	//
	// IntersectionBuffer is a struct containing device pointers to intersection data.
	//

	struct IntersectionBuffer
	{
		float*		m_position[3];
		float*		m_geomteric_normal[3];
		float*		m_shading_normal[3];
		float*		m_tangent[3];
		float*		m_bitangent[3];
		float*		m_incoming_direction[3];
		float2*		m_uv;
		GIndexType* m_primitive_id;
		uint32_t*	m_primitive_material_id;
		uint8_t*	m_front;
		uint8_t*	m_intersected;
		float*		m_epsilon;
		uint32_t*	m_depth;
		float4*		m_contribution;
		GIndexType	m_size;
	};


	//
	// Intersection Buffer is responsible for maintaining stream of intersections.
	//

	class IntersectionBufferClass
	{
	public:

		IntersectionBufferClass(MemoryAllocator& allocator);

		~IntersectionBufferClass();

		void allocate(GIndexType m_size);

		IntersectionBuffer get_buffer() const;

	private:

		MemoryAllocator&	m_allocator;
		
		// Actual Data.
		DevicePointer		m_position[3];
		DevicePointer		m_geometric_normal[3];
		DevicePointer		m_shading_normal[3];
		DevicePointer		m_tangent[3];
		DevicePointer		m_bitangent[3];
		DevicePointer		m_incoming_direction[3];
		DevicePointer		m_uv;
		DevicePointer		m_primitive_id;
		DevicePointer		m_primitive_material_id;
		DevicePointer		m_front;
		DevicePointer		m_intersected;
		DevicePointer		m_epsilon;
		DevicePointer		m_depth;
		DevicePointer		m_contribution;
		
		GIndexType			m_size;
	};
}			// !namespace renderbox2

#endif		// !INTERSECTION_H