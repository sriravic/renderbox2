
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
#include <core/intersection.h>

// Cuda specific headers.

// Standard c++ headers.

namespace renderbox2
{

	//
	// IntersectionBufferClass implementation.
	//

	IntersectionBufferClass::IntersectionBufferClass(MemoryAllocator& allocator)
		: m_allocator(allocator)
	{
		// Empty constructor.
	}

	IntersectionBufferClass::~IntersectionBufferClass()
	{
		for (uint32_t i = 0; i < 3; i++)
		{
			m_allocator.free(m_position[i]);
			m_allocator.free(m_geometric_normal[i]);
			m_allocator.free(m_shading_normal[i]);
			m_allocator.free(m_tangent[i]);
			m_allocator.free(m_bitangent[i]);
			m_allocator.free(m_incoming_direction[i]);
		}
		m_allocator.free(m_uv);
		m_allocator.free(m_primitive_id);
		m_allocator.free(m_primitive_material_id);
		m_allocator.free(m_front);
		m_allocator.free(m_intersected);
		m_allocator.free(m_epsilon);
		m_allocator.free(m_depth);
		m_allocator.free(m_contribution);
	}

	void IntersectionBufferClass::allocate(GIndexType size)
	{
		for (uint32_t i = 0; i < 3; i++)
		{
			m_position[i]			= m_allocator.allocate(sizeof(float) * size);
			m_geometric_normal[i]	= m_allocator.allocate(sizeof(float) * size);
			m_shading_normal[i]		= m_allocator.allocate(sizeof(float) * size);
			m_tangent[i]			= m_allocator.allocate(sizeof(float) * size);
			m_bitangent[i]			= m_allocator.allocate(sizeof(float) * size);
			m_incoming_direction[i] = m_allocator.allocate(sizeof(float) * size);
		}

		m_uv			= m_allocator.allocate(sizeof(float2) * size);
		m_primitive_id	= m_allocator.allocate(sizeof(GIndexType) * size);
		m_primitive_material_id = m_allocator.allocate(sizeof(uint32_t) * size);
		m_front			= m_allocator.allocate(sizeof(uint8_t) * size);
		m_intersected	= m_allocator.allocate(sizeof(uint8_t) * size);
		m_epsilon		= m_allocator.allocate(sizeof(float) * size);
		m_depth			= m_allocator.allocate(sizeof(uint32_t) * size);
		m_contribution	= m_allocator.allocate(sizeof(float4) * size);
	}

	IntersectionBuffer IntersectionBufferClass::get_buffer() const
	{
		IntersectionBuffer ret;

		for (uint32_t i = 0; i < 3; i++)
		{
			ret.m_position[i]			= static_cast<float*>(m_position[i].m_ptr);
			ret.m_geomteric_normal[i]	= static_cast<float*>(m_geometric_normal[i].m_ptr);
			ret.m_shading_normal[i]		= static_cast<float*>(m_shading_normal[i].m_ptr);
			ret.m_tangent[i]			= static_cast<float*>(m_tangent[i].m_ptr);
			ret.m_bitangent[i]			= static_cast<float*>(m_bitangent[i].m_ptr);
			ret.m_incoming_direction[i] = static_cast<float*>(m_incoming_direction[i].m_ptr);
		}

		ret.m_uv			= static_cast<float2*>(m_uv.m_ptr);
		ret.m_primitive_id	= static_cast<GIndexType*>(m_primitive_id.m_ptr);
		ret.m_primitive_material_id = static_cast<uint32_t*>(m_primitive_material_id.m_ptr);
		ret.m_front			= static_cast<uint8_t*>(m_front.m_ptr);
		ret.m_intersected	= static_cast<uint8_t*>(m_intersected.m_ptr);
		ret.m_epsilon		= static_cast<float*>(m_epsilon.m_ptr);
		ret.m_depth			= static_cast<uint32_t*>(m_depth.m_ptr);
		ret.m_contribution	= static_cast<float4*>(m_contribution.m_ptr);
		ret.m_size			= m_size;

		return ret;
	}

}