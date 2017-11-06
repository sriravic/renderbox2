
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

#ifndef MEMORY_ALLOCATOR_H
#define MEMORY_ALLOCATOR_H

// Application specific headers.

// Cuda specific headers.

// Standard c++ headers.
#include <cstdint>
#include <vector>
#include <tuple>

namespace renderbox2
{
	
	//
	// DevicePointer is an encapsulating structure that can be used to identify memory allocations using an UUID
	//

	struct DevicePointer
	{
		void*		m_ptr;
		uint32_t	m_id;		// unique id to identify the pointer.
	};
	

	//
	// Utility methods to get different sizes of data.
	//

	static inline size_t get_kb(const size_t bytes)
	{
		return (bytes / (1 << 10));
	}

	static inline size_t get_mb(const size_t bytes)
	{
		return (bytes / (1 << 20));
	}

	static inline size_t get_gb(const size_t bytes)
	{
		return (bytes / (1 << 30));
	}

	//
	// All device memory allocations happen through the memory allocator.
	// It provides functions with which memory is allocated on the gpu.
	// Each memory allocation is also provided with an unique id used to identify the pointer and so determine how much memory is being
	// currently used by the application. 
	// NOTE: All device memory allocations must go through this allocator only.!
	// NOTE: We don't consider allocate memory on host CPU side through this allocator.

	class MemoryAllocator
	{
	public:

		MemoryAllocator();

		~MemoryAllocator();

		// Allocate memory for 'size' bytes.
		DevicePointer			allocate(const size_t size);
		
		void					free(const uint32_t id);
		
		void                    free(const DevicePointer& ptr);

		// Gets the total bytes of data allocated so far by the application.
		size_t					total_size() const { return m_total_size; }

		// Gets the total bytes of data allocated for a particular memory chunk.
		size_t					size(const uint32_t id) const { return std::get<1>(m_data[id]); }
		
		// Get routines to get const data as well as data references.
		DevicePointer&			get(const uint32_t id) { return std::get<0>(m_data[id]); }
		const DevicePointer&	get(const uint32_t id) const { return std::get<0>(m_data[id]); }
		
	private:
		std::vector<std::pair<DevicePointer, size_t>>	m_data;
		uint32_t										m_ids;				// unique running identifier.
		size_t											m_total_size;		// total memory allocated.
	};
}			// !namespace renderbox2

#endif		// !MEMORY_ALLOCATOR_H
