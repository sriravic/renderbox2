
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
#include <core/defs.h>
#include <memory/memoryallocator.h>

// Cuda specific headers.

// Standard c++ headers.

namespace renderbox2
{

	//
	// MemoryAllocator class definition.
	//

	MemoryAllocator::MemoryAllocator()
	{
		m_ids = 0;							// reset the starting id to be allocated as zero.
		m_total_size = 0;
	}

	// The destructor works on the principle that deallocation permanently removes the 
	
	MemoryAllocator::~MemoryAllocator()
	{
		for (auto i : m_data)
		{	
			CUDA_SAFE_RELEASE(std::get<0>(i).m_ptr);	// safely release memory.
			std::get<1>(i) = 0;							// reset the size of current memory to zero.
		}
		
		m_ids = 0;
	}

	DevicePointer MemoryAllocator::allocate(const size_t bytes)
	{
		DevicePointer ret;
		ret.m_id = m_ids++;
		checkCuda(cudaMalloc((void**)&ret.m_ptr, bytes));
		checkCuda(cudaMemset(ret.m_ptr, 0, bytes));

		// If everything is okay insert into the list.
		// Add the total size and return the reference.
		m_data.push_back(std::make_pair(ret, bytes));
		m_total_size += bytes;

		return std::get<0>(m_data[ret.m_id]);
	}

	void MemoryAllocator::free(const uint32_t id)
	{
		CUDA_SAFE_RELEASE(std::get<0>(m_data[id]).m_ptr);
		size_t& ptr_size = std::get<1>(m_data[id]);
		m_total_size -= ptr_size;								// subtract from global pool size
		ptr_size = 0;											// reset pointer size to 0
	}

	void MemoryAllocator::free(const DevicePointer& ptr)
	{
		const uint32_t id = ptr.m_id;
		CUDA_SAFE_RELEASE(std::get<0>(m_data[id]).m_ptr);
		size_t& ptr_size = std::get<1>(m_data[id]);
		m_total_size -= ptr_size;
		ptr_size = 0;
	}
}
