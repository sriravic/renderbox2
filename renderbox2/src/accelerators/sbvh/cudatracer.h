
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

#ifndef CUDA_TRACER_H
#define CUDA_TRACER_H

// Application specific headers.
#include <accelerators/sbvh/cudabvh.h>
#include <accelerators/sbvh/cudatracerkernels.h>
#include <core/buffer.h>
#include <core/defs.h>
#include <core/intersection.h>
#include <core/primitives.h>
#include <memory/memoryallocator.h>

// Cuda specific headers.
#include <cuda_runtime.h>

// Standard c++ headers.

namespace renderbox2
{

	//
	// BVHStruct encapsulates all the bvh pointers for convenient usage.
	//

	struct BvhStruct
	{
#ifdef USE_KERNEL_KEPLER
		cudaTextureObject_t m_nodes;
		cudaTextureObject_t m_tris;
		cudaTextureObject_t m_tri_idx;
#elif defined USE_KERNEL_FERMI
		float4*				m_nodes;
		float4*				m_tris;
		int4*				m_tri_idx;
#endif
	};


	//
	// CudaTracer class helps us in launching streamed versions of the kernels for ray traversal.
	// NOTE: The actual tracer kernel that is called is dependant on the macro USE_KERNEL_XXXXX which determines which is compiled and used.
	//

	class CudaTracer
	{
	public:

		CudaTracer(const CudaBvh& bvh, MemoryAllocator& m);

		~CudaTracer()
		{
			m_allocator.free(m_nodes);
			m_allocator.free(m_tri_woop);
			m_allocator.free(m_tri_idx);

#ifdef USE_KERNEL_KEPLER
			cudaDestroyTextureObject(t_nodes);
			cudaDestroyTextureObject(t_tris);
			cudaDestroyTextureObject(t_tri_idx);
			cudaDestroyTextureObject(t_rays);
#elif defined USE_KERNEL_FERMI
			cudaUnbindTexture(t_nodesA);
			cudaUnbindTexture(t_trisA);
			cudaUnbindTexture(t_triIndices);
#endif
		}

		float trace(const RayBuffer& rb, IntersectionBuffer& isects, SceneBuffer& scene_data, uint8_t* stencil, bool any_hit = true);

		BvhStruct get_bvh() const
		{
			BvhStruct ret;

#ifdef USE_KERNEL_KEPLER
			ret.m_nodes = t_nodes;
			ret.m_tris = t_tris;
			ret.m_tri_idx = t_tri_idx;
#elif defined USE_KERNEL_FERMI
#endif

			return ret;
		}

	private:
		DevicePointer		m_nodes;
		DevicePointer		m_tri_woop;
		DevicePointer		m_tri_idx;
		MemoryAllocator&	m_allocator;

#ifdef USE_KERNEL_KEPLER
		// For kepler we will use cuda texture object to bind the memory.
		cudaTextureObject_t t_nodes;
		cudaTextureObject_t t_tris;
		cudaTextureObject_t t_tri_idx;
		cudaTextureObject_t t_rays;
#endif

	};
}				// !namespace renderbox2

#endif			// !CUDA_TRACER