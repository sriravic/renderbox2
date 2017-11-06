
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
#include <accelerators/sbvh/cudatracer.h>

#ifdef USE_KERNEL_FERMI
#include <accelerators/sbvh/fermi_speculative_while_while.h>
#elif defined USE_KERNEL_KEPLER
#include <accelerators/sbvh/kepler_dynamic_fetch.h>
#endif

#include <util/cudatimer.h>

// Cuda specific headers.

// Standard c++ headers.
#include <stdio.h>

namespace renderbox2
{

	//
	// CudaTracer class implementation.
	//

	CudaTracer::CudaTracer(const CudaBvh& bvh, MemoryAllocator& m) : m_allocator(m)
	{
		// Allocate memory on the device and immediately copy them into it.
		uint32_t m_num_nodes = static_cast<uint32_t>(bvh.get_node_buffer().size());
		uint32_t m_num_tri_woop = static_cast<uint32_t>(bvh.get_tri_woop_buffer().size());
		uint32_t m_num_tri_idx = static_cast<uint32_t>(bvh.get_tri_index_buffer().size());

		m_nodes = m_allocator.allocate(sizeof(uint4) * m_num_nodes);
		m_tri_woop = m_allocator.allocate(sizeof(uint4) * m_num_tri_woop);
		m_tri_idx = m_allocator.allocate(sizeof(uint) * m_num_tri_idx);

		checkCuda(cudaMemcpy(m_nodes.m_ptr, bvh.get_node_buffer().data(), sizeof(uint4) * m_num_nodes, cudaMemcpyHostToDevice));
		checkCuda(cudaMemcpy(m_tri_woop.m_ptr, bvh.get_tri_woop_buffer().data(), sizeof(uint4) * m_num_tri_woop, cudaMemcpyHostToDevice));
		checkCuda(cudaMemcpy(m_tri_idx.m_ptr, bvh.get_tri_index_buffer().data(), sizeof(GIndexType) * m_num_tri_idx, cudaMemcpyHostToDevice));

		// Bind them appropriately to texture references based on architecture for compiled version.
#ifdef USE_KERNEL_FERMI

		// Create format descriptors.
		cudaChannelFormatDesc fdesc4 = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);	// 4 component float desc - read element as it is
		cudaChannelFormatDesc idesc1 = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindSigned);	    // 1 component desc - read element as it is

		// convert pointers and bind them.
		// Rays are bound only during traversal where we have access to them.
		float4* nodePtr = static_cast<float4*>(m_nodes.m_ptr);
		float4* triPtr  = static_cast<float4*>(m_tri_woop.m_ptr);
		GIndexType* triIdxPtr = static_cast<GIndexType*>(m_tri_idx.m_ptr);

		cudaBindTexture(NULL, &t_nodesA, nodePtr, &fdesc4, sizeof(float4) * m_num_nodes);
		cudaBindTexture(NULL, &t_trisA, triPtr, &fdesc4, sizeof(float4) * m_num_tri_woop);
		cudaBindTexture(NULL, &t_triIndices, triIdxPtr, &idesc1, sizeof(GIndexType) * m_num_tri_idx);

#elif defined USE_KERNEL_KEPLER

		// Create resource Descriptiors.
		cudaResourceDesc rdesc1, rdesc2, rdesc3;
		memset(&rdesc1, 0, sizeof(rdesc1));
		memset(&rdesc2, 0, sizeof(rdesc2));
		memset(&rdesc3, 0, sizeof(rdesc3));

		// Node data.
		rdesc1.resType = cudaResourceTypeLinear;
		rdesc1.res.linear.devPtr = m_nodes.m_ptr;
		rdesc1.res.linear.desc.f = cudaChannelFormatKindFloat;
		rdesc1.res.linear.desc.x = 32;
		rdesc1.res.linear.desc.y = 32;
		rdesc1.res.linear.desc.z = 32;
		rdesc1.res.linear.desc.w = 32;
		rdesc1.res.linear.sizeInBytes = m_allocator.size(m_nodes.m_id);

		// Triangle data.
		rdesc2.resType = cudaResourceTypeLinear;
		rdesc2.res.linear.devPtr = m_tri_woop.m_ptr;
		rdesc2.res.linear.desc.f = cudaChannelFormatKindFloat;
		rdesc2.res.linear.desc.x = 32;
		rdesc2.res.linear.desc.y = 32;
		rdesc2.res.linear.desc.z = 32;
		rdesc2.res.linear.desc.w = 32;
		rdesc2.res.linear.sizeInBytes = m_allocator.size(m_tri_woop.m_id);
		
		// Tri-idx data.
		// NOTE: We are assuming 32bit indices. For 64 bit indices we have to assign ids correctly.
		rdesc3.resType = cudaResourceTypeLinear;
		rdesc3.res.linear.devPtr = m_tri_idx.m_ptr;
		rdesc3.res.linear.desc.f = cudaChannelFormatKindUnsigned;
		rdesc3.res.linear.desc.x = 32;
		rdesc3.res.linear.desc.y = 0;
		rdesc3.res.linear.desc.z = 0;
		rdesc3.res.linear.desc.w = 0;
		rdesc3.res.linear.sizeInBytes = m_allocator.size(m_tri_idx.m_id);

		cudaTextureDesc tdesc;
		memset(&tdesc, 0, sizeof(tdesc));
		tdesc.readMode = cudaReadModeElementType;

		t_nodes = 0;
		t_rays = 0;
		t_tris = 0;
		t_tri_idx = 0;

		checkCuda(cudaCreateTextureObject(&t_nodes, &rdesc1, &tdesc, NULL));
		checkCuda(cudaCreateTextureObject(&t_tris, &rdesc2, &tdesc, NULL));
		checkCuda(cudaCreateTextureObject(&t_tri_idx, &rdesc3, &tdesc, NULL));
#endif
	}


	//
	// Trace function calls the correct kernel based on compiled architecture.
	//

	float CudaTracer::trace(const RayBuffer& rb, IntersectionBuffer& isect, SceneBuffer& scene_data, uint8_t* stencil, bool any_hit)
	{

		// Check if the stencil buffer is to be used.
		bool mask = (stencil != nullptr);

		// call kernel.
#ifdef USE_KERNEL_FERMI
#elif defined USE_KERNEL_KEPLER

		// Bind rays.
		cudaResourceDesc rdesc;
		memset(&rdesc, 0, sizeof(rdesc));
		rdesc.resType = cudaResourceTypeLinear;
		rdesc.res.linear.devPtr = rb.m_data;
		rdesc.res.linear.desc.f = cudaChannelFormatKindFloat;
		rdesc.res.linear.desc.x = 32;
		rdesc.res.linear.desc.y = 32;
		rdesc.res.linear.desc.z = 32;
		rdesc.res.linear.desc.w = 32;
		rdesc.res.linear.sizeInBytes = rb.m_size * sizeof(Ray);

		cudaTextureDesc desc;
		memset(&desc, 0, sizeof(desc));
		desc.readMode = cudaReadModeElementType;

		cudaTextureObject_t t_rays = 0;
		checkCuda(cudaCreateTextureObject(&t_rays, &rdesc, &desc, NULL));

		// Set the device counter variable.
		int init = 0;
		checkCuda(cudaMemcpyToSymbol(g_warpCounter, &init, sizeof(int), 0, cudaMemcpyHostToDevice));

		// Kepler uses a persistent threads approach to trace rays.
		CudaTimer t1("tracer timer");

		int2 block_size = make_int2(32, 4);
		int desired_warps = 720;
		int block_warps = (block_size.x * block_size.y + 31) / 32;
		int num_blocks = (desired_warps + block_warps - 1) / block_warps;

		t1.start();
		kernel_kepler_trace<<<dim3(num_blocks, 1, 1), dim3(block_size.x, block_size.y, 1)>>>(rb.m_size, any_hit, (float4*)(rb.m_data), isect, t_nodes, t_tris, t_tri_idx, scene_data, stencil, mask);
		t1.stop();

		checkCuda(cudaDestroyTextureObject(t_rays));

#endif
		return 0.0f;
	}
}
