
/*
*  Copyright (c) 2009-2011, NVIDIA Corporation
*  All rights reserved.
*
*  Redistribution and use in source and binary forms, with or without
*  modification, are permitted provided that the following conditions are met:
*      * Redistributions of source code must retain the above copyright
*        notice, this list of conditions and the following disclaimer.
*      * Redistributions in binary form must reproduce the above copyright
*        notice, this list of conditions and the following disclaimer in the
*        documentation and/or other materials provided with the distribution.
*      * Neither the name of NVIDIA Corporation nor the
*        names of its contributors may be used to endorse or promote products
*        derived from this software without specific prior written permission.
*
*  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
*  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
*  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
*  DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
*  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
*  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
*  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
*  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
*  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
*  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

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

//
// Note: This file has been modified to better suit the architecture of renderbox2.
//       As such we support only fermi/kepler class devices for rendering.
//

#ifndef CUDA_BVH_H
#define CUDA_BVH_H

// Application specific headers.
#include <accelerators/sbvh/bvh.h>
#include <accelerators/sbvh/cudatracerkernels.h>

// Cuda specific headers.

// Standard c++ headers.


//
// CudaBvhStruct is an encapsulating structure that can be used for passing bvh related data to the gpu much more easily.
// NOTE: No memory management is ever done by this struct. This just encapuslates.
//

struct CudaBvhStruct
{
	uint4*		m_nodes;
	uint4*		m_tri_woop;
	GIndexType* m_tri_index;
};

class CudaBvh
{
public:
	
	CudaBvh(const BVH& bvh, BVHLayout layout);
	
	CudaBvh(const std::string& filename, bool& isLoadedProperly);
	
	//
	// Serialize a cuda bvh into memory.
	//

	void serialize(const std::string& filename);


	//
	// CudaBVH various get methods.
	//

	const std::vector<uint4>&       get_node_buffer(void) const			{ return m_nodes; }
	const std::vector<uint4>&       get_tri_woop_buffer(void) const		{ return m_tri_woop; }
	const std::vector<GIndexType>&	get_tri_index_buffer(void) const	{ return m_tri_index; }
	BVHLayout						get_layout() const					{ return m_layout; }

private:
	
	//
	// We have two versions of bvh with different nodoffsets.
	// BVHCompact is used for fermi kernels
	// BVHCompact2 is used in kepler kernels.
	// We will not run the application if the correct version of the kernel and correct bvh types are loaded.
	//

	void							create_compact(const BVH& bvh, int nodeOffsetSizeDiv);
	
	void							woopify_tri(const BVH& bvh, GIndexType idx);

private:
	BVHLayout						m_layout;
	std::vector<uint4>				m_nodes;
	std::vector<uint4>				m_tri_woop;
	std::vector<GIndexType>			m_tri_index;
	float4							m_woop[3];
};

#endif			// !CUDA_BVH_BUILDER_H
