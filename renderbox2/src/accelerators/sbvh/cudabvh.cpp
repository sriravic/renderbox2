
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
// Note: These classes are not enclosed within renderbox2 namespace for want of code clarity.
//

// Application specific headers.
#include <accelerators/sbvh/cudabvh.h>
#include <core/transform.h>

// Cuda specific headers.

// Standard c++ headers.
#include <fstream>
#include <iostream>
#include <stack>


//
// Free functions that are used only in the cudabvhbuilder.
//

inline unsigned int floatToBits(float val) { return *((unsigned int*)&val); }

inline float        bitsToFloat(unsigned int val) { return *((float*)&val); }

template<typename T>
inline size_t getNumBytes(const std::vector<T>& V)
{
	return sizeof(T) * V.size();
}


//
// CudaBvhBuilder class implementation.
//

CudaBvh::CudaBvh(const BVH& bvh, BVHLayout layout)
{
	m_layout = layout;
	if (layout == BVHLayout::BVHLayout_Compact)
	{
		create_compact(bvh, 1);
		return;
	}
	else if (layout == BVHLayout::BVHLayout_Compact2)
	{
		create_compact(bvh, 16);
		return;
	}
	else
	{
		std::cerr << "Unknown BVH layout selected\n." << std::endl;
		return;
	}
}

//
// CudaBVH constructor.
// Loads from a stored file.
//

CudaBvh::CudaBvh(const std::string& filename, bool& isLoadedProperly)
{
	std::ifstream in;
	in.open(filename.c_str(), std::ios::in);
	if (in.bad())
	{
		printf("Invalid filename\n");
	}

	// first read the magic number to see if its our file
	unsigned int magic;
	in >> magic;
	if (magic != 0x76543210) { printf("Invalid cuda bvh file\n"); return; }

	// read the sizes and allocate as much as before
	size_t m_nodes_size;
	size_t m_tris_size;
	size_t m_triidx_size;
	unsigned int layout;
	in >> m_nodes_size;
	in >> m_tris_size;
	in >> m_triidx_size;
	in >> layout;

	m_layout = (BVHLayout)layout;
	m_nodes.resize(m_nodes_size);
	m_tri_woop.resize(m_tris_size);
	m_tri_index.resize(m_triidx_size);

	// copy data as such
	for (size_t i = 0; i < m_nodes_size; i++)
	{
		uint4 data;
		in >> data.x >> data.y >> data.z >> data.w;
		m_nodes[i] = data;
	}

	for (size_t i = 0; i < m_tris_size; i++)
	{
		uint4 data;
		in >> data.x >> data.y >> data.z >> data.w;
		m_tri_woop[i] = data;
	}

	for (size_t i = 0; i < m_triidx_size; i++)
	{
		GIndexType data;
		in >> data;
		m_tri_index[i] = data;
	}

	unsigned int eof;
	in >> eof;
	if (eof == 0x1234567)
		isLoadedProperly = true;
	else
		isLoadedProperly = false;
}

// write the contents of the cuda bvh into a file in binary format
// Format for writing
// #1. Magic Number
// #2. Size of node buffer
// #3. Size of triangle buffer
// #4. Size of index buffer
// #5. layout
// all the buffers in continuous fashion

void CudaBvh::serialize(const std::string& filename)
{
	std::ofstream out;
	try{
		out.open(filename.c_str(), std::ios::binary);

		// write the file structure
		out << ((unsigned int)0x76543210) << "\n";
		out << m_nodes.size() << "\n";
		out << m_tri_woop.size() << "\n";
		out << m_tri_index.size() << "\n";
		out << (unsigned int)m_layout << "\n";

		// write out the rest of the data
		for (size_t i = 0; i < m_nodes.size(); i++)
		{
			// write out x,y,z,w\n format
			out << m_nodes[i].x << " " << m_nodes[i].y << " " << m_nodes[i].z << " " << m_nodes[i].w << "\n";
		}

		for (size_t i = 0; i < m_tri_woop.size(); i++)
		{
			out << m_tri_woop[i].x << " " << m_tri_woop[i].y << " " << m_tri_woop[i].z << " " << m_tri_woop[i].w << "\n";
		}

		for (size_t i = 0; i < m_tri_index.size(); i++)
		{
			out << m_tri_index[i] << "\n";
		}

		out << (unsigned int)0x1234567 << "\n";		// end of file value
		out.close();
	}
	catch (exception e)
	{
		std::cerr << e.what() << std::endl;
		return;
	}
}

void CudaBvh::create_compact(const BVH& bvh, int nodeOffsetSizeDiv)
{

	struct StackEntry
	{
		const BVHNode*  node;
		GIndexType      idx;

		StackEntry(const BVHNode* n = nullptr, int i = 0) : node(n), idx(i) {}
	};

	// Construct data.
	std::vector<uint4> nodeData(4);
	std::vector<uint4> triWoopData;
	std::vector<GIndexType> triIndexData;
	std::stack<StackEntry> stack;
	stack.push(StackEntry(bvh.get_root(), 0));

	while (stack.size())
	{
		StackEntry e = stack.top(); stack.pop();			// get the last element and pop.
		assert(e.node->get_num_child_nodes() == 2);
		const AABB* cbox[2];
		int cidx[2];

		// Process children.

		for (int i = 0; i < 2; i++)
		{
			// Inner node => push to stack.

			const BVHNode* child = e.node->get_child_node(i);
			cbox[i] = &child->m_bounds;
			if (!cbox[i]->valid())
				printf("We have an invalid bounding box at i : %d at stack : %d", i, stack.size());
			if (!child->is_leaf())
			{
				cidx[i] = static_cast<int>(getNumBytes(nodeData)) / nodeOffsetSizeDiv;
				stack.push(StackEntry(child, static_cast<int>(nodeData.size())));
				nodeData.resize(nodeData.size() + 4);
				continue;
			}

			// Leaf => append triangles.
			const LeafNode* leaf = reinterpret_cast<const LeafNode*>(child);
			cidx[i] = ~(static_cast<int>(triWoopData.size()));
			for (GIndexType j = leaf->m_lo; j < leaf->m_hi; j++)
			{
				woopify_tri(bvh, j);
				if (m_woop[0].x == 0.0f)
					m_woop[0].x = 0.0f;
				
				triWoopData.push_back(make_uint4(floatToBits(m_woop[0].x), floatToBits(m_woop[0].y), floatToBits(m_woop[0].z), floatToBits(m_woop[0].w)));
				triWoopData.push_back(make_uint4(floatToBits(m_woop[1].x), floatToBits(m_woop[1].y), floatToBits(m_woop[1].z), floatToBits(m_woop[1].w)));
				triWoopData.push_back(make_uint4(floatToBits(m_woop[2].x), floatToBits(m_woop[2].y), floatToBits(m_woop[2].z), floatToBits(m_woop[2].w)));
				triIndexData.push_back(bvh.get_tri_indices()[j]);
				triIndexData.push_back(0);
				triIndexData.push_back(0);
			}

			// Terminator.
			triWoopData.push_back(make_uint4(0x80000000));
			triIndexData.push_back(0);
		}

		// Write entry.
		uint4* dstPtr = nodeData.data();
		uint4* dst = &(dstPtr[e.idx]);
		dst[0] = make_uint4(floatToBits(cbox[0]->m_min.x), floatToBits(cbox[0]->m_max.x), floatToBits(cbox[0]->m_min.y), floatToBits(cbox[0]->m_max.y));
		dst[1] = make_uint4(floatToBits(cbox[1]->m_min.x), floatToBits(cbox[1]->m_max.x), floatToBits(cbox[1]->m_min.y), floatToBits(cbox[1]->m_max.y));
		dst[2] = make_uint4(floatToBits(cbox[0]->m_min.z), floatToBits(cbox[0]->m_max.z), floatToBits(cbox[1]->m_min.z), floatToBits(cbox[1]->m_max.z));
		dst[3] = make_uint4(cidx[0], cidx[1], 0, 0);
	}

	m_nodes.resize(nodeData.size());
	m_tri_woop.resize(triWoopData.size());
	m_tri_index.resize(triIndexData.size());
	memcpy(m_nodes.data(), nodeData.data(), getNumBytes(nodeData));
	memcpy(m_tri_woop.data(), triWoopData.data(), getNumBytes(triWoopData));
	memcpy(m_tri_index.data(), triIndexData.data(), getNumBytes(triIndexData));
}


//
// Converts the triangle to sven-woop notation and stores internally in the tree for faster and simpler intersection test with ray.
//

void CudaBvh::woopify_tri(const BVH& bvh, GIndexType idx)
{
	const uint3* triVtxIndex = (const uint3*)bvh.get_scene()->get_tri_vtx_indices();
	const float3* vtxPos = (const float3*)bvh.get_scene()->get_tri_vertices();
	const uint3& inds = triVtxIndex[bvh.get_tri_indices()[idx]];
	const float3& v0 = vtxPos[inds.x];
	const float3& v1 = vtxPos[inds.y];
	const float3& v2 = vtxPos[inds.z];

	Matrix4x4 mtx, invMtx;
	mtx.set_col(0, make_float4(v0 - v2, 0.0f));
	mtx.set_col(1, make_float4(v1 - v2, 0.0f));
	mtx.set_col(2, make_float4(normalize(cross(v0 - v2, v1 - v2)), 0.0f));
	mtx.set_col(3, make_float4(v2, 1.0f));
	invMtx = mtx.invert();

	m_woop[0] = make_float4(invMtx(2, 0), invMtx(2, 1), invMtx(2, 2), -invMtx(2, 3));
	m_woop[1] = invMtx.get_row(0);
	m_woop[2] = invMtx.get_row(1);
}
