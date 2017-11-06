
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

#ifndef __PLATFORM_H__
#define __PLATFORM_H__

// Application specific headers.
#include <core/globaltypes.h>

// Cuda specific headers.

// Standard c++ headers.
#include <cstdint>
#include <string>

using namespace renderbox2;

//
// Forward declarations.
//

class LeafNode;
class BVHNode;


//
// Platform class implementation.
//

class Platform
{
public:
	Platform()
	{
		m_name = std::string("Default");
		m_sah_node_cost = 1.f;
		m_sah_triangle_cost = 1.f;
		m_node_batch_size = 1;
		m_tri_batch_size = 1;
		m_min_leaf_size = 1;
		m_max_leaf_size = 0x7FFFFFF;
	}

	Platform(const std::string& name, float nodeCost = 1.f, float triCost = 1.f, GIndexType nodeBatchSize = 1u, GIndexType triBatchSize = 1u)
	{
		m_name = name;
		m_sah_node_cost = nodeCost;
		m_sah_triangle_cost = triCost;
		m_node_batch_size = nodeBatchSize;
		m_tri_batch_size = triBatchSize;
		m_min_leaf_size = 1u;
		m_max_leaf_size = 0x7FFFFFF;
	}

	const std::string&   getName() const									{ return m_name; }

	// SAH weights
	float get_sah_triangle_cost() const										{ return m_sah_triangle_cost; }
	float get_sah_node_cost() const											{ return m_sah_node_cost; }

	// SAH costs, raw and batched
	float get_cost(GIndexType numChildNodes, GIndexType numTris) const		{ return get_node_cost(numChildNodes) + get_triangle_cost(numTris); }
	float get_triangle_cost(GIndexType n) const								{ return round_to_triangle_batch_size(n) * m_sah_triangle_cost; }
	float get_node_cost(GIndexType n) const									{ return round_to_node_batch_size(n) * m_sah_node_cost; }

	// batch processing (how many ops at the price of one)
	GIndexType  get_triangle_batch_size() const								{ return m_tri_batch_size; }
	GIndexType  get_node_batch_size() const									{ return m_node_batch_size; }
	void        set_triangle_batch_size(int32_t triBatchSize)				{ m_tri_batch_size = triBatchSize; }
	void        set_node_batch_size(int32_t nodeBatchSize)					{ m_node_batch_size = nodeBatchSize; }
	GIndexType  round_to_triangle_batch_size(int32_t n) const				{ return ((n + m_tri_batch_size - 1) / m_tri_batch_size) * m_tri_batch_size; }
	GIndexType  round_to_node_batch_size(int32_t n) const					{ return ((n + m_node_batch_size - 1) / m_node_batch_size) * m_node_batch_size; }

	// leaf preferences
	void        set_leaf_preferences(int32_t minSize, int32_t maxSize)		{ m_min_leaf_size = minSize; m_max_leaf_size = maxSize; }
	GIndexType  get_min_leaf_size() const									{ return m_min_leaf_size; }
	GIndexType  get_max_leaf_size() const									{ return m_max_leaf_size; }

private:
	std::string		m_name;
	float			m_sah_node_cost;
	float			m_sah_triangle_cost;
	GIndexType		m_tri_batch_size;
	GIndexType		m_node_batch_size;
	GIndexType		m_min_leaf_size;
	GIndexType		m_max_leaf_size;
};

#endif			// !PLATFORM_H
