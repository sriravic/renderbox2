
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

#ifndef BVH_NODE_H
#define BVH_NODE_H

// Application specific headers.
#include <accelerators/sbvh/platform.h>
#include <core/globaltypes.h>
#include <core/primitives.h>

// Cuda specific headers.

// Standard c++ headers.
#include <cassert>

using namespace renderbox2;

enum class BVH_STAT
{
    BVH_STAT_NODE_COUNT,
    BVH_STAT_INNER_COUNT,
    BVH_STAT_LEAF_COUNT,
    BVH_STAT_TRIANGLE_COUNT,
    BVH_STAT_CHILDNODE_COUNT,
};

// Forward Declarations.
class InnerNode;
class LeafNode;

//
// BVHNode class.
//

class BVHNode
{
public:
    
	BVHNode() : m_probability(1.f), m_parent_probability(1.f)
	{
		// Empty constructor.
	}

    virtual bool        is_leaf() const = 0;    
	virtual GIndexType  get_num_child_nodes() const = 0;
	virtual BVHNode*    get_child_node(GIndexType i) const = 0;
	virtual GIndexType  get_num_triangles() const { return 0; }
    float				get_area() const     { return m_bounds.area(); }    

	// Subtree functions.
	void delete_subtree()
	{
		for (GIndexType i = 0; i < get_num_child_nodes(); i++)
			get_child_node(i)->delete_subtree();

		delete this;
	}

	GIndexType get_subtree_size(BVH_STAT stat) const;

	void compute_subtree_probabilities(const Platform& p, float probability, float& sah);

	// Data.
    // These are somewhat experimental, for some specific test and may be invalid.
	AABB     m_bounds;
    float    m_probability;           // probability of coming here (widebvh uses this).
    float    m_parent_probability;    // probability of coming to parent (widebvh uses this).
};


//
// Internal node of the BVH.
//

class InnerNode : public BVHNode
{
public:

    InnerNode(const AABB& bounds,BVHNode* child0,BVHNode* child1) 
	{
		m_bounds=bounds;
		m_children[0]=child0;
		m_children[1]=child1;
	}

    bool        is_leaf() const							{ return false; }
	GIndexType  get_num_child_nodes() const				{ return 2; }
	BVHNode*    get_child_node(GIndexType i) const		{ assert(i < 2); return m_children[i]; }

    BVHNode*    m_children[2];
};


//
// Leaf node of the BVH.
//

class LeafNode : public BVHNode
{
public:

	LeafNode(const AABB& bounds, GIndexType lo, GIndexType hi)	{ m_bounds = bounds; m_lo = lo; m_hi = hi; }
    
	LeafNode(const LeafNode& s)									{ *this = s; }

    bool        is_leaf() const									{ return true; }
	GIndexType  get_num_child_nodes() const						{ return 0; }
    BVHNode*    get_child_node(GIndexType i) const				{ return nullptr; }

	GIndexType  get_num_triangles() const						{ return m_hi - m_lo; }
	GIndexType  m_lo;
	GIndexType  m_hi;
};

#endif			// !BVH_NODE
