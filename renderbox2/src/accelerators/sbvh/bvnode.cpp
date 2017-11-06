
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
#include <accelerators/sbvh/bvhnode.h>

// Cuda specific headers.

// Standard c++ headers.


//
// BVHNode function definitions.
//

GIndexType BVHNode::get_subtree_size(BVH_STAT stat) const
{
	GIndexType cnt = 0;
	switch (stat)
	{
	case BVH_STAT::BVH_STAT_NODE_COUNT:      cnt = 1; break;
	case BVH_STAT::BVH_STAT_LEAF_COUNT:      cnt = is_leaf() ? 1 : 0; break;
	case BVH_STAT::BVH_STAT_INNER_COUNT:     cnt = is_leaf() ? 0 : 1; break;
	case BVH_STAT::BVH_STAT_TRIANGLE_COUNT:  cnt = is_leaf() ? reinterpret_cast<const LeafNode*>(this)->get_num_triangles() : 0; break;
	case BVH_STAT::BVH_STAT_CHILDNODE_COUNT: cnt = get_num_child_nodes(); break;
	}

	if (!is_leaf())
	{
		for (GIndexType i = 0; i < get_num_child_nodes(); i++)
			cnt += get_child_node(i)->get_subtree_size(stat);
	}

	return cnt;
}

void BVHNode::compute_subtree_probabilities(const Platform& p, float probability, float& sah)
{
	sah += probability * p.get_cost(this->get_num_child_nodes(), this->get_num_triangles());

	m_probability = probability;

	for (GIndexType i = 0; i < get_num_child_nodes(); i++)
	{
		BVHNode* child = get_child_node(i);
		child->m_parent_probability = probability;
		float childProbability = 0.0f;
		if (probability > 0.0f)
			childProbability = probability * child->m_bounds.area() / this->m_bounds.area();
		child->compute_subtree_probabilities(p, childProbability, sah);
	}
}
