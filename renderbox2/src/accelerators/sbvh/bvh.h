
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

#ifndef BVH_H
#define BVH_H

// Application specific headers.
#include <accelerators/sbvh/bvhnode.h>
#include <accelerators/sbvh/platform.h>
#include <core/globaltypes.h>
#include <core/scene.h>

// Cuda specific headers.

// Standard c++ headers.
#include <algorithm>
#include <cstdint>
#include <vector>

using namespace renderbox2;


//
// BVH class.
//

class BVH
{
public:
	struct Stats
	{
		Stats()             { clear(); }
		void clear()        { memset(this, 0, sizeof(Stats)); }
		void print() const  {
			printf("Tree stats: [bfactor=%d] %d nodes (%d+%d), %.2f SAHCost, %.1f children/inner, %.1f tris/leaf\n",
				m_branching_factor,
				m_num_leaf_nodes + m_num_inner_nodes,
				m_num_leaf_nodes, m_num_inner_nodes,
				m_sah_cost,
				1.f * m_num_child_nodes / std::max(m_num_inner_nodes, 1u),
				1.f * m_num_tris / std::max(m_num_leaf_nodes, 1u));
		}

		float      m_sah_cost;
		GIndexType m_branching_factor;
		GIndexType m_num_inner_nodes;
		GIndexType m_num_leaf_nodes;
		GIndexType m_num_child_nodes;
		GIndexType m_num_tris;
	};

	struct BuildParams
	{
		Stats*      m_stats;
		bool        m_enable_prints;
		float       m_split_alpha;     // spatial split area threshold.

		BuildParams(void)
		{
			m_stats = NULL;
			m_enable_prints = true;
			m_split_alpha = 1.0e-5f;
		}
	};

public:
	BVH(Scene* scene, const Platform& platform, const BuildParams& params);

	~BVH(void) { if (m_root) m_root->delete_subtree(); }

	Scene*								get_scene(void) const       { return m_scene; }
	const Platform&						get_platform(void) const    { return m_platform; }
	BVHNode*							get_root(void) const        { return m_root; }

	std::vector<GIndexType>&			get_tri_indices(void)       { return m_tri_indices; }
	const std::vector<GIndexType>&		get_tri_indices(void) const	{ return m_tri_indices; }

private:
	
	Scene*					    m_scene;
	Platform					m_platform;

	BVHNode*					m_root;
	std::vector<GIndexType>		m_tri_indices;
};

#endif			// !BVH_H
