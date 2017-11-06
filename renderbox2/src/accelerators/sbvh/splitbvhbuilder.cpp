
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
#include <accelerators/sbvh/splitbvhbuilder.h>
#include <core/util.h>

// Cuda specific headers.

// Standard c++ headers.
#include <omp.h>

//
// Sort functions.
//

#define QSORT_STACK_SIZE    32
#define QSORT_MIN_SIZE      16

void insertionSort(GIndexType start, GIndexType size, void* data, SortCompareFunc compareFunc, SortSwapFunc swapFunc)
{
	assert(compareFunc && swapFunc);
	assert(size >= 0);

	for (int32_t i = 1; i < static_cast<int32_t>(size); i++)
	{
		int32_t j = static_cast<int32_t>(start) + i - 1;
		while (j >= static_cast<int32_t>(start) && compareFunc(data, j + 1, j))
		{
			swapFunc(data, j, j + 1);
			j--;
		}
	}
}

int median3(GIndexType low, GIndexType high, void* data, SortCompareFunc compareFunc)
{
	assert(compareFunc);
	assert(low >= 0 && high >= 2);

	GIndexType l = low;
	GIndexType c = (low + high) >> 1;
	GIndexType h = high - 2;

	if (compareFunc(data, h, l)) std::swap(l, h);
	if (compareFunc(data, c, l)) c = l;
	return (compareFunc(data, h, c)) ? h : c;
}

GIndexType partition(GIndexType low, GIndexType high, void* data, SortCompareFunc compareFunc, SortSwapFunc swapFunc)
{
	// Select pivot using median-3, and hide it in the highest entry.
	swapFunc(data, median3(low, high, data, compareFunc), high - 1);

	// Partition data.
	GIndexType i = low - 1;
	GIndexType j = high - 1;
	for (;;)
	{
		do
			i++;
		while (compareFunc(data, i, high - 1));
		do
			j--;
		while (compareFunc(data, high - 1, j));

		assert(i >= low && j >= low && i < high && j < high);
		if (i >= j)
			break;

		swapFunc(data, i, j);
	}

	// Restore pivot.
	swapFunc(data, i, high - 1);
	return i;
}

void qsort(GIndexType low, GIndexType high, void* data, SortCompareFunc compareFunc, SortSwapFunc swapFunc)
{
	assert(compareFunc && swapFunc);
	assert(low <= high);

	int stack[QSORT_STACK_SIZE];
	int sp = 0;
	stack[sp++] = high;

	while (sp)
	{
		high = stack[--sp];
		assert(low <= high);

		// Small enough or stack full => use insertion sort.
		if (high - low < QSORT_MIN_SIZE || sp + 2 > QSORT_STACK_SIZE)
		{
			insertionSort(low, high - low, data, compareFunc, swapFunc);
			low = high + 1;
			continue;
		}

		// Partition and sort sub-partitions.
		GIndexType i = partition(low, high, data, compareFunc, swapFunc);
		assert(sp + 2 <= QSORT_STACK_SIZE);
		if (high - i > 2)
			stack[sp++] = high;
		if (i - low > 1)
			stack[sp++] = i;
		else
			low = i + 1;
	}
}

void sort(void* data, GIndexType start, GIndexType end, SortCompareFunc compareFunc, SortSwapFunc swapFunc, bool multicore)
{
	assert(start <= end);
	assert(compareFunc && swapFunc);

	// Nothing to do => skip.
	if (end - start < 2)
		return;
	// Single-core.
	qsort(start, end, data, compareFunc, swapFunc);
}

//
// BVH class constructor.
//

BVH::BVH(Scene* scene, const Platform& platform, const BuildParams& params)
{

	double start, end;
	float time = 0.0f;
	assert(scene);
	m_scene = scene;
	m_platform = platform;

	if (params.m_enable_prints)
		printf("BVH builder: %d tris\n", scene->get_num_triangles());

	start = omp_get_wtime();
	m_root = SplitBVHBuilder(*this, params).run();
	end = omp_get_wtime();
	time = (float)(end - start) * 1000.0f;
	printf("\n\nSplitBVH built in : %f ms\n", time);

	AABB scene_bound = scene->get_scene_aabb();
	if (params.m_enable_prints)
		printf("BVH: Scene bounds: (%f,%f,%f) - (%f,%f,%f)\n", scene_bound.m_min.x, scene_bound.m_min.y, scene_bound.m_min.z,
		scene_bound.m_max.x, scene_bound.m_max.y, scene_bound.m_max.z);

	float sah = 0.f;
	m_root->compute_subtree_probabilities(m_platform, 1.f, sah);
	if (params.m_enable_prints)
		printf("top-down sah: %.2f\n", sah);

	if (params.m_stats)
	{
		params.m_stats->m_sah_cost = sah;
		params.m_stats->m_branching_factor = 2;
		params.m_stats->m_num_leaf_nodes = m_root->get_subtree_size(BVH_STAT::BVH_STAT_LEAF_COUNT);
		params.m_stats->m_num_inner_nodes = m_root->get_subtree_size(BVH_STAT::BVH_STAT_INNER_COUNT);
		params.m_stats->m_num_tris = m_root->get_subtree_size(BVH_STAT::BVH_STAT_TRIANGLE_COUNT);
		params.m_stats->m_num_child_nodes = m_root->get_subtree_size(BVH_STAT::BVH_STAT_CHILDNODE_COUNT);
	}
}

//
// SplitBVHBuilder class implementation and methods.
// 

SplitBVHBuilder::SplitBVHBuilder(BVH& bvh, const BVH::BuildParams& params)
	: m_bvh(bvh),
	m_platform(bvh.get_platform()),
	m_params(params),
	m_min_overlap(0.0f),
	m_sort_dim(-1)
{
}

SplitBVHBuilder::~SplitBVHBuilder(void)
{
	// Empty destructor.
}

BVHNode* SplitBVHBuilder::run(void)
{
	// Initialize reference stack and determine root bounds.
	const GIndexVec3Type* tris = (const uint3*)m_bvh.get_scene()->get_tri_vtx_indices();
	const float3* verts = (const float3*)m_bvh.get_scene()->get_tri_vertices();

	NodeSpec rootSpec;
	rootSpec.numRef = m_bvh.get_scene()->get_num_triangles();
	m_ref_stack.resize(rootSpec.numRef);

	for (GIndexType i = 0; i < rootSpec.numRef; i++)
	{
		m_ref_stack[i].triIdx = i;
		m_ref_stack[i].bounds.insert(verts[tris[i].x]);
		m_ref_stack[i].bounds.insert(verts[tris[i].y]);
		m_ref_stack[i].bounds.insert(verts[tris[i].z]);
		assert(m_ref_stack[i].bounds.valid());
		rootSpec.bounds.grow(m_ref_stack[i].bounds);
	}

	// we can do a sanity check to see if the rootspec bounds is the same is the scene bounds.
	printf("Root Spec : [%f,%f,%f], [%f,%f,%f]\n", rootSpec.bounds.m_min.x, rootSpec.bounds.m_min.y, rootSpec.bounds.m_min.z,
												   rootSpec.bounds.m_max.x, rootSpec.bounds.m_max.y, rootSpec.bounds.m_max.z);

	// Initialize rest of the members.
	m_min_overlap = rootSpec.bounds.area() * m_params.m_split_alpha;
	m_right_bounds.resize(max(rootSpec.numRef, (int)NumSpatialBins) - 1);
	m_num_duplicates = 0;

	// Build recursively.
	BVHNode* root = build_node(rootSpec, 0, 0.0f, 1.0f);

	// Done.
	if (m_params.m_enable_prints)
		printf("SplitBVHBuilder: progress %.0f%%, duplicates %.0f%%\n",
		100.0f, (float)m_num_duplicates / (float)m_bvh.get_scene()->get_num_triangles() * 100.0f);
	return root;
}

bool SplitBVHBuilder::sort_compare(void* data, int idxA, int idxB)
{
	const SplitBVHBuilder* ptr = (const SplitBVHBuilder*)data;
	int dim = ptr->m_sort_dim;
	const Reference& ra = ptr->m_ref_stack[idxA];
	const Reference& rb = ptr->m_ref_stack[idxB];
	float ca = renderbox2::get(ra.bounds.m_min, dim) + renderbox2::get(ra.bounds.m_max, dim);
	float cb = renderbox2::get(rb.bounds.m_min, dim) + renderbox2::get(rb.bounds.m_max, dim);
	return (ca < cb || (ca == cb && ra.triIdx < rb.triIdx));
}

void SplitBVHBuilder::sort_swap(void* data, int idxA, int idxB)
{
	SplitBVHBuilder* ptr = (SplitBVHBuilder*)data;
	std::swap(ptr->m_ref_stack[idxA], ptr->m_ref_stack[idxB]);
}

BVHNode* SplitBVHBuilder::build_node(NodeSpec spec, int level, float progressStart, float progressEnd)
{
	// Display progress.

	if (m_params.m_enable_prints)
	{
		printf("SplitBVHBuilder: progress %.0f%%, duplicates %.0f%%\r",
			progressStart * 100.0f, (float)m_num_duplicates / (float)m_bvh.get_scene()->get_num_triangles() * 100.0f);
	}

	// Remove degenerates.
	{
		int firstRef = static_cast<int>(m_ref_stack.size()) - spec.numRef;
		for (int i = static_cast<int>(m_ref_stack.size()) - 1; i >= firstRef; i--)
		{
			float3 size = m_ref_stack[i].bounds.m_max - m_ref_stack[i].bounds.m_min;
			if (renderbox2::get_min(size) < 0.0f || renderbox2::get_sum(size) == renderbox2::get_max(size))
			{
#ifdef _DEBUG
				printf("Degenerate triangle : %d\n", i);
#endif
				removeSwap(m_ref_stack, i);
			}
			//m_ref_stack.removeSwap(i);
		}
		spec.numRef = static_cast<int>(m_ref_stack.size()) - firstRef;
	}

	// Small enough or too deep => create leaf.
	if (spec.numRef <= m_platform.get_min_leaf_size() || level >= MaxDepth)
		return create_leaf(spec);

	// Find split candidates.
	float area = spec.bounds.area();
	float leafSAH = area * m_platform.get_triangle_cost(spec.numRef);
	float nodeSAH = area * m_platform.get_node_cost(2);
	ObjectSplit object = find_object_split(spec, nodeSAH);


	SpatialSplit spatial;
	if (level < MaxSpatialDepth)
	{
		AABB overlap = object.leftBounds;
		overlap.intersect(object.rightBounds);
		if (overlap.area() >= m_min_overlap)
			spatial = find_spatial_split(spec, nodeSAH);
	}

	// Leaf SAH is the lowest => create leaf.
	float minSAH = std::min(leafSAH, std::min(object.sah, spatial.sah));
	//float minSAH = std::min(leafSAH, object.sah);
	if (minSAH == leafSAH && spec.numRef <= m_platform.get_max_leaf_size())
		return create_leaf(spec);

	// Perform split.
	NodeSpec left, right;
	if (minSAH == spatial.sah)
		perform_spatial_split(left, right, spec, spatial);
	if (!left.numRef || !right.numRef)
		perform_object_split(left, right, spec, object);

	// Create inner node.
	m_num_duplicates += left.numRef + right.numRef - spec.numRef;
	float progressMid = lerp(progressStart, progressEnd, (float)right.numRef / (float)(left.numRef + right.numRef));
	BVHNode* rightNode = build_node(right, level + 1, progressStart, progressMid);
	BVHNode* leftNode = build_node(left, level + 1, progressMid, progressEnd);
	return new InnerNode(spec.bounds, leftNode, rightNode);
}

BVHNode* SplitBVHBuilder::create_leaf(const NodeSpec& spec)
{
	std::vector<GIndexType>& tris = m_bvh.get_tri_indices();
	for (GIndexType i = 0; i < spec.numRef; i++) {
		Reference last = removeLast(m_ref_stack);
		tris.push_back(last.triIdx);
	}
	return new LeafNode(spec.bounds, static_cast<int>(tris.size())- spec.numRef, static_cast<int>(tris.size()));
}

SplitBVHBuilder::ObjectSplit SplitBVHBuilder::find_object_split(const NodeSpec& spec, float nodeSAH)
{
	ObjectSplit split;

	const Reference* refPtr = (const Reference*)&(m_ref_stack[m_ref_stack.size() - spec.numRef]);
	float bestTieBreak = FLT_MAX;

	// Sort along each dimension.
	for (m_sort_dim = 0; m_sort_dim < 3; m_sort_dim++)
	{
		sort(this, static_cast<int>(m_ref_stack.size()) - spec.numRef, static_cast<int>(m_ref_stack.size()), sort_compare, sort_swap);

		// Sweep right to left and determine bounds.
		AABB rightBounds;
		for (GIndexType i = spec.numRef - 1; i > 0; i--)
		{
			rightBounds.grow(refPtr[i].bounds);
			m_right_bounds[i - 1] = rightBounds;
		}

		// Sweep left to right and select lowest SAH.
		AABB leftBounds;
		for (GIndexType i = 1; i < spec.numRef; i++)
		{
			leftBounds.grow(refPtr[i - 1].bounds);
			float sah = nodeSAH + leftBounds.area() * m_platform.get_triangle_cost(i) + m_right_bounds[i - 1].area() * m_platform.get_triangle_cost(spec.numRef - i);
			float tieBreak = renderbox2::sqr((float)i) + renderbox2::sqr((float)(spec.numRef - i));
			if (sah < split.sah || (sah == split.sah && tieBreak < bestTieBreak))
			{
				split.sah = sah;
				split.sortDim = m_sort_dim;
				split.numLeft = i;
				split.leftBounds = leftBounds;
				split.rightBounds = m_right_bounds[i - 1];
				bestTieBreak = tieBreak;
			}
		}
	}
	return split;
}
void SplitBVHBuilder::perform_object_split(NodeSpec& left, NodeSpec& right, const NodeSpec& spec, const ObjectSplit& split)
{
	m_sort_dim = split.sortDim;
	sort(this, static_cast<int>(m_ref_stack.size()) - spec.numRef, static_cast<int>(m_ref_stack.size()), sort_compare, sort_swap);

	left.numRef = split.numLeft;
	left.bounds = split.leftBounds;
	right.numRef = spec.numRef - split.numLeft;
	right.bounds = split.rightBounds;
}

SplitBVHBuilder::SpatialSplit SplitBVHBuilder::find_spatial_split(const NodeSpec& spec, float nodeSAH)
{
	// Initialize bins.
	float3 origin = spec.bounds.m_min;
	float3 binSize = (spec.bounds.m_max - origin) * (1.0f / (float)NumSpatialBins);
	float3 invBinSize = 1.0f / binSize;

	for (int dim = 0; dim < 3; dim++)
	{
		for (int i = 0; i < NumSpatialBins; i++)
		{
			SpatialBin& bin = m_bins[dim][i];
			bin.bounds = AABB();
			bin.enter = 0;
			bin.exit = 0;
		}
	}

	// Chop references into bins.
	for (size_t refIdx = m_ref_stack.size() - spec.numRef; refIdx < m_ref_stack.size(); refIdx++)
	{
		const Reference& ref = m_ref_stack[refIdx];

		float3 refmin = (ref.bounds.m_min - origin) * invBinSize;
		float3 refmax = (ref.bounds.m_max - origin) * invBinSize;
		int3 firstBin = clamp(make_int3((int)refmin.x, (int)refmin.y, (int)refmin.z), 0, NumSpatialBins - 1);
		int3 lastBin = clamp(make_int3((int)refmax.x, (int)refmax.y, (int)refmax.z), firstBin, make_int3(NumSpatialBins - 1));

		for (int dim = 0; dim < 3; dim++)
		{
			Reference currRef = ref;
			for (int i = renderbox2::get(firstBin, dim); i < renderbox2::get(lastBin, dim); i++)
			{
				Reference leftRef, rightRef;
				split_reference(leftRef, rightRef, currRef, dim, renderbox2::get(origin, dim) + renderbox2::get(binSize, dim) * (float)(i + 1));
				m_bins[dim][i].bounds.grow(leftRef.bounds);
				currRef = rightRef;
			}
			m_bins[dim][renderbox2::get(lastBin, dim)].bounds.grow(currRef.bounds);
			m_bins[dim][renderbox2::get(firstBin, dim)].enter++;
			m_bins[dim][renderbox2::get(lastBin, dim)].exit++;
		}
	}

	// Select best split plane.
	SpatialSplit split;
	for (int dim = 0; dim < 3; dim++)
	{
		// Sweep right to left and determine bounds.
		AABB rightBounds;
		for (int i = NumSpatialBins - 1; i > 0; i--)
		{
			rightBounds.grow(m_bins[dim][i].bounds);
			m_right_bounds[i - 1] = rightBounds;
		}

		// Sweep left to right and select lowest SAH.
		AABB leftBounds;
		GIndexType leftNum = 0;
		GIndexType rightNum = spec.numRef;

		for (int i = 1; i < NumSpatialBins; i++)
		{
			leftBounds.grow(m_bins[dim][i - 1].bounds);
			leftNum += m_bins[dim][i - 1].enter;
			rightNum -= m_bins[dim][i - 1].exit;

			float sah = nodeSAH + leftBounds.area() * m_platform.get_triangle_cost(leftNum) + m_right_bounds[i - 1].area() * m_platform.get_triangle_cost(rightNum);
			if (sah < split.sah)
			{
				split.sah = sah;
				split.dim = dim;
				split.pos = renderbox2::get(origin, dim) + renderbox2::get(binSize, dim) * (float)i;
			}
		}
	}
	return split;
}

void SplitBVHBuilder::perform_spatial_split(NodeSpec& left, NodeSpec& right, const NodeSpec& spec, const SpatialSplit& split)
{
	// Categorize references and compute bounds.
	//
	// Left-hand side:      [leftStart, leftEnd[
	// Uncategorized/split: [leftEnd, rightStart[
	// Right-hand side:     [rightStart, refs.getSize()[

	std::vector<Reference>& refs = m_ref_stack;
	GIndexType leftStart = static_cast<GIndexType>(refs.size()) - spec.numRef;
	GIndexType leftEnd = leftStart;
	GIndexType rightStart = static_cast<GIndexType>(refs.size());
	left.bounds = right.bounds = AABB();

	for (GIndexType i = leftEnd; i < rightStart; i++)
	{
		// Entirely on the left-hand side?
		if (renderbox2::get(refs[i].bounds.m_max, split.dim) <= split.pos)
		{
			left.bounds.grow(refs[i].bounds);
			std::swap(refs[i], refs[leftEnd++]);
		}

		// Entirely on the right-hand side?
		else if (renderbox2::get(refs[i].bounds.m_min, split.dim) >= split.pos)
		{
			right.bounds.grow(refs[i].bounds);
			std::swap(refs[i--], refs[--rightStart]);
		}
	}

	// Duplicate or unsplit references intersecting both sides.
	while (leftEnd < rightStart)
	{
		// Split reference.
		Reference lref, rref;
		split_reference(lref, rref, refs[leftEnd], split.dim, split.pos);

		// Compute SAH for duplicate/unsplit candidates.
		AABB lub = left.bounds;  // Unsplit to left:     new left-hand bounds.
		AABB rub = right.bounds; // Unsplit to right:    new right-hand bounds.
		AABB ldb = left.bounds;  // Duplicate:           new left-hand bounds.
		AABB rdb = right.bounds; // Duplicate:           new right-hand bounds.
		lub.grow(refs[leftEnd].bounds);
		rub.grow(refs[leftEnd].bounds);
		ldb.grow(lref.bounds);
		rdb.grow(rref.bounds);

		float lac = m_platform.get_triangle_cost(leftEnd - leftStart);
		float rac = m_platform.get_triangle_cost(static_cast<int>(refs.size()) - rightStart);
		float lbc = m_platform.get_triangle_cost(leftEnd - leftStart + 1);
		float rbc = m_platform.get_triangle_cost(static_cast<int>(refs.size()) - rightStart + 1);

		float unsplitLeftSAH = lub.area() * lbc + right.bounds.area() * rac;
		float unsplitRightSAH = left.bounds.area() * lac + rub.area() * rbc;
		float duplicateSAH = ldb.area() * lbc + rdb.area() * rbc;
		float minSAH = std::min(unsplitLeftSAH, std::min(unsplitRightSAH, duplicateSAH));

		// Unsplit to left?
		if (minSAH == unsplitLeftSAH)
		{
			left.bounds = lub;
			leftEnd++;
		}

		// Unsplit to right?
		else if (minSAH == unsplitRightSAH)
		{
			right.bounds = rub;
			std::swap(refs[leftEnd], refs[--rightStart]);
		}

		// Duplicate?
		else
		{
			left.bounds = ldb;
			right.bounds = rdb;
			refs[leftEnd++] = lref;
			//refs.add(rref);
			refs.push_back(rref);
		}
	}

	left.numRef = leftEnd - leftStart;
	right.numRef = static_cast<int>(refs.size()) - rightStart;
}

void SplitBVHBuilder::split_reference(Reference& left, Reference& right, const Reference& ref, int dim, float pos)
{
	// Initialize references.
	left.triIdx = right.triIdx = ref.triIdx;
	left.bounds = right.bounds = AABB();

	// Loop over vertices/edges.
	const GIndexVec3Type* tris = (const uint3*)m_bvh.get_scene()->get_tri_vtx_indices();
	const float3* verts = (const float3*)m_bvh.get_scene()->get_tri_vertices();
	const GIndexVec3Type& inds = tris[ref.triIdx];
	float3 v1 = verts[inds.z];

	for (uint i = 0; i < 3; i++)
	{
		float3 v0 = v1;
				
		int id = renderbox2::get(inds, i);
		v1 = verts[id];
		float v0p = renderbox2::get(v0, dim);
		float v1p = renderbox2::get(v1, dim);

		// Insert vertex to the boxes it belongs to.
		if (v0p <= pos)
			left.bounds.insert(v0);
		if (v0p >= pos)
			right.bounds.insert(v0);

		// Edge intersects the plane => insert intersection to both boxes.
		if ((v0p < pos && v1p > pos) || (v0p > pos && v1p < pos))
		{
			float3 t = lerp(v0, v1, clamp((pos - v0p) / (v1p - v0p), 0.0f, 1.0f));
			left.bounds.insert(t);
			right.bounds.insert(t);
		}
	}

	// Intersect with original bounds.
	renderbox2::get(left.bounds.m_max, dim) = pos;
	renderbox2::get(right.bounds.m_min, dim) = pos;
	left.bounds.intersect(ref.bounds);
	right.bounds.intersect(ref.bounds);
}