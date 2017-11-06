
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

#ifndef SPLIT_BVH_BUILDER_H
#define SPLIT_BVH_BUILDER_H

// Application specific headers.
#include <accelerators/sbvh/bvh.h>

// Cuda specific headers.

// Standard C++ headers.

//
// Sort functionality that the splitbvhbuilder class utilizes.
// This is legacy code that has been carried forward from the original nvidia code. 
// NOTE: To be cleaned to be much more efficient.
//

//------------------------------------------------------------------------
// Generic quick sort implementation.
// Somewhat cumbersome to use - see below for convenience wrappers.
//
// Sort integers into ascending order:
//
//   static bool myCompareFunc   (void* data, int idxA, int idxB)    { return (((S32*)data)[idxA] < ((S32*)data)[idxB]); }
//   static void mySwapFunc      (void* data, int idxA, int idxB)    { swap(((S32*)data)[idxA], ((S32*)data)[idxB]); }
//
//   Array<S32> myArray = ...;
//   sort(myArray.getPtr(), 0, myArray.getSize(), myCompareFunc, mySwapFunc);
//------------------------------------------------------------------------

typedef bool(*SortCompareFunc) (void* data, int idxA, int idxB);    // Returns true if A should come before B.
typedef void(*SortSwapFunc)    (void* data, int idxA, int idxB);    // Swaps A and B.

void sort(void* data, GIndexType start, GIndexType end, SortCompareFunc compareFunc, SortSwapFunc swapFunc, bool multicore = false);

//------------------------------------------------------------------------
// Template-based wrappers.
// Use these if your type defines operator< and you want to sort in
// ascending order, OR if you want to explicitly define a custom
// comparator function.
//
// Sort integers into ascending order:
//
//   Array<S32> myArray = ...;
//   sort(myArray);                             // sort the full array
//   sort(myArray, 0, myArray.getSize());       // specify range of elements
//   sort(myArray.getPtr(), myArray.getSize()); // sort C array
//
// Sort integers into descending order:
//
//   static bool myCompareFunc(void* data, int idxA, int idxB)
//   {
//       const S32& a = ((S32*)data)[idxA];
//       const S32& b = ((S32*)data)[idxB];
//       return (a > b);
//   }
//
//   Array<S32> myArray = ...;
//   sort(myArray, myCompareFunc);
//------------------------------------------------------------------------

template <class T> bool sortDefaultCompare(void* data, GIndexType idxA, GIndexType idxB);
template <class T> void sortDefaultSwap(void* data, GIndexType idxA, GIndexType idxB);

template <class T> void sort(T* data, GIndexType num, SortCompareFunc compareFunc = sortDefaultCompare<T>, SortSwapFunc swapFunc = sortDefaultSwap<T>, bool multicore = false);

template <class T> bool sortDefaultCompare(void* data, GIndexType idxA, GIndexType idxB)
{
	return (((T*)data)[idxA] < ((T*)data)[idxB]);
}

template <class T> void sortDefaultSwap(void* data, GIndexType idxA, GIndexType idxB)
{
	swap(((T*)data)[idxA], ((T*)data)[idxB]);
}

template <class T> void sort(T* data, GIndexType num, SortCompareFunc compareFunc, SortSwapFunc swapFunc, bool multicore)
{
	sort(data, 0, num, compareFunc, swapFunc, multicore);
}

class SplitBVHBuilder
{
private:
	enum
	{
		MaxDepth = 64,
		MaxSpatialDepth = 48,
		NumSpatialBins = 128,
	};

	struct Reference
	{
		GIndexType          triIdx;
		AABB                bounds;

		Reference(void) {}
	};

	struct NodeSpec
	{
		GIndexType          numRef;
		AABB                bounds;

		NodeSpec(void) : numRef(0) {}
	};

	struct ObjectSplit
	{
		float               sah;
		int                 sortDim;
		GIndexType          numLeft;
		AABB                leftBounds;
		AABB                rightBounds;

		ObjectSplit(void) : sah(FLT_MAX), sortDim(0), numLeft(0) {}
	};

	struct SpatialSplit
	{
		float               sah;
		int                 dim;
		float               pos;

		SpatialSplit(void) : sah(FLT_MAX), dim(0), pos(0.0f) {}
	};

	struct SpatialBin
	{
		AABB                bounds;
		GIndexType          enter;
		GIndexType          exit;
	};

public:
	SplitBVHBuilder(BVH& bvh, const BVH::BuildParams& params);
	~SplitBVHBuilder(void);

	BVHNode*                run(void);

private:
	static bool             sort_compare(void* data, int idxA, int idxB);
	static void             sort_swap(void* data, int idxA, int idxB);

	BVHNode*                build_node(NodeSpec spec, int level, float progressStart, float progressEnd);
	BVHNode*                create_leaf(const NodeSpec& spec);

	ObjectSplit             find_object_split(const NodeSpec& spec, float nodeSAH);
	void                    perform_object_split(NodeSpec& left, NodeSpec& right, const NodeSpec& spec, const ObjectSplit& split);

	SpatialSplit            find_spatial_split(const NodeSpec& spec, float nodeSAH);
	void                    perform_spatial_split(NodeSpec& left, NodeSpec& right, const NodeSpec& spec, const SpatialSplit& split);
	void                    split_reference(Reference& left, Reference& right, const Reference& ref, int dim, float pos);

private:
	SplitBVHBuilder(const SplitBVHBuilder&);								// forbidden
	SplitBVHBuilder&        operator=             (const SplitBVHBuilder&); // forbidden

private:

	// Some util functions added so that they are compatible with the original code provided in the NVIDIA sdk.
	template<typename T>
	T removeSwap(std::vector<T>& V, size_t idx) {
		assert(idx >= 0 && idx < V.size());
		T old = V[idx];
		T back = V.back();
		V.pop_back();			// remove the last element which is basically an size-- operation.
		if (idx < V.size())
			V[idx] = back;
		return old;
	}

	template<typename T>
	T removeLast(std::vector<T>& V) {
		T old = V.back();
		V.pop_back();
		return old;
	}

private:

	BVH&                    m_bvh;
	const Platform&         m_platform;
	const BVH::BuildParams& m_params;

	std::vector<Reference>  m_ref_stack;
	float                   m_min_overlap;
	std::vector<AABB>       m_right_bounds;
	int                     m_sort_dim;
	SpatialBin              m_bins[3][NumSpatialBins];

	GIndexType              m_num_duplicates;
};

#endif			// !SPLIT_BVH_BUILDER_H
