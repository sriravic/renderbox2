
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

#ifndef CUDA_TRACER_KERNELS_H
#define CUDA_TRACER_KERNELS_H

// Application specific headers.
#include <core/buffer.h>
#include <core/defs.h>
#include <core/globaltypes.h>
#include <core/intersection.h>
#include <core/primitives.h>

// Cuda specific headers.
#include <cuda.h>
#include <cuda_runtime.h>

// Standard c++ headers.
#include <cstdint>

using namespace renderbox2;

//
// Constants.
//

enum
{
	MaxBlockHeight = 6,                // Upper bound for blockDim.y.
	EntrypointSentinel = 0x76543210,   // Bottom-most stack entry, indicating the end of traversal.
};

//
// BVH memory layout.
//

enum class BVHLayout
{
	BVHLayout_Compact,                  // Variant of BVHLayout_AOS_AOS with implicit leaf nodes.
	BVHLayout_Compact2,                 // Variant of BVHLayout_AOS_AOS with implicit leaf nodes.
	BVHLayout_Max
};

//
// Trace kernel for Fermi architecture used in streamed traversal.
//

#define KERNEL_FERMI_TRACE \
    extern "C" __global__ void kernel_fermi_trace( \
        renderbox2::GIndexType		numRays,        /* Total number of rays in the batch. */ \
        bool						anyHit,         /* False if rays need to find the closest hit. */ \
        float4*						rays,           /* Ray input: float3 origin, float tmin, float3 direction, float tmax. */ \
        int4*						results,        /* Ray output: int triangleID, float hitT, int2 padding. */ \
        float4*						nodes,          /* SOA: bytes 0-15 of each node, AOS/Compact: 64 bytes per node. */ \
        float4*						tris,           /* SOA: bytes 0-15 of each triangle, AOS: 64 bytes per triangle, Compact: 48 bytes per triangle. */ \
        renderbox2::GIndexType*		triIndices,		/* Triangle index remapping table. */ \
		uchar1*						masks,			/* Mask array to mask out certain rays*/ \
		bool						use_mask)		/* Use a mask? */


//
// Trace kernel for Kepler architecture used in streamed traversal.
//

#define KERNEL_KEPLER_TRACE \
	extern "C" __global__ void kernel_kepler_trace( \
		renderbox2::GIndexType		numRays,		\
		bool						anyHit,			\
		float4*						rays,			\
		IntersectionBuffer			intersections,	\
		cudaTextureObject_t			nodes,			\
		cudaTextureObject_t			tris,			\
		cudaTextureObject_t			triIndices,		\
		SceneBuffer				    scene_data,		\
		uint8_t*					masks,			\
		bool						use_mask)



//
// Single threaded traversal of the hierarchy
//

#define TRACE_FUNC_KEPLER					\
	extern "C" __device__ bool trace_ray(	\
		const Ray&				ray,		\
		bool					anyhit,		\
		Intersection*			isect,		\
		cudaTextureObject_t		nodes,		\
		cudaTextureObject_t		tris,		\
		cudaTextureObject_t		triIndices,	\
		SceneBuffer				scene_data	\
		)


//
// Temporary data stored in shared memory to reduce register pressure.
//

struct RayStruct
{
	float   idirx;  // 1.0f / ray.direction.x
	float   idiry;  // 1.0f / ray.direction.y
	float   idirz;  // 1.0f / ray.direction.z
	float   tmin;   // ray.tmin
	float   dummy;  // Padding to avoid bank conflicts.
};


//
// Globals.
//

#ifdef __CUDACC__
extern "C"
{
	KERNEL_FERMI_TRACE;					// Stream traversal - Fermi architecture.
	KERNEL_KEPLER_TRACE;				// Streamed traversal - Kepler architecture.
	TRACE_FUNC_KEPLER;                 // Single threaded traversal.	
}
#endif




//
// Utilities.
//

#define FETCH_GLOBAL(NAME, IDX, TYPE) ((const TYPE*)NAME)[IDX]
#define FETCH_TEXTURE(NAME, IDX, TYPE) tex1Dfetch(t_ ## NAME, IDX)
#define STORE_RESULT(RAY, TRI, T) ((int2*)results)[(RAY) * 2] = make_int2(TRI, __float_as_int(T))


#ifdef __CUDACC__

template <class T> __device__ __inline__ void tswap(T& a, T& b)
{
	T t = a;
	a = b;
	b = t;
}

__device__ __inline__ float min4(float a, float b, float c, float d)
{
	return fminf(fminf(fminf(a, b), c), d);
}

__device__ __inline__ float max4(float a, float b, float c, float d)
{
	return fmaxf(fmaxf(fmaxf(a, b), c), d);
}

__device__ __inline__ float min3(float a, float b, float c)
{
	return fminf(fminf(a, b), c);
}

__device__ __inline__ float max3(float a, float b, float c)
{
	return fmaxf(fmaxf(a, b), c);
}

// Using integer min,max
__inline__ __device__ float fminf2(float a, float b)
{
	int a2 = __float_as_int(a);
	int b2 = __float_as_int(b);
	return __int_as_float(a2<b2 ? a2 : b2);
}

__inline__ __device__ float fmaxf2(float a, float b)
{
	int a2 = __float_as_int(a);
	int b2 = __float_as_int(b);
	return __int_as_float(a2>b2 ? a2 : b2);
}

// Using video instructions
__device__ __inline__ int   min_min(int a, int b, int c) { int v; asm("vmin.s32.s32.s32.min %0, %1, %2, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(c)); return v; }
__device__ __inline__ int   min_max(int a, int b, int c) { int v; asm("vmin.s32.s32.s32.max %0, %1, %2, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(c)); return v; }
__device__ __inline__ int   max_min(int a, int b, int c) { int v; asm("vmax.s32.s32.s32.min %0, %1, %2, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(c)); return v; }
__device__ __inline__ int   max_max(int a, int b, int c) { int v; asm("vmax.s32.s32.s32.max %0, %1, %2, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(c)); return v; }
__device__ __inline__ float fmin_fmin(float a, float b, float c) { return __int_as_float(min_min(__float_as_int(a), __float_as_int(b), __float_as_int(c))); }
__device__ __inline__ float fmin_fmax(float a, float b, float c) { return __int_as_float(min_max(__float_as_int(a), __float_as_int(b), __float_as_int(c))); }
__device__ __inline__ float fmax_fmin(float a, float b, float c) { return __int_as_float(max_min(__float_as_int(a), __float_as_int(b), __float_as_int(c))); }
__device__ __inline__ float fmax_fmax(float a, float b, float c) { return __int_as_float(max_max(__float_as_int(a), __float_as_int(b), __float_as_int(c))); }


__device__ __inline__ float magic_max7(float a0, float a1, float b0, float b1, float c0, float c1, float d)
{
	float t1 = fmin_fmax(a0, a1, d);
	float t2 = fmin_fmax(b0, b1, t1);
	float t3 = fmin_fmax(c0, c1, t2);
	return t3;
}

__device__ __inline__ float magic_min7(float a0, float a1, float b0, float b1, float c0, float c1, float d)
{
	float t1 = fmax_fmin(a0, a1, d);
	float t2 = fmax_fmin(b0, b1, t1);
	float t3 = fmax_fmin(c0, c1, t2);
	return t3;
}

// Experimentally determined best mix of float/int/video minmax instructions for Kepler.
__device__ __inline__ float spanBeginKepler(float a0, float a1, float b0, float b1, float c0, float c1, float d){ return fmax_fmax(fminf(a0, a1), fminf(b0, b1), fmin_fmax(c0, c1, d)); }
__device__ __inline__ float spanEndKepler(float a0, float a1, float b0, float b1, float c0, float c1, float d)	{ return fmin_fmin(fmaxf(a0, a1), fmaxf(b0, b1), fmax_fmin(c0, c1, d)); }

// Same for Fermi.
__device__ __inline__ float spanBeginFermi(float a0, float a1, float b0, float b1, float c0, float c1, float d) { return magic_max7(a0, a1, b0, b1, c0, c1, d); }
__device__ __inline__ float spanEndFermi(float a0, float a1, float b0, float b1, float c0, float c1, float d)	{ return magic_min7(a0, a1, b0, b1, c0, c1, d); }

#endif

#endif		// !CUDA_TRACER_KERNELS_H
