
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
// Note: This kernel has been modified to better suit the architecture of renderbox2
//

#ifndef KEPLER_DYNAMIC_FETCH
#define KEPLER_DYNAMIC_FETCH

// Application specific headers.
#include <accelerators/sbvh/cudatracerkernels.h>

// Cuda specific headers.
#include <cuda_runtime.h>

// Standard c++ headers.

#define STACK_SIZE              64          // Size of the traversal stack in local memory.
#define DYNAMIC_FETCH_THRESHOLD 20          // If fewer than this active, fetch new rays

__device__ int g_warpCounter;    // Work counter for persistent threads.

KERNEL_KEPLER_TRACE
{
	// Traversal stack in CUDA thread-local memory.

	int traversalStack[STACK_SIZE];
	traversalStack[0] = EntrypointSentinel; // Bottom-most entry.

	// Live state during traversal, stored in registers.
	float   origx, origy, origz;            // Ray origin.
	char*   stackPtr;                       // Current position in traversal stack.
	int     leafAddr;                       // First postponed leaf, non-negative if none.
	int     nodeAddr = EntrypointSentinel;  // Non-negative: current internal node, negative: second postponed leaf.
	int     hitIndex;                       // Triangle index of the closest intersection, -1 if none.
	float   hitT;                           // t-value of the closest intersection.
	float   tmin;
	int     rayidx;
	float   oodx;
	float   oody;
	float   oodz;
	float   dirx;
	float   diry;
	float   dirz;
	float   idirx;
	float   idiry;
	float   idirz;

	// Initialize persistent threads.

	__shared__ volatile int nextRayArray[MaxBlockHeight]; // Current ray index in global buffer.

	// Persistent threads: fetch and process rays in a loop.

	do
	{
		const int tidx = threadIdx.x;
		volatile int& rayBase = nextRayArray[threadIdx.y];

		// Fetch new rays from the global pool using lane 0.

		const bool          terminated = nodeAddr == EntrypointSentinel;
		const unsigned int  maskTerminated = __ballot(terminated);
		const int           numTerminated = __popc(maskTerminated);
		const int           idxTerminated = __popc(maskTerminated & ((1u << tidx) - 1));
		float4 o, d;
		float u, v;

		if (terminated)
		{
			if (idxTerminated == 0)
				rayBase = atomicAdd(&g_warpCounter, numTerminated);

			rayidx = rayBase + idxTerminated;
			if (rayidx >= numRays)
				break;

			// Check against mask.
			if (use_mask == true && masks[rayidx] == 0)
			{
				break;
			}

			// Fetch ray
			o = FETCH_GLOBAL(rays, rayidx * 2 + 0, float4);
			d = FETCH_GLOBAL(rays, rayidx * 2 + 1, float4);			
			origx = o.x;
			origy = o.y;
			origz = o.z;
			tmin = o.w;
			dirx = d.x;
			diry = d.y;
			dirz = d.z;
			hitT = d.w;
			float ooeps = exp2f(-80.0f); // Avoid div by zero.
			idirx = 1.0f / (fabsf(d.x) > ooeps ? d.x : copysignf(ooeps, d.x));
			idiry = 1.0f / (fabsf(d.y) > ooeps ? d.y : copysignf(ooeps, d.y));
			idirz = 1.0f / (fabsf(d.z) > ooeps ? d.z : copysignf(ooeps, d.z));
			oodx = origx * idirx;
			oody = origy * idiry;
			oodz = origz * idirz;

			// Setup traversal.

			stackPtr = (char*)&traversalStack[0];
			leafAddr = 0;   // No postponed leaf.
			nodeAddr = 0;   // Start from the root.
			hitIndex = -1;  // No triangle intersected so far.
		}

		// Traversal loop.

		while (nodeAddr != EntrypointSentinel)
		{
			// Traverse internal nodes until all SIMD lanes have found a leaf.

			while (unsigned int(nodeAddr) < unsigned int(EntrypointSentinel))   // functionally equivalent, but faster
			{
				// Fetch AABBs of the two child nodes.

				const float4 n0xy = tex1Dfetch<float4>(nodes, nodeAddr + 0); // (c0.lo.x, c0.hi.x, c0.lo.y, c0.hi.y)
				const float4 n1xy = tex1Dfetch<float4>(nodes, nodeAddr + 1); // (c1.lo.x, c1.hi.x, c1.lo.y, c1.hi.y)
				const float4 nz = tex1Dfetch<float4>(nodes, nodeAddr + 2); // (c0.lo.z, c0.hi.z, c1.lo.z, c1.hi.z)
				float4 tmp = tex1Dfetch<float4>(nodes, nodeAddr + 3); // child_index0, child_index1
				int2  cnodes = *(int2*)&tmp;

				// Intersect the ray against the child nodes.

				const float c0lox = n0xy.x * idirx - oodx;
				const float c0hix = n0xy.y * idirx - oodx;
				const float c0loy = n0xy.z * idiry - oody;
				const float c0hiy = n0xy.w * idiry - oody;
				const float c0loz = nz.x   * idirz - oodz;
				const float c0hiz = nz.y   * idirz - oodz;
				const float c1loz = nz.z   * idirz - oodz;
				const float c1hiz = nz.w   * idirz - oodz;
				const float c0min = spanBeginKepler(c0lox, c0hix, c0loy, c0hiy, c0loz, c0hiz, tmin);
				const float c0max = spanEndKepler(c0lox, c0hix, c0loy, c0hiy, c0loz, c0hiz, hitT);
				const float c1lox = n1xy.x * idirx - oodx;
				const float c1hix = n1xy.y * idirx - oodx;
				const float c1loy = n1xy.z * idiry - oody;
				const float c1hiy = n1xy.w * idiry - oody;
				const float c1min = spanBeginKepler(c1lox, c1hix, c1loy, c1hiy, c1loz, c1hiz, tmin);
				const float c1max = spanEndKepler(c1lox, c1hix, c1loy, c1hiy, c1loz, c1hiz, hitT);

				bool swp = (c1min < c0min);

				bool traverseChild0 = (c0max >= c0min);
				bool traverseChild1 = (c1max >= c1min);

				// Neither child was intersected => pop stack.

				if (!traverseChild0 && !traverseChild1)
				{
					nodeAddr = *(int*)stackPtr;
					stackPtr -= 4;
				}

				// Otherwise => fetch child pointers.

				else
				{
					nodeAddr = (traverseChild0) ? cnodes.x : cnodes.y;

					// Both children were intersected => push the farther one.

					if (traverseChild0 && traverseChild1)
					{
						if (swp)
							tswap(nodeAddr, cnodes.y);
						stackPtr += 4;
						*(int*)stackPtr = cnodes.y;
					}
				}

				// First leaf => postpone and continue traversal.

				if (nodeAddr < 0 && leafAddr >= 0)     // Postpone max 1					
				{
					leafAddr = nodeAddr;
					nodeAddr = *(int*)stackPtr;
					stackPtr -= 4;
				}

				// All SIMD lanes have found a leaf? => process them.

				// NOTE: inline PTX implementation of "if(!__any(leafAddr >= 0)) break;".
				// tried everything with CUDA 4.2 but always got several redundant instructions.

				unsigned int mask;
				asm("{\n"
					"   .reg .pred p;               \n"
					"setp.ge.s32        p, %1, 0;   \n"
					"vote.ballot.b32    %0,p;       \n"
					"}"
					: "=r"(mask)
					: "r"(leafAddr));
				if (!mask)
					break;				
			}

			// Process postponed leaf nodes.
			while (leafAddr < 0)
			{
				for (int triAddr = ~leafAddr;; triAddr += 3)
				{
					// Tris in TEX (good to fetch as a single batch)
					const float4 v00 = tex1Dfetch<float4>(tris, triAddr + 0);
					const float4 v11 = tex1Dfetch<float4>(tris, triAddr + 1);
					const float4 v22 = tex1Dfetch<float4>(tris, triAddr + 2);

					// End marker (negative zero) => all triangles processed.
					if (__float_as_int(v00.x) == 0x80000000)
						break;

					float Oz = v00.w - origx*v00.x - origy*v00.y - origz*v00.z;
					float invDz = 1.0f / (dirx*v00.x + diry*v00.y + dirz*v00.z);
					float t = Oz * invDz;

					if (t > tmin && t < hitT)
					{
						// Compute and check barycentric u.

						float Ox = v11.w + origx*v11.x + origy*v11.y + origz*v11.z;
						float Dx = dirx*v11.x + diry*v11.y + dirz*v11.z;
						u = Ox + t*Dx;

						if (u >= 0.0f)
						{
							// Compute and check barycentric v.

							float Oy = v22.w + origx*v22.x + origy*v22.y + origz*v22.z;
							float Dy = dirx*v22.x + diry*v22.y + dirz*v22.z;
							v = Oy + t*Dy;

							if (v >= 0.0f && u + v <= 1.0f)
							{
								// Record intersection.
								// Closest intersection not required => terminate.

								hitT = t;
								hitIndex = triAddr;
								if (anyHit)
								{
									nodeAddr = EntrypointSentinel;
									break;
								}
							}
						}
					}
				} // triangle

				// Another leaf was postponed => process it as well.
				{
					leafAddr = nodeAddr;
					if (nodeAddr < 0)
					{
						nodeAddr = *(int*)stackPtr;
						stackPtr -= 4;
					}
				}
			} // leaf

			// DYNAMIC FETCH
			if (__popc(__ballot(true)) < DYNAMIC_FETCH_THRESHOLD)
				break;

		} // traversal
		
		if (!anyHit)
		{
			if (hitIndex != -1)
			{
				GIndexType hit_id = tex1Dfetch<GIndexType>(triIndices, hitIndex);

				// Compute geometric normal.
				const uint3& vertex_index = make_uint3(scene_data.m_vtx_indices[0][hit_id], scene_data.m_vtx_indices[1][hit_id], scene_data.m_vtx_indices[2][hit_id]);
				const float3& v1 = make_float3(scene_data.m_vertices[0][vertex_index.x], scene_data.m_vertices[1][vertex_index.x], scene_data.m_vertices[2][vertex_index.x]);
				const float3& v2 = make_float3(scene_data.m_vertices[0][vertex_index.y], scene_data.m_vertices[1][vertex_index.y], scene_data.m_vertices[2][vertex_index.y]);
				const float3& v3 = make_float3(scene_data.m_vertices[0][vertex_index.z], scene_data.m_vertices[1][vertex_index.z], scene_data.m_vertices[2][vertex_index.z]);

				const float3 Ng = compute_normal(v1, v2, v3);

				// Compute shading normal.
				const uint3& normal_index = make_uint3(scene_data.m_nor_indices[0][hit_id], scene_data.m_nor_indices[1][hit_id], scene_data.m_nor_indices[2][hit_id]);
				const float3 n0 = make_float3(scene_data.m_normals[0][normal_index.x], scene_data.m_normals[1][normal_index.x], scene_data.m_normals[2][normal_index.x]);
				const float3 n1 = make_float3(scene_data.m_normals[0][normal_index.y], scene_data.m_normals[1][normal_index.y], scene_data.m_normals[2][normal_index.y]);
				const float3 n2 = make_float3(scene_data.m_normals[0][normal_index.z], scene_data.m_normals[1][normal_index.z], scene_data.m_normals[2][normal_index.z]);

				const float3 interpolated_normal = (1.0f - u - v) * n0 + u * n0 + v * n1;

				Ray input_ray(o, d);
				const float3 intersected_position = input_ray(hitT);

				// Get scene data only if early exit is not required.
				intersections.m_primitive_id[rayidx] = hit_id;
				intersections.m_primitive_material_id[rayidx] = scene_data.m_tri_material_ids[hit_id];
				intersections.m_position[0][rayidx] = intersected_position.x;
				intersections.m_position[1][rayidx] = intersected_position.y;
				intersections.m_position[2][rayidx] = intersected_position.z;
				intersections.m_geomteric_normal[0][rayidx] = Ng.x;
				intersections.m_geomteric_normal[1][rayidx] = Ng.y;
				intersections.m_geomteric_normal[2][rayidx] = Ng.z;
				intersections.m_shading_normal[0][rayidx] = interpolated_normal.x;
				intersections.m_shading_normal[1][rayidx] = interpolated_normal.y;
				intersections.m_shading_normal[2][rayidx] = interpolated_normal.z;
				intersections.m_incoming_direction[0][rayidx] = -d.x;
				intersections.m_incoming_direction[1][rayidx] = -d.y;
				intersections.m_incoming_direction[2][rayidx] = -d.z;
				intersections.m_front[rayidx] = dot(Ng, make_float3(d.x, d.y, d.z)) < 0.0f ? 1 : 0;
				intersections.m_uv[rayidx] = make_float2(u, v);
				intersections.m_epsilon[rayidx] = 1.0e-3f * hitT;
			}
		}

		intersections.m_intersected[rayidx] = (hitIndex != -1) ? 1 : 0;

	} while (true);
}

#endif			// !KEPLER_DYNAMIC_FETCH