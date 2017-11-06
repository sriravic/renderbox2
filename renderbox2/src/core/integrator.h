
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

#ifndef INTEGRATOR_H
#define INTEGRATOR_H

// Application specific headers.
#include <accelerators/sbvh/cudatracer.h>
#include <core/buffer.h>
#include <core/intersection.h>
#include <memory/memoryallocator.h>

// Cuda specific headers.

// Standard c++ headers.

// Forward declaration.
namespace renderbox2 { class CudaTracer; }
namespace renderbox2 { class Scene; }

namespace renderbox2
{

	//
	// Base Class Integrator for all the integrators implemented.
	//

	class Integrator
	{
	public:
		
		Integrator(MemoryAllocator& allocator) : m_allocator(allocator)
		{
			// Empty constructor.
		}

		//
		// Render method takes in a data pointer that can refer to any arbitrary data packet.
		// We will check at callee side to assert that number of data elements are same as those provided raising
		// an exception in case data is missing. This keeps the interface simple and neat.
		//

		virtual void render(Scene* scene, CudaTracer* tracer, void** data, const uint32_t n_data) = 0;

	protected:

		Scene*				m_scene;
		CudaTracer*			m_tracer;
		MemoryAllocator&	m_allocator;
	};


	//
	// Direct Lighting is a much used featuer in all integrators.
	// So we have a separate device function for computing direct lighting.
	//

	__device__ float3 compute_direct_lighting(SceneBuffer scene_buffer, MaterialBuffer material_buffer, BvhStruct bvh, Intersection& isect, uint32_t seed);

	
	//
	// Some util methods for all integrators.
	//

	__device__ void get_triangle_vertices(SceneBuffer scene_buffer, GIndexType id, float3& v0, float3& v1, float3& v2);

	__device__ void get_light_vertices(SceneBuffer scene_buffer, uint32_t light_id, float3& v0, float3& v1, float3& v2);

}				// !namespace renderbox2

#endif			// !INTEGRATORS_H
