
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

#ifndef BUFFER_H
#define BUFFER_H

// Application specific headers.
#include <core/globaltypes.h>
#include <core/primitives.h>
#include <memory/memoryallocator.h>

// Cuda specific headers.
#include <vector_types.h>

// Standard c++ headers.
#include <cstdint>

// Forward declarations.
namespace renderbox2 { struct Material; }
namespace renderbox2 { struct LambertianBsdfParams; }
namespace renderbox2 { struct OrenNayarBsdfParams; }
namespace renderbox2 { struct GlassBsdfParams; }
namespace renderbox2 { struct MirrorBsdfParams; }
namespace renderbox2 { struct DiffuseEmitterParams; }
namespace renderbox2 { struct MicrofacetBsdfParams; }

namespace renderbox2
{
	
	//
	// Buffers are the way data is passed in and around the GPU. They are primarily in either Structure of Arrays(SOA) format or Array of Structures(AoS) format.
	// Mostly data will be stored in SoA format with helper structures to create/move/manipulate/encapsulate the pointers effectively.
	// Buffer classes will be responsible for managing their own memory. Buffer Structs will be used to move data to kernels on the gpu.
	//


	//
	// Float2 Buffer.
	// Stores all components of a float2 in SoA format.
	//

	struct Float2Buffer
	{
		float*		m_data[2];
		GIndexType	m_size;
	};
	
	typedef Float2Buffer TextureBuffer;


	//
	// Float3 Buffer.
	// Stores all the components of a float3 in SoA format.
	//

	struct Float3Buffer
	{
		float*     m_data[3];
		GIndexType m_size;
	};

	typedef Float3Buffer VertexBuffer;
	typedef Float3Buffer NormalBuffer;


	//
	// Index2 Buffer
	// Stores two indices of the primary components for texture uv data.
	//

	struct Index2Buffer
	{
		GIndexType* m_data[2];
		GIndexType  m_size;
	};


	//
	// Index3 Buffer
	// Stores three indices of primary components for vertices/normals.
	//

	struct Index3Buffer
	{
		GIndexType* m_data[3];
		GIndexType  m_size;
	};

	typedef Index3Buffer VertexIndexBuffer;
	typedef Index3Buffer NormalIndexBuffer;
	typedef Index3Buffer TextureIndexBuffer;

	//
	// RayBufferStruct - Encapsulation Structure for moving ray data in a SoA format inside the gpu kernels.
	// Ray data is packed as [origin, tmin] and [direction, tmax]
	//

	struct RayBuffer
	{
		Ray*		m_data;
		GIndexType	m_size;
	};
	

	// NOTE: All buffer allocation strategies allocate memory for 'size' elements and NOT 'size' bytes.

	//
	// RayBufferClass - Stores all ray data in a SoA format.
	// Explicitly manages device memory from the host side.
	//

	class RayBufferClass
	{
	public:
		
		RayBufferClass(MemoryAllocator& m);
		
		~RayBufferClass();

		void allocate(GIndexType size);

		void copy_to_buffer(const Ray* data, GIndexType size);

		void copy_from_buffer(Ray* data, GIndexType size) const;

		RayBuffer get_buffer();
	
	private:

		MemoryAllocator&	m_allocator;
		DevicePointer		m_data;
		GIndexType			m_size;
	};


	//
	// Float2BufferClass - stores and manages all float2 data on the gpu.
	//

	class Float2BufferClass
	{
	public:

		Float2BufferClass(MemoryAllocator& m);

		~Float2BufferClass();

		void allocate(GIndexType size);

		void copy_to_buffer(const float* data0, const float* data1, GIndexType size);

		void copy_from_buffer(float* data0, float* data1, GIndexType size) const;

		Float2Buffer get_buffer();

	private:
		
		MemoryAllocator&	m_allocator;
		DevicePointer		m_data[2];
		GIndexType			m_size;
	};

	typedef Float2BufferClass TextureBufferClass;


	//
	// Float3BufferClass - stores and manages all float3 data on the gpu.
	//

	class Float3BufferClass
	{
	public:
		
		Float3BufferClass(MemoryAllocator& m);
		
		~Float3BufferClass();

		void allocate(GIndexType size);

		void copy_to_buffer(const float* data0, const float* data1, const float* data2, GIndexType size);

		void copy_from_buffer(float* data0, float* data1, float* data2, GIndexType size) const;

		Float3Buffer get_buffer();
	
	private:

		MemoryAllocator&	m_allocator;
		DevicePointer		m_data[3];
		GIndexType			m_size;
	};
	
	typedef Float3BufferClass VertexBufferClass;
	typedef Float3BufferClass NormalBufferClass;


	//
	// Index2BufferClass - stores/manages all index2 data on the gpu.
	//

	class Index2BufferClass
	{
	public:

		Index2BufferClass(MemoryAllocator& m);

		~Index2BufferClass();

		void allocate(GIndexType size);

		void copy_to_buffer(const GIndexType* data0, const GIndexType* data1, GIndexType size);

		void copy_from_buffer(GIndexType* data0, GIndexType* data1, GIndexType size) const;

		Index2Buffer get_buffer();

	private:

		MemoryAllocator& m_allocator;
		DevicePointer    m_data[2];
		GIndexType       m_size;
	};
	
	
	//
	// Index3BufferClass - stores/manages all index3 data on the gpu.
	//

	class Index3BufferClass
	{
	public:

		Index3BufferClass(MemoryAllocator& m);

		~Index3BufferClass();

		void allocate(GIndexType size);

		void copy_to_buffer(const GIndexType* data0, const GIndexType* data1, const GIndexType* data2, GIndexType size);

		void copy_from_buffer(GIndexType* data0, GIndexType* data1, GIndexType* data2, GIndexType size) const;

		Index3Buffer get_buffer();
	
	private:
		
		MemoryAllocator&	m_allocator;
		DevicePointer		m_data[3];
		GIndexType			m_size;
	};

	typedef Index3BufferClass VertexIndexBufferClass;
	typedef Index3BufferClass NormalIndexBufferClass;
	typedef Index3BufferClass TextureIndexBufferClass;


	//
	// CameraSampleBuffer is a buffer containing all data that will be used by camera rays to store value in pixels.
	//

	struct CameraSampleBuffer
	{
		uint32_t*			m_ids;
		float2*				m_pixel_coords;
		uint8_t*			m_alives;
		float4*				m_throughput;
		float*				m_continue_probability;
		float4*				m_contribution;
		GIndexType			m_size;
	};


	//
	// CameraSampleBufferClass is responsible for maintaining camera samples.
	//

	class CameraSampleBufferClass
	{
	public:

		CameraSampleBufferClass(MemoryAllocator& allocator);

		~CameraSampleBufferClass();

		void allocate(GIndexType size);

		CameraSampleBuffer get_buffer() const;

	private:

		DevicePointer		m_ids;
		DevicePointer		m_pixel_coords;
		DevicePointer		m_alives;
		DevicePointer		m_throughput;
		DevicePointer		m_continue_probability;
		DevicePointer		m_contribution;
		MemoryAllocator&	m_allocator;
		GIndexType			m_size;
	};


	//
	// SceneBuffer contains all the pointers to scene data.
	//

	struct SceneBuffer
	{
		float*		m_vertices[3];
		float*		m_normals[3];
		float*		m_tex_coords[2];
		GIndexType* m_vtx_indices[3];
		GIndexType* m_nor_indices[3];
		GIndexType* m_tex_indices[3];
		GIndexType* m_light_indices[3];
		uint32_t*	m_tri_material_ids;
		GIndexType* m_light_ids_to_primitive_ids;
		GIndexType	m_size;
		uint32_t	m_num_lights;
	};

	
	//
	// Material buffer contains all pointers to material data.
	//

	struct MaterialBuffer
	{
		Material*				m_materials;

		LambertianBsdfParams*	m_lambertian_bsdfs;
		OrenNayarBsdfParams*	m_orennayar_bsdfs;
		MirrorBsdfParams*		m_mirror_bsdfs;
		GlassBsdfParams*		m_glass_bsdfs;
		MicrofacetBsdfParams*   m_microfacet_bsdfs;
		DiffuseEmitterParams*	m_diffuse_emitter_bsdfs;

		uint32_t				m_num_materials;
		uint32_t				m_num_lambertian_bsdfs;
		uint32_t				m_num_orennayar_bsdfs;
		uint32_t				m_num_mirror_bsdfs;
		uint32_t				m_num_glass_bsdfs;
		uint32_t				m_num_microfacet_bsdfs;
		uint32_t				m_num_diffuse_emitter_bsdfs;
	};


	class MaterialBufferClass
	{
	public:

		MaterialBufferClass(MemoryAllocator& allocator);

		~MaterialBufferClass();

		// Allocate methods to allocate gpu memory for all the material data.
		void allocate_materials(uint32_t size);
		void allocate_lambertian_bsdfs(uint32_t size);
		void allocate_orennayar_bsdfs(uint32_t size);
		void allocate_mirror_bsdfs(uint32_t size);
		void allocate_glass_bsdfs(uint32_t size);
		void allocate_microfacet_bsdfs(uint32_t size);
		void allocate_diffuse_emitter_bsdfs(uint32_t size);
		
		// Copy methods to copy material data from host to the gpu.
		void copy_to_material_buffer(Material* materials, uint32_t size);
		void copy_to_lambertian_buffer(LambertianBsdfParams* params, uint32_t size);
		void copy_to_orennayar_buffer(OrenNayarBsdfParams* params, uint32_t size);
		void copy_to_mirror_buffer(MirrorBsdfParams* params, uint32_t size);
		void copy_to_glass_buffer(GlassBsdfParams* params, uint32_t size);
		void copy_to_microfacet_buffer(MicrofacetBsdfParams* params, uint32_t size);
		void copy_to_diffuse_emitter_buffer(DiffuseEmitterParams* params, uint32_t size);

		MaterialBuffer get_buffer() const;

	private:
		DevicePointer		m_materials;
		DevicePointer		m_lambertian_bsdfs;
		DevicePointer		m_orennayar_bsdfs;
		DevicePointer		m_mirror_bsdfs;
		DevicePointer		m_glass_bsdfs;
		DevicePointer		m_microfacet_bsdfs;
		DevicePointer		m_diffuse_emitter_bsdfs;

		uint32_t			m_num_materials;
		uint32_t			m_num_lambertian_bsdfs;
		uint32_t			m_num_orennayar_bsdfs;
		uint32_t			m_num_mirror_bsdfs;
		uint32_t			m_num_glass_bsdfs;
		uint32_t			m_num_microfacet_bsdfs;
		uint32_t			m_num_diffuse_emitter_bsdfs;

		MemoryAllocator&	m_allocator;
	};
};			// !namespace renderbox2

#endif		// !BUFFER_H
