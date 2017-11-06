
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

// Application specific headers.
#include <core/buffer.h>
#include <core/defs.h>
#include <core/material.h>
#include <core/params.h>

// Cuda specific headers.

// Standard c++ headers.

namespace renderbox2
{

	//
	// RayBufferClass implementation.
	//

	RayBufferClass::RayBufferClass(MemoryAllocator& m)
		: m_allocator(m)
	{
		m_size = 0;
	}

	RayBufferClass::~RayBufferClass()
	{
		m_allocator.free(m_data);
	}

	void RayBufferClass::allocate(GIndexType size)
	{
		m_size = size;
		m_data = m_allocator.allocate(sizeof(Ray) * size);
	}

	RayBuffer RayBufferClass::get_buffer()
	{
		RayBuffer rb;
		rb.m_data = static_cast<Ray*>(m_data.m_ptr);
		rb.m_size = m_size;
		return rb;
	}

	// NOTE: We assume that buffer class has enough memory allocated already.

	void RayBufferClass::copy_to_buffer(const Ray* data0, GIndexType size)
	{
		assert(size <= m_size);
		checkCuda(cudaMemcpy(m_data.m_ptr, data0, sizeof(Ray) * size, cudaMemcpyHostToDevice));
	}

	void RayBufferClass::copy_from_buffer(Ray* data, GIndexType size) const
	{
		assert(size <= m_size);
		checkCuda(cudaMemcpy(data, m_data.m_ptr, sizeof(Ray) * size, cudaMemcpyDeviceToHost));
	}


	//
	// Float2BufferClass implmentation.
	//

	Float2BufferClass::Float2BufferClass(MemoryAllocator& m)
		: m_allocator(m)
	{
		m_size = 0;
	}

	Float2BufferClass::~Float2BufferClass()
	{
		m_allocator.free(m_data[0]);
		m_allocator.free(m_data[1]);
	}

	void Float2BufferClass::allocate(GIndexType size)
	{
		m_size = size;
		m_data[0] = m_allocator.allocate(sizeof(float) * m_size);
		m_data[1] = m_allocator.allocate(sizeof(float) * m_size);
	}

	Float2Buffer Float2BufferClass::get_buffer()
	{
		Float2Buffer fb;
		fb.m_size = m_size;
		fb.m_data[0] = static_cast<float*>(m_data[0].m_ptr);
		fb.m_data[1] = static_cast<float*>(m_data[1].m_ptr);
		return fb;
	}

	void Float2BufferClass::copy_to_buffer(const float* data0, const float* data1, GIndexType size)
	{
		assert(size <= m_size);
		checkCuda(cudaMemcpy(m_data[0].m_ptr, data0, sizeof(float) * size, cudaMemcpyHostToDevice));
		checkCuda(cudaMemcpy(m_data[1].m_ptr, data1, sizeof(float) * size, cudaMemcpyHostToDevice));
	}

	void Float2BufferClass::copy_from_buffer(float* data0, float* data1, GIndexType size) const
	{
		assert(size <= m_size);
		checkCuda(cudaMemcpy(data0, m_data[0].m_ptr, sizeof(float) * size, cudaMemcpyDeviceToHost));
		checkCuda(cudaMemcpy(data1, m_data[0].m_ptr, sizeof(float) * size, cudaMemcpyDeviceToHost));
	}


	//
	// Float3BufferClass implementation.
	//

	Float3BufferClass::Float3BufferClass(MemoryAllocator& m)
		: m_allocator(m)
	{
		m_size = 0;
	}

	Float3BufferClass::~Float3BufferClass()
	{
		m_allocator.free(m_data[0]);
		m_allocator.free(m_data[1]);
		m_allocator.free(m_data[2]);
	}

	void Float3BufferClass::allocate(GIndexType size)
	{
		m_size = size;
		m_data[0] = m_allocator.allocate(sizeof(float) * m_size);
		m_data[1] = m_allocator.allocate(sizeof(float) * m_size);
		m_data[2] = m_allocator.allocate(sizeof(float) * m_size);
	}

	Float3Buffer Float3BufferClass::get_buffer()
	{
		Float3Buffer fb;
		fb.m_size = m_size;
		fb.m_data[0] = static_cast<float*>(m_data[0].m_ptr);
		fb.m_data[1] = static_cast<float*>(m_data[1].m_ptr);
		fb.m_data[2] = static_cast<float*>(m_data[2].m_ptr);
		return fb;
	}

	void Float3BufferClass::copy_to_buffer(const float* data0, const float* data1, const float* data2, GIndexType size)
	{
		assert(size <= m_size);
		checkCuda(cudaMemcpy(m_data[0].m_ptr, data0, sizeof(float) * size, cudaMemcpyHostToDevice));
		checkCuda(cudaMemcpy(m_data[1].m_ptr, data1, sizeof(float) * size, cudaMemcpyHostToDevice));
		checkCuda(cudaMemcpy(m_data[2].m_ptr, data2, sizeof(float) * size, cudaMemcpyHostToDevice));
	}

	void Float3BufferClass::copy_from_buffer(float* data0, float* data1, float* data2, GIndexType size) const
	{
		assert(size <= size);
		checkCuda(cudaMemcpy(data0, m_data[0].m_ptr, sizeof(float) * size, cudaMemcpyDeviceToHost));
		checkCuda(cudaMemcpy(data1, m_data[1].m_ptr, sizeof(float) * size, cudaMemcpyDeviceToHost));
		checkCuda(cudaMemcpy(data2, m_data[2].m_ptr, sizeof(float) * size, cudaMemcpyDeviceToHost));
	}


	//
	// Index2BufferClass implementation.
	//

	Index2BufferClass::Index2BufferClass(MemoryAllocator& m)
		: m_allocator(m)
	{
		m_size = 0;
	}

	Index2BufferClass::~Index2BufferClass()
	{
		m_allocator.free(m_data[0]);
		m_allocator.free(m_data[1]);
	}

	void Index2BufferClass::allocate(GIndexType size)
	{
		m_size = size;
		m_data[0] = m_allocator.allocate(sizeof(GIndexType) * m_size);
		m_data[1] = m_allocator.allocate(sizeof(GIndexType) * m_size);
	}

	Index2Buffer Index2BufferClass::get_buffer()
	{
		Index2Buffer ib;
		ib.m_size = m_size;
		ib.m_data[0] = static_cast<GIndexType*>(m_data[0].m_ptr);
		ib.m_data[1] = static_cast<GIndexType*>(m_data[1].m_ptr);
		return ib;
	}

	void Index2BufferClass::copy_to_buffer(const GIndexType* data0, const GIndexType* data1, GIndexType size)
	{
		assert(size <= m_size);
		checkCuda(cudaMemcpy(m_data[0].m_ptr, data0, sizeof(GIndexType) * size, cudaMemcpyHostToDevice));
		checkCuda(cudaMemcpy(m_data[1].m_ptr, data1, sizeof(GIndexType) * size, cudaMemcpyHostToDevice));
	}

	void Index2BufferClass::copy_from_buffer(GIndexType* data0, GIndexType* data1, GIndexType size) const
	{
		assert(size <= m_size);
		checkCuda(cudaMemcpy(data0, m_data[0].m_ptr, sizeof(GIndexType) * size, cudaMemcpyDeviceToHost));
		checkCuda(cudaMemcpy(data1, m_data[1].m_ptr, sizeof(GIndexType) * size, cudaMemcpyDeviceToHost));
	}


	//
	// Index3BufferClass implementation.
	//

	Index3BufferClass::Index3BufferClass(MemoryAllocator& m)
		: m_allocator(m)
	{
		m_size = 0;
	}

	Index3BufferClass::~Index3BufferClass()
	{
		m_allocator.free(m_data[0]);
		m_allocator.free(m_data[1]);
		m_allocator.free(m_data[2]);
	}

	void Index3BufferClass::allocate(GIndexType size)
	{
		m_size = size;
		m_data[0] = m_allocator.allocate(sizeof(GIndexType) * m_size);
		m_data[1] = m_allocator.allocate(sizeof(GIndexType) * m_size);
		m_data[2] = m_allocator.allocate(sizeof(GIndexType) * m_size);
	}

	Index3Buffer Index3BufferClass::get_buffer()
	{
		Index3Buffer ib;
		ib.m_size = m_size;
		ib.m_data[0] = static_cast<GIndexType*>(m_data[0].m_ptr);
		ib.m_data[1] = static_cast<GIndexType*>(m_data[1].m_ptr);
		ib.m_data[2] = static_cast<GIndexType*>(m_data[2].m_ptr);
		return ib;
	}

	void Index3BufferClass::copy_to_buffer(const GIndexType* data0, const GIndexType* data1, const GIndexType* data2, GIndexType size)
	{
		assert(size <= m_size);
		checkCuda(cudaMemcpy(m_data[0].m_ptr, data0, sizeof(GIndexType) * size, cudaMemcpyHostToDevice));
		checkCuda(cudaMemcpy(m_data[1].m_ptr, data1, sizeof(GIndexType) * size, cudaMemcpyHostToDevice));
		checkCuda(cudaMemcpy(m_data[2].m_ptr, data2, sizeof(GIndexType) * size, cudaMemcpyHostToDevice));
	}

	void Index3BufferClass::copy_from_buffer(GIndexType* data0, GIndexType* data1, GIndexType* data2, GIndexType size) const
	{
		assert(size <= m_size);
		checkCuda(cudaMemcpy(data0, m_data[0].m_ptr, sizeof(GIndexType) * size, cudaMemcpyDeviceToHost));
		checkCuda(cudaMemcpy(data1, m_data[1].m_ptr, sizeof(GIndexType) * size, cudaMemcpyDeviceToHost));
		checkCuda(cudaMemcpy(data2, m_data[2].m_ptr, sizeof(GIndexType) * size, cudaMemcpyDeviceToHost));
	}


	//
	// CameraSampleBufferClass implementation.
	//

	CameraSampleBufferClass::CameraSampleBufferClass(MemoryAllocator& allocator)
		: m_allocator(allocator)
	{
		m_size = 0;
	}

	CameraSampleBufferClass::~CameraSampleBufferClass()
	{
		m_allocator.free(m_ids);
		m_allocator.free(m_pixel_coords);
		m_allocator.free(m_alives);
		m_allocator.free(m_throughput);
		m_allocator.free(m_continue_probability);
		m_allocator.free(m_contribution);
	}

	void CameraSampleBufferClass::allocate(GIndexType size)
	{
		m_size					= size;
		m_ids					= m_allocator.allocate(sizeof(uint32_t) * size);
		m_pixel_coords			= m_allocator.allocate(sizeof(float2) * size);
		m_alives				= m_allocator.allocate(sizeof(uint8_t) * size);
		m_throughput			= m_allocator.allocate(sizeof(float4) * size);
		m_continue_probability	= m_allocator.allocate(sizeof(float) * size);
		m_contribution			= m_allocator.allocate(sizeof(float4) * size);
	}

	CameraSampleBuffer CameraSampleBufferClass::get_buffer() const
	{
		CameraSampleBuffer ret;

		ret.m_size = m_size;
		ret.m_ids					= static_cast<uint32_t*>(m_ids.m_ptr);
		ret.m_pixel_coords			= static_cast<float2*>(m_pixel_coords.m_ptr);
		ret.m_alives				= static_cast<uint8_t*>(m_alives.m_ptr);
		ret.m_throughput			= static_cast<float4*>(m_throughput.m_ptr);
		ret.m_continue_probability	= static_cast<float*>(m_continue_probability.m_ptr);
		ret.m_contribution			= static_cast<float4*>(m_contribution.m_ptr);

		return ret;
	}


	//
	// Material Buffer Class implementation.
	//

	MaterialBufferClass::MaterialBufferClass(MemoryAllocator& allocator)
		: m_allocator(allocator)
	{
		// Empty constructor.
	}

	MaterialBufferClass::~MaterialBufferClass()
	{
		m_allocator.free(m_materials);
		m_allocator.free(m_lambertian_bsdfs);
		m_allocator.free(m_orennayar_bsdfs);
		m_allocator.free(m_glass_bsdfs);
		m_allocator.free(m_mirror_bsdfs);
		m_allocator.free(m_microfacet_bsdfs);
		m_allocator.free(m_diffuse_emitter_bsdfs);
	}

	void MaterialBufferClass::allocate_materials(uint32_t size)
	{
		m_num_materials = size;
		m_materials = m_allocator.allocate(sizeof(Material) * size);
	}

	void MaterialBufferClass::allocate_lambertian_bsdfs(uint32_t size)
	{
		m_num_lambertian_bsdfs = size;
		m_lambertian_bsdfs = m_allocator.allocate(sizeof(LambertianBsdfParams) * size);
	}

	void MaterialBufferClass::allocate_orennayar_bsdfs(uint32_t size)
	{
		m_num_orennayar_bsdfs = size;
		m_orennayar_bsdfs = m_allocator.allocate(sizeof(OrenNayarBsdfParams) * size);
	}

	void MaterialBufferClass::allocate_glass_bsdfs(uint32_t size)
	{
		m_num_glass_bsdfs = size;
		m_glass_bsdfs = m_allocator.allocate(sizeof(GlassBsdfParams) * size);
	}

	void MaterialBufferClass::allocate_mirror_bsdfs(uint32_t size)
	{
		m_num_mirror_bsdfs = size;
		m_mirror_bsdfs = m_allocator.allocate(sizeof(MirrorBsdfParams) * size);
	}

	void MaterialBufferClass::allocate_microfacet_bsdfs(uint32_t size)
	{
		m_num_microfacet_bsdfs = size;
		m_microfacet_bsdfs = m_allocator.allocate(sizeof(MicrofacetBsdfParams) * size);
	}

	void MaterialBufferClass::allocate_diffuse_emitter_bsdfs(uint32_t size)
	{
		m_num_diffuse_emitter_bsdfs = size;
		m_diffuse_emitter_bsdfs = m_allocator.allocate(sizeof(DiffuseEmitterParams) * size);
	}

	void MaterialBufferClass::copy_to_material_buffer(Material* materials, uint32_t size)
	{
		assert(size <= m_num_materials);
		checkCuda(cudaMemcpy(m_materials.m_ptr, materials, sizeof(Material) * size, cudaMemcpyHostToDevice));
	}

	void MaterialBufferClass::copy_to_lambertian_buffer(LambertianBsdfParams* params, uint32_t size)
	{
		assert(size <= m_num_lambertian_bsdfs);
		checkCuda(cudaMemcpy(m_lambertian_bsdfs.m_ptr, params, sizeof(LambertianBsdfParams) * size, cudaMemcpyHostToDevice));
	}

	void MaterialBufferClass::copy_to_orennayar_buffer(OrenNayarBsdfParams* params, uint32_t size)
	{
		assert(size <= m_num_orennayar_bsdfs);
		checkCuda(cudaMemcpy(m_orennayar_bsdfs.m_ptr, params, sizeof(OrenNayarBsdfParams) * size, cudaMemcpyHostToDevice));
	}

	void MaterialBufferClass::copy_to_glass_buffer(GlassBsdfParams* params, uint32_t size)
	{
		assert(size <= m_num_glass_bsdfs);
		checkCuda(cudaMemcpy(m_glass_bsdfs.m_ptr, params, sizeof(GlassBsdfParams) * size, cudaMemcpyHostToDevice));
	}

	void MaterialBufferClass::copy_to_mirror_buffer(MirrorBsdfParams* params, uint32_t size)
	{
		assert(size <= m_num_mirror_bsdfs);
		checkCuda(cudaMemcpy(m_mirror_bsdfs.m_ptr, params, sizeof(MirrorBsdfParams) * size, cudaMemcpyHostToDevice));
	}

	void MaterialBufferClass::copy_to_microfacet_buffer(MicrofacetBsdfParams* params, uint32_t size)
	{
		assert(size <= m_num_microfacet_bsdfs);
		checkCuda(cudaMemcpy(m_microfacet_bsdfs.m_ptr, params, sizeof(MicrofacetBsdfParams) * size, cudaMemcpyHostToDevice));
	}

	void MaterialBufferClass::copy_to_diffuse_emitter_buffer(DiffuseEmitterParams* params, uint32_t size)
	{
		assert(size <= m_num_diffuse_emitter_bsdfs);
		checkCuda(cudaMemcpy(m_diffuse_emitter_bsdfs.m_ptr, params, sizeof(DiffuseEmitterParams) * size, cudaMemcpyHostToDevice));
	}

	MaterialBuffer MaterialBufferClass::get_buffer() const
	{
		MaterialBuffer ret;

		ret.m_materials				= static_cast<Material*>(m_materials.m_ptr);
		ret.m_lambertian_bsdfs		= static_cast<LambertianBsdfParams*>(m_lambertian_bsdfs.m_ptr);
		ret.m_orennayar_bsdfs		= static_cast<OrenNayarBsdfParams*>(m_orennayar_bsdfs.m_ptr);
		ret.m_glass_bsdfs			= static_cast<GlassBsdfParams*>(m_glass_bsdfs.m_ptr);
		ret.m_mirror_bsdfs			= static_cast<MirrorBsdfParams*>(m_mirror_bsdfs.m_ptr);
		ret.m_microfacet_bsdfs		= static_cast<MicrofacetBsdfParams*>(m_microfacet_bsdfs.m_ptr);
		ret.m_diffuse_emitter_bsdfs = static_cast<DiffuseEmitterParams*>(m_diffuse_emitter_bsdfs.m_ptr);

		ret.m_num_materials				= m_num_materials;
		ret.m_num_lambertian_bsdfs		= m_num_lambertian_bsdfs;
		ret.m_num_orennayar_bsdfs		= m_num_orennayar_bsdfs;
		ret.m_num_glass_bsdfs			= m_num_glass_bsdfs;
		ret.m_num_mirror_bsdfs			= m_num_mirror_bsdfs;
		ret.m_num_microfacet_bsdfs		= m_num_microfacet_bsdfs;
		ret.m_num_diffuse_emitter_bsdfs	= m_num_diffuse_emitter_bsdfs;
		
		return ret;
	}
}
