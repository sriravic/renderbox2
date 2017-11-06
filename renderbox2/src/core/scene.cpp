
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
#include <accelerators/sbvh/cudabvh.h>
#include <core/defs.h>
#include <core/camera.h>
#include <core/film.h>
#include <core/scene.h>

// Cuda specific headers.

// Standard c++ headers.
#include <iostream>

namespace renderbox2
{
	
	//
	// Scene class implementation.
	//

	Scene::Scene(MemoryAllocator& m) : m_allocator(m)
	{
		// Set all counters to zero
		m_num_submeshes = 0;
		m_num_normals = 0;
		m_num_nor_indices = 0;
		m_num_texcoords = 0;
		m_num_tex_indices = 0;
		m_num_vertices = 0;
		m_num_vtx_indices = 0;
		m_num_lights = 0;

		// set all device pointers to null.
		vb = nullptr;
		nb = nullptr;
		tb = nullptr;
		vib = nullptr;
		nib = nullptr;
		tib = nullptr;
		mb = nullptr;

		// other data
		m_bvh = nullptr;
		m_camera = nullptr;
		m_output_film = nullptr;
	}

	Scene::~Scene()
	{
		SAFE_RELEASE(vb);
		SAFE_RELEASE(nb);
		SAFE_RELEASE(tb);
		SAFE_RELEASE(vib);
		SAFE_RELEASE(nib);
		SAFE_RELEASE(tib);
		SAFE_RELEASE(mb);
		m_allocator.free(tri_material_ids);
		m_allocator.free(light_ids_to_primitive_ids);
	}

	//
	// Add a mesh to the scene and update the counters appropriately.
	//

	void Scene::add_mesh(const Mesh& M, bool is_emitter)
	{
		// Grow the scene's aabb
		m_scene_aabb.grow(M.m_bounds);

		// Copy the data into scene memory
		for (size_t i = 0; i < M.m_vertices.size(); i++)
			h_vertices.push_back(M.m_vertices[i]);
		for (size_t i = 0; i < M.m_normals.size(); i++)
			h_normals.push_back(M.m_normals[i]);
		for (size_t i = 0; i < M.m_texcoords.size(); i++)
			h_texcoords.push_back(M.m_texcoords[i]);
		for (size_t i = 0; i < M.m_vtx_indices.size(); i++)
			h_vtx_indices.push_back(M.m_vtx_indices[i]);
		for (size_t i = 0; i < M.m_nor_indices.size(); i++)
			h_nor_indices.push_back(M.m_nor_indices[i]);
		for (size_t i = 0; i < M.m_tex_indices.size(); i++)
			h_tex_indices.push_back(M.m_tex_indices[i]);
		for (size_t i = 0; i < M.m_mat_ids.size(); i++)
			h_tri_material_ids.push_back(M.m_mat_ids[i]);		

		// Check if its light.
		if (is_emitter)
		{
			for (size_t j = 0; j < M.m_vtx_indices.size(); j++)
			{
				h_light_indices.push_back(M.m_vtx_indices[j]);
				h_light_ids_to_primitive_ids.push_back(m_num_vtx_indices + static_cast<GIndexType>(j));
				m_num_lights++;
			}
		}

		// Update the counters.
		m_num_vtx_indices += static_cast<GIndexType>(M.m_vtx_indices.size());
		m_num_nor_indices += static_cast<GIndexType>(M.m_nor_indices.size());
		m_num_tex_indices += static_cast<GIndexType>(M.m_tex_indices.size());
		m_num_vertices += static_cast<GIndexType>(M.m_vertices.size());
		m_num_normals += static_cast<GIndexType>(M.m_normals.size());
		m_num_texcoords += static_cast<GIndexType>(M.m_texcoords.size());
		m_num_submeshes += 1;
	}


	//
	// CPU data retrieval functions.
	//

	const GIndexVec3Type* Scene::get_tri_vtx_indices() const { return &h_vtx_indices[0]; }

	const GIndexVec3Type* Scene::get_tri_nor_indices() const { return &h_nor_indices[0]; }

	const float3* Scene::get_tri_vertices() const { return &h_vertices[0]; }

	const float3* Scene::get_tri_normals() const { return &h_normals[0]; }

	const GIndexVec3Type* Scene::get_light_indices() const { return &h_light_indices[0]; }

	const uint32_t* Scene::get_tri_material_ids() const { return &h_tri_material_ids[0]; }


	//
	// GPU data retrieval methods.
	// All functions return smart pointers.
	//

	VertexBuffer Scene::gpu_get_vertices() const { return vb->get_buffer(); }

	NormalBuffer Scene::gpu_get_normals() const { return nb->get_buffer(); }

	TextureBuffer Scene::gpu_get_texcoords() const { return tb->get_buffer(); }

	VertexIndexBuffer Scene::gpu_get_vertex_indices() const { return vib->get_buffer(); }

	NormalIndexBuffer Scene::gpu_get_normal_indices() const { return nib->get_buffer(); }

	TextureIndexBuffer Scene::gpu_get_texcoord_indices() const { return tib->get_buffer(); }

	VertexIndexBuffer Scene::gpu_get_light_indices() const { return lib->get_buffer(); }

	const uint32_t* Scene::gpu_get_tri_material_ids() const { return static_cast<uint32_t*>(tri_material_ids.m_ptr); }

	// GPU prepare function.
	
	void Scene::gpu_prepare()
	{
		// First we split the tupled data into constituent terms so that we can easily copy them to device memory.
		std::cout << "Preparing Scene data for GPU. This might take some time. \n";

		std::vector<float> h_vtx_buffer[3];
		std::vector<float> h_nor_buffer[3];
		std::vector<float> h_tex_buffer[2];
		std::vector<GIndexType> h_vtx_index_buffer[3];
		std::vector<GIndexType> h_nor_index_buffer[3];
		std::vector<GIndexType> h_tex_index_buffer[3];
		std::vector<GIndexType> h_light_index_buffer[3];

		// Resize arrays.
		for (int i = 0; i < 3; i++)
		{
			h_vtx_buffer[i].resize(h_vertices.size());
			h_nor_buffer[i].resize(h_normals.size());
			h_vtx_index_buffer[i].resize(h_vtx_indices.size());
			h_nor_index_buffer[i].resize(h_nor_indices.size());
			h_tex_index_buffer[i].resize(h_tex_indices.size());
			h_light_index_buffer[i].resize(h_light_indices.size());
			
			if (i < 2)
			{
				h_tex_buffer[i].resize(h_texcoords.size());				
			}
		}

		// Strip and copy all data.
		for (size_t i = 0; i < h_vertices.size(); i++)
		{
			const float3& vertex = h_vertices[i];
			h_vtx_buffer[0][i] = vertex.x;
			h_vtx_buffer[1][i] = vertex.y;
			h_vtx_buffer[2][i] = vertex.z;
		}

		for (size_t i = 0; i < h_normals.size(); i++)
		{
			const float3& normal = h_normals[i];
			h_nor_buffer[0][i] = normal.x;
			h_nor_buffer[1][i] = normal.y;
			h_nor_buffer[2][i] = normal.z;
		}

		for (size_t i = 0; i < h_texcoords.size(); i++)
		{
			const float2& texcoord = h_texcoords[i];
			h_tex_buffer[0][i] = texcoord.x;
			h_tex_buffer[1][i] = texcoord.y;
		}

		for (size_t i = 0; i < h_vtx_indices.size(); i++)
		{
			const GIndexVec3Type& index = h_vtx_indices[i];
			h_vtx_index_buffer[0][i] = index.x;
			h_vtx_index_buffer[1][i] = index.y;
			h_vtx_index_buffer[2][i] = index.z;
		}

		for (size_t i = 0; i < h_nor_indices.size(); i++)
		{
			const GIndexVec3Type& index = h_nor_indices[i];
			h_nor_index_buffer[0][i] = index.x;
			h_nor_index_buffer[1][i] = index.y;
			h_nor_index_buffer[2][i] = index.z;
		}

		for (size_t i = 0; i < h_tex_indices.size(); i++)
		{
			const GIndexVec3Type& index = h_tex_indices[i];
			h_tex_index_buffer[0][i] = index.x;
			h_tex_index_buffer[1][i] = index.y;
			h_tex_index_buffer[2][i] = index.z;
		}

		for (size_t i = 0; i < h_light_indices.size(); i++)
		{
			const GIndexVec3Type& index = h_light_indices[i];
			h_light_index_buffer[0][i] = index.x;
			h_light_index_buffer[1][i] = index.y;
			h_light_index_buffer[2][i] = index.z;
		}

		// Allocate memory for buffers and copy contents into gpu.
		vb = new VertexBufferClass(m_allocator);
		nb = new NormalBufferClass(m_allocator);
		tb = new TextureBufferClass(m_allocator);
		vib = new VertexIndexBufferClass(m_allocator);
		nib = new NormalIndexBufferClass(m_allocator);
		tib = new TextureIndexBufferClass(m_allocator);
		lib = new VertexIndexBufferClass(m_allocator);

		vb->allocate(static_cast<GIndexType>(h_vertices.size()));
		nb->allocate(static_cast<GIndexType>(h_normals.size()));
		tb->allocate(static_cast<GIndexType>(h_texcoords.size()));
		vib->allocate(static_cast<GIndexType>(h_vtx_indices.size()));
		nib->allocate(static_cast<GIndexType>(h_nor_indices.size()));
		tib->allocate(static_cast<GIndexType>(h_tex_indices.size()));
		lib->allocate(static_cast<GIndexType>(h_light_indices.size()));
		tri_material_ids = m_allocator.allocate(sizeof(uint32_t) * h_tri_material_ids.size());
		light_ids_to_primitive_ids = m_allocator.allocate(sizeof(GIndexType) * h_light_ids_to_primitive_ids.size());

		vb->copy_to_buffer(h_vtx_buffer[0].data(), h_vtx_buffer[1].data(), h_vtx_buffer[2].data(), static_cast<GIndexType>(h_vertices.size()));
		nb->copy_to_buffer(h_nor_buffer[0].data(), h_nor_buffer[1].data(), h_nor_buffer[2].data(), static_cast<GIndexType>(h_normals.size()));
		tb->copy_to_buffer(h_tex_buffer[0].data(), h_tex_buffer[1].data(), static_cast<GIndexType>(h_texcoords.size()));
		vib->copy_to_buffer(h_vtx_index_buffer[0].data(), h_vtx_index_buffer[1].data(), h_vtx_index_buffer[2].data(), static_cast<GIndexType>(h_vtx_indices.size()));
		nib->copy_to_buffer(h_nor_index_buffer[0].data(), h_nor_index_buffer[1].data(), h_nor_index_buffer[2].data(), static_cast<GIndexType>(h_nor_indices.size()));
		tib->copy_to_buffer(h_tex_index_buffer[0].data(), h_tex_index_buffer[1].data(), h_tex_index_buffer[2].data(), static_cast<GIndexType>(h_tex_indices.size()));
		lib->copy_to_buffer(h_light_index_buffer[0].data(), h_light_index_buffer[1].data(), h_light_index_buffer[2].data(), static_cast<GIndexType>(h_light_indices.size()));

		checkCuda(cudaMemcpy(tri_material_ids.m_ptr, h_tri_material_ids.data(), sizeof(uint32_t) * h_tri_material_ids.size(), cudaMemcpyHostToDevice));
		checkCuda(cudaMemcpy(light_ids_to_primitive_ids.m_ptr, h_light_ids_to_primitive_ids.data(), sizeof(GIndexType) * h_light_ids_to_primitive_ids.size(), cudaMemcpyHostToDevice));

		// Move materials to gpu.
		mb = new MaterialBufferClass(m_allocator);

		mb->allocate_materials(static_cast<uint32_t>(h_materials.size()));
		mb->allocate_lambertian_bsdfs(static_cast<uint32_t>(h_lambertian_bsdfs.size()));
		mb->allocate_orennayar_bsdfs(static_cast<uint32_t>(h_orennayar_bsdfs.size()));
		mb->allocate_glass_bsdfs(static_cast<uint32_t>(h_glass_bsdfs.size()));
		mb->allocate_mirror_bsdfs(static_cast<uint32_t>(h_mirror_bsdfs.size()));
		mb->allocate_microfacet_bsdfs(static_cast<uint32_t>(h_microfacet_bsdfs.size()));
		mb->allocate_diffuse_emitter_bsdfs(static_cast<uint32_t>(h_diffuse_emitter_bsdfs.size()));

		mb->copy_to_material_buffer(h_materials.data(), static_cast<uint32_t>(h_materials.size()));
		mb->copy_to_lambertian_buffer(h_lambertian_bsdfs.data(), static_cast<uint32_t>(h_lambertian_bsdfs.size()));
		mb->copy_to_orennayar_buffer(h_orennayar_bsdfs.data(), static_cast<uint32_t>(h_orennayar_bsdfs.size()));
		mb->copy_to_glass_buffer(h_glass_bsdfs.data(), static_cast<uint32_t>(h_glass_bsdfs.size()));
		mb->copy_to_mirror_buffer(h_mirror_bsdfs.data(), static_cast<uint32_t>(h_mirror_bsdfs.size()));
		mb->copy_to_microfacet_buffer(h_microfacet_bsdfs.data(), static_cast<uint32_t>(h_microfacet_bsdfs.size()));
		mb->copy_to_diffuse_emitter_buffer(h_diffuse_emitter_bsdfs.data(), static_cast<uint32_t>(h_diffuse_emitter_bsdfs.size()));

		std::cout << "Finished moving scene data into GPU.\n";
	}

	SceneBuffer Scene::gpu_get_buffer() const
	{
		SceneBuffer ret;

		VertexBuffer vbuffer = vb->get_buffer();
		NormalBuffer nbuffer = nb->get_buffer();
		TextureBuffer tbuffer = tb->get_buffer();
		VertexIndexBuffer vibuffer = vib->get_buffer();
		NormalIndexBuffer nibuffer = nib->get_buffer();
		TextureIndexBuffer tibuffer = tib->get_buffer();
		VertexIndexBuffer libuffer = lib->get_buffer();

		for (uint32_t i = 0; i < 3; i++)
		{
			ret.m_vertices[i] = vbuffer.m_data[i];
			ret.m_normals[i] = nbuffer.m_data[i];
			
			if (i < 2)
			{
				ret.m_tex_coords[i] = tbuffer.m_data[i];
			}

			ret.m_vtx_indices[i] = vibuffer.m_data[i];
			ret.m_nor_indices[i] = nibuffer.m_data[i];
			ret.m_tex_indices[i] = tibuffer.m_data[i];
			ret.m_light_indices[i] = libuffer.m_data[i];
		}
		
		ret.m_tri_material_ids = static_cast<uint32_t*>(tri_material_ids.m_ptr);
		ret.m_light_ids_to_primitive_ids = static_cast<GIndexType*>(light_ids_to_primitive_ids.m_ptr);

		ret.m_num_lights = m_num_lights;

		// Size is equal to number of scene elements.
		ret.m_size = vibuffer.m_size;
		return ret;
	}

	MaterialBuffer Scene::gpu_get_material_buffer() const { return mb->get_buffer(); }

	// BSDF processing functions.

	void Scene::add_diffuse_bsdf(LambertianBsdfParams& params) { h_lambertian_bsdfs.push_back(params); }

	void Scene::add_orennayar_bsdf(OrenNayarBsdfParams& params) { h_orennayar_bsdfs.push_back(params); }

	void Scene::add_mirror_bsdf(MirrorBsdfParams& params) { h_mirror_bsdfs.push_back(params); }

	void Scene::add_glass_bsdf(GlassBsdfParams& params) { h_glass_bsdfs.push_back(params); }

	void Scene::add_microfacet_bsdf(MicrofacetBsdfParams& params) { h_microfacet_bsdfs.push_back(params); }

	void Scene::add_diffuse_light_bsdf(DiffuseEmitterParams& params) { h_diffuse_emitter_bsdfs.push_back(params); }

	void Scene::add_material(Material& material) { h_materials.push_back(material); }
}
