
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

#ifndef SCENE_H
#define SCENE_H

// Application specific headers.
#include <core/buffer.h>
#include <core/globaltypes.h>
#include <core/material.h>
#include <core/mesh.h>
#include <core/params.h>
#include <core/primitives.h>

// Cuda specific headers.
#include <vector_types.h>

// Standard c++ headers.
#include <cassert>
#include <memory>
#include <vector>

// Forward declarations.
class CudaBvh;
namespace renderbox2 { class PerspectiveCamera; }
namespace renderbox2 { class Film; }
namespace renderbox2 { struct Filter; }

using namespace std;

namespace renderbox2
{

	//
	// Scene class.
	// Note: We never move the scene object to gpu directly. Instead we provide access to all data buffers allocated as device pointers.
	//

	class Scene
	{
	public:

		Scene(MemoryAllocator& m);

		~Scene();

		//
		// This method allocates all memory on the gpu and copies all the contents of the scene into appropriate pointers into device memory.
		//

		void                                    gpu_prepare();


		//
		// Get/Set methods to retrieve data.
		// NOTE: returns pointers to host memory.
		//

		const GIndexVec3Type*					get_tri_vtx_indices() const;
		const GIndexVec3Type*                   get_tri_nor_indices() const;
		const GIndexVec3Type*                   get_light_indices() const;
		const float3*	    				    get_tri_vertices() const;
		const float3*                           get_tri_normals() const;
		const uint32_t*							get_tri_material_ids() const;
		const GIndexType*						get_light_ids_to_primitive_ids() const;

		GIndexType                              get_num_triangles() const { return m_num_vtx_indices; }

		uint32_t								get_num_lights() const { return m_num_lights; }

		uint32_t								get_num_materials() const { return static_cast<uint32_t>(h_materials.size()); }

		AABB                                    get_scene_aabb() const { return m_scene_aabb; }

		//
		// Get Buffers to Device Memory contents.
		//

		VertexBuffer							gpu_get_vertices() const;
		NormalBuffer							gpu_get_normals() const;
		TextureBuffer							gpu_get_texcoords() const;
		VertexIndexBuffer						gpu_get_vertex_indices() const;
		NormalIndexBuffer						gpu_get_normal_indices() const;
		TextureIndexBuffer						gpu_get_texcoord_indices() const;
		VertexIndexBuffer						gpu_get_light_indices() const;
		SceneBuffer								gpu_get_buffer() const;
		MaterialBuffer							gpu_get_material_buffer() const;
		const uint32_t*							gpu_get_tri_material_ids() const;
		const GIndexType*						gpu_get_light_ids_to_primitive_ids() const;

		void                                    set_camera(PerspectiveCamera* camera) { m_camera = camera; }
		void                                    set_output_film(Film* film) { m_output_film = film; }
		void                                    set_bvh(CudaBvh* bvh) { m_bvh = bvh; }

		PerspectiveCamera*                      get_camera() { return m_camera; }
		Film*									get_output_film() { return m_output_film; }
		CudaBvh*								get_bvh() { return m_bvh; }


		//
		// Add a mesh to the scene.
		//

		void									add_mesh(const Mesh& M, bool is_emissive = false);

		
		//
		// Add/Get methods for bsdfs into the scene list.
		//

		void                                    add_diffuse_bsdf(LambertianBsdfParams& params);
		void									add_orennayar_bsdf(OrenNayarBsdfParams& params);
		void                                    add_mirror_bsdf(MirrorBsdfParams& params);
		void                                    add_glass_bsdf(GlassBsdfParams& params);
		void									add_microfacet_bsdf(MicrofacetBsdfParams& params);
		void									add_diffuse_light_bsdf(DiffuseEmitterParams& params);
		void									add_material(Material& material);

	private:


		//
		// CPU data. Eventually moved into the GPU.
		// 

		vector<float3>					h_vertices;
		vector<float3>					h_normals;
		vector<float2>					h_texcoords;
		vector<GIndexVec3Type>			h_vtx_indices;
		vector<GIndexVec3Type>			h_nor_indices;
		vector<GIndexVec3Type>			h_tex_indices;
		vector<GIndexVec3Type>			h_light_indices;
		vector<GIndexType>				h_light_ids_to_primitive_ids;				// Scenes are modelled explicitily with certain objects as emitters which act as lights.
		vector<uint32_t>				h_tri_material_ids;		
		
		
		//
		// Material related data.
		//

		vector<Material>				h_materials;
		vector<LambertianBsdfParams>	h_lambertian_bsdfs;
		vector<OrenNayarBsdfParams>		h_orennayar_bsdfs;
		vector<MirrorBsdfParams>		h_mirror_bsdfs;
		vector<GlassBsdfParams>			h_glass_bsdfs;
		vector<MicrofacetBsdfParams>	h_microfacet_bsdfs;
		vector<DiffuseEmitterParams>	h_diffuse_emitter_bsdfs;

		GIndexType						m_num_submeshes;
		GIndexType						m_num_vertices;
		GIndexType						m_num_normals;
		GIndexType						m_num_texcoords;
		GIndexType						m_num_vtx_indices;
		GIndexType						m_num_nor_indices;
		GIndexType						m_num_tex_indices;

		AABB							m_scene_aabb;

		uint32_t						m_num_lights;

		MemoryAllocator&				m_allocator;

		//
		// GPU data.
		//

		VertexBufferClass*				vb;
		NormalBufferClass*				nb;
		TextureBufferClass*				tb;
		VertexIndexBufferClass*			vib;
		NormalIndexBufferClass*			nib;
		TextureIndexBufferClass*		tib;
		VertexIndexBufferClass*			lib;								// Light index buffer.
		DevicePointer					light_ids_to_primitive_ids;			// One-to-One map between light ids and their actual primitive ids.
		DevicePointer					tri_material_ids;					// Material id of each triangle.

		MaterialBufferClass*			mb;

		// Other scene data
		CudaBvh*						m_bvh;
		PerspectiveCamera*				m_camera;
		Film*							m_output_film;
	};

}			// !namespace renderbox2

#endif		// !SCENE_H
