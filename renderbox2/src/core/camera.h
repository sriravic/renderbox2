
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

#ifndef CAMERA_H
#define CAMERA_H

// Application specific headers.
#include <core/primitives.h>
#include <core/transform.h>

// Cuda specific headers.

// Standard c++ headers.
#include <cstdint>

namespace renderbox2
{

	//
	// Perspective Camera class.
	//

	class PerspectiveCamera
	{
	public:

		__host__ __device__ PerspectiveCamera()
		{
			// Empty constructor.
		}

		__host__ __device__ PerspectiveCamera(const float3& position,
											  const float3& lookat,
											  const float3& up,
											  const float2& resolution,
											  float horizontal_fov);

		__host__ __device__ ~PerspectiveCamera()
		{
			// Nothing being done here.
		}

		__host__ __device__ int raster_to_index(const float2& pixel_coords) const;
		
		__host__ __device__ float2 index_to_raster(const int& pixel_index) const;

		__host__ __device__ float3 raster_to_world(const float2& raster_xy) const;
		
		__host__ __device__ float2 world_to_raster(const float3& world_pos) const;
		
		__host__ __device__ float3 world_to_ndc(const float3& world_pos) const;

		__host__ __device__ bool check_raster(const float2& raster_xy) const;

		__host__ __device__ Ray generate_ray(float x, float y) const;

		__host__ __device__  __inline__ uint32_t get_film_height() const { return static_cast<uint32_t>(floor(m_resolution.x)); }

		__host__ __device__  __inline__ uint32_t get_film_width()  const { return static_cast<uint32_t>(floor(m_resolution.y)); }

		__host__ __device__ __inline__ uint32_t get_total_pixels() const { return get_film_width() * get_film_height(); }

	private:
		float3 m_position;
		float3 m_forward;
		float2 m_resolution;

		Transform t_camera_to_screen, t_raster_to_camera;
		Transform t_screen_to_raster, t_raster_to_screen;
		Transform t_world_to_raster, t_raster_to_world;
		Transform t_world_to_screen, t_screen_to_world;
		Transform t_camera_to_world, t_world_to_camera;

		float  m_image_plane_distance;
	};

}			// !namespace renderbox2

#endif		// !CAMERA_H
