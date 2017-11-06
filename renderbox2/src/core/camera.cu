
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
#include <core/camera.h>
#include <core/util.h>

// Cuda specific headers.

// Standard c++ headers.

namespace renderbox2
{
	__host__ __device__ PerspectiveCamera::PerspectiveCamera(const float3& _position,
															 const float3& _lookat,
															 const float3& _up,
															 const float2& _resolution,
															 float         _horizontal_fov)
	{
		m_position = _position;
		m_forward = normalize(_lookat - _position);
		m_resolution = _resolution;

		t_world_to_camera = lookat(_position, _lookat, _up) * scale(-1.0f, 1.0f, 1.0f);
		t_camera_to_world = t_world_to_camera.inverse();

		Transform t_cam_to_screen = perspective(_horizontal_fov, 1.0f, 1000.0f);

		t_world_to_screen = t_cam_to_screen * t_world_to_camera;
		t_screen_to_world = t_world_to_screen.inverse();

		t_screen_to_raster = scale(_resolution.x, _resolution.y, 1.0f) * scale(0.5f, -0.5f, 1.0f) * translate(1.0f, -1.0f, 0.0f);
		t_raster_to_screen = t_screen_to_raster.inverse();

		t_world_to_raster = t_screen_to_raster * t_world_to_screen;
		t_raster_to_world = t_world_to_raster.inverse();

		t_raster_to_camera = t_cam_to_screen.inverse() * t_raster_to_screen;
		m_image_plane_distance = m_resolution.x / ( 2.0f * tanf(to_radians(_horizontal_fov) * 0.5f));
	}

	__host__ __device__ int PerspectiveCamera::raster_to_index(const float2& raster_xy) const
	{
		return int(floor(raster_xy.x) + floor(raster_xy.y) * m_resolution.x);
	}

	__host__ __device__ float2 PerspectiveCamera::index_to_raster(const int& pixel_index) const
	{
		const float y = floor(pixel_index / m_resolution.x);
		const float x = float(pixel_index) - y * m_resolution.x;
		return make_float2(x, y);
	}

	__host__ __device__ float3 PerspectiveCamera::raster_to_world(const float2& raster_xy) const
	{
		return t_raster_to_world(make_float3(raster_xy.x, raster_xy.y, 0.0f), true);
	}

	__host__ __device__ float2 PerspectiveCamera::world_to_raster(const float3& world_pt) const
	{
		float3 temp = t_world_to_raster(world_pt, true);
		return make_float2(temp.x, temp.y);
	}

	__host__ __device__ float3 PerspectiveCamera::world_to_ndc(const float3& world_pt) const
	{
		return t_world_to_screen(world_pt, true);
	}

	__host__ __device__ bool PerspectiveCamera::check_raster(const float2& raster_pos) const
	{
		return raster_pos.x >= 0 && raster_pos.x < m_resolution.x && raster_pos.y >= 0 && raster_pos.y < m_resolution.y;
	}

	__host__ __device__ Ray PerspectiveCamera::generate_ray(float x, float y) const
	{
		float3 pras = make_float3(x, y, 0.0f);
		float3 p_camera = t_raster_to_camera(pras, true);
		Ray r = Ray(make_float3(0.0f), normalize(p_camera), 0.0f);
		Ray ret = t_camera_to_world(r);
		return ret;
	}
}
