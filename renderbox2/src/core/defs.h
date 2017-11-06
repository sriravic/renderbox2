
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

#ifndef DEFS_H
#define DEFS_H

// Application specific headers.

// Cuda specific headers.
#include <cuda.h>
#include <cuda_runtime.h>

// Standard c++ headers.
#include <cassert>
#include <stdio.h>

#define EPSILON        0.000001f
#define ZERO_TOLERANCE 0.01f
#define M_PI           3.14159265358979323846f
#define INV_PI         0.31830988618379067154f
#define INV_TWOPI      0.15915494309189533577f
#define INV_FOURPI     0.07957747154594766788f

//
// SAFE_RELEASE macro used for deleting pointers in a safe way.
//


#ifndef SAFE_RELEASE
#define SAFE_RELEASE_ARRAY(A) if(A != nullptr) { delete[] A; A = nullptr; }
#define SAFE_RELEASE(A) if(A != nullptr) { delete A; A = nullptr; }
#endif


//
// CheckCuda macro encapsulates checking cuda calls for errors. Handy for debugging when something fails somewhere.
//

#define checkCuda(call)                                     \
{                                                           \
  cudaError_t cucheck_err = (call);                         \
  if(cucheck_err != cudaSuccess) {                          \
    const char *err_str = cudaGetErrorString(cucheck_err);  \
    printf("%s (%d): %s\n", __FILE__, __LINE__, err_str);   \
    assert(0);                                              \
    }                                                       \
}


//
// CUDA_SAFE_RELEASE macro enables safe freeing up of device memory.
//

#ifndef CUDA_SAFE_RELEASE
#define CUDA_SAFE_RELEASE(A) if(A != nullptr) { checkCuda(cudaFree(A)); A = nullptr; }
#endif


//
// Macro to determine for which architecture the code has been compiled.
// User has to set and must not expect cudacc to set this variable.
// CMake can also set these variables.
// {USE_KERNEL_FERMI, USE_KERNEL_KEPLER}
// NOTE: These variables only affect the traversal kernel used and not anything else within the system.
//       Hence the application might crash when using code generated for different architectures together.

//#define USE_KERNEL_KEPLER 1

//
// RendererType enums
//

enum class RendererType
{
	RENDERER_SAMPLER
};


//
// RenderingMethod enums
//

enum class RenderMethod
{
	RENDER_METHOD_PT,
	RENDER_METHOD_BDPT
};

#endif
