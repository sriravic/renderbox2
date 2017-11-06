
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

#ifndef GLOBAL_TYPES_H
#define GLOBAL_TYPES_H

// Application specific headers.

// Cuda specific headers.
#include <vector_types.h>

// Standard c++ headers.
#include <cstdint>

namespace renderbox2
{
	
	//
	// Gloabal precision type.
	//

	typedef std::uint32_t GIndexType;					// Type 1 indexing with 32bits.

	typedef uint2 GIndexVec2Type;						// Indexing using a 2 tuple.

	typedef uint3 GIndexVec3Type;						// Indexing using a 3 tuple.

	typedef uint4 GIndexVec4Type;						// Indexing using a 4 tuple.

}			// !namespace renderbox2

#endif		// !GLOBAL_TYPES_H
