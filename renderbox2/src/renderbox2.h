
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

#ifndef RENDERBOX2_H
#define RENDERBOX2_H

// Application specific headers.

// Cuda specific headers.

// Standard c++ headers.
#include <string>

namespace renderbox2
{
	//
	// Globally defined data that the entire application can use.
	//

	static const std::string g_application_string("RenderBox");
	static const std::string g_application_version("Version 2.0");
	static const std::string g_application_developer("Srinath Ravichandran");

}			// !namespace renderbox2

#endif		// !RENDERBOX2_H
