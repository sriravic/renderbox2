
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
#include <core/application.h>
#include <renderbox2.h>

// Cuda specific headers.

// Standard c++ headers.
#include <string>
#include <iostream>

using namespace renderbox2;
using namespace std;

void print_license()
{
	cout << "renderbox2 Copyright(C) 2014 - Srinath Ravichandran" << endl;
	cout << "This program comes with ABSOLUTELY NO WARRANTY; for details type `show w'. " << endl;
	cout << "This is free software, and you are welcome to redistribute it" << endl;
	cout << "under certain conditions." << endl;
}

int main(int argc, char** argv)
{
	cout << g_application_string << endl;
	cout << g_application_version << endl;

	// Process application.
	Application my_application;

	if (!my_application.init())
	{
		std::cerr << "Error initializing application" << std::endl;
		return -1;
	}
	
	if (!my_application.gui())
	{
		if (!my_application.prepare())
		{
			std::cerr << "Error preparing application" << std::endl;
			return -1;
		}

		if (!my_application.run())
		{
			std::cerr << "Error running the application" << std::endl;
			return -1;
		}

		if (!my_application.shutdown())
		{
			std::cerr << "Error shutting down the application" << std::endl;
			return -1;
		}
	}
	else
	{
		// Start gui.
	}
	
	return 0;
}
