
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

#ifndef TIMER_H
#define TIMER_H

// Application specific headers.
#include <core/defs.h>

// Cuda specific headers.
#include <cuda_runtime.h>

// Standard c++ headers.
#include <iostream>
#include <string>

namespace renderbox2
{

	//
	// Timer claass for timing cuda kernel calls.
	//

	class Timer
	{
	public:
		
		Timer(std::string name) : _name(name), time(0)
		{
			checkCuda(cudaEventCreate(&_start));
			checkCuda(cudaEventCreate(&_stop));
		}

		void start()
		{
			checkCuda(cudaEventRecord(_start, 0));
		}

		void start(cudaStream_t stream)
		{
			checkCuda(cudaEventRecord(_start, stream));
		}

		void stop()
		{
			float temp = 0.0f;
			checkCuda(cudaEventRecord(_stop, 0));
			checkCuda(cudaEventSynchronize(_stop));
			checkCuda(cudaEventElapsedTime(&temp, _start, _stop));
			time += temp;
		}

		void stop(cudaStream_t stream)
		{
			float temp = 0.0f;
			checkCuda(cudaEventRecord(_stop, stream));
			checkCuda(cudaEventSynchronize(_stop));
			checkCuda(cudaEventElapsedTime(&temp, _start, _stop));
			time += temp;
		}

		void reset() { time = 0.0f; }

		void print() const
		{
			std::cout << "Timer :: Elapsed: " << _name << "    " << time << " ms" << std::endl;
		}

		float get_ms() const { return time; };

	private:
		cudaEvent_t _start, _stop;
		std::string _name;
		float       time;
	};

}			// !namespace renderbox2

#endif		// !TIMER_H