// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Dus'li

#pragma once

#include <chrono>
#include <thread>

using Clock = std::chrono::high_resolution_clock;

namespace host {

class FPSLimiter {
	std::chrono::milliseconds      period;
	std::chrono::time_point<Clock> prev;
	std::chrono::time_point<Clock> next;

  public:
	FPSLimiter(unsigned fps)
	    : period(1000 / fps)
	    , prev(Clock::now())
	    , next(prev + period)
	{
	}

	int wait()
	{
		std::this_thread::sleep_until(next);

		auto now = Clock::now();
		auto ret = now - prev;

		prev = next;
		next = now + period;

		return std::chrono::duration_cast<std::chrono::milliseconds>(
		    ret)
		    .count();
	}
};

} // namespace host
