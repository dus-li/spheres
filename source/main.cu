// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Dus'li

#include <exception>
#include <iostream>

#include "device/appstate.cuh"

int main()
{
	try {
		device::AppState app(5, 1000, 1);
		app.run();
	} catch (const std::exception &e) {
		std::cerr << "Error: " << e.what() << std::endl;
		return 1;
	}

	return 0;
}
