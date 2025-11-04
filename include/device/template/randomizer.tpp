// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Dus'li

#include <algorithm>

namespace device {

template <typename C, typename T, typename G>
static int __rand_fill(C *dest, size_t n, T lo, T up, builder<C, T> fn)
{
	std::random_device rd;

	std::mt19937 gen(rd());
	cudaError_t  ret;

	G dist(lo, up);

	auto   next = [&dist, &gen]() { return dist(gen); };
	size_t size = sizeof(C) * n;

	std::vector<C> tmp(n);
	std::generate(tmp.begin(), tmp.end(), [&fn, &next]() {
		return fn(next);
	});

	ret = cudaMemcpy(dest, tmp.data(), size, cudaMemcpyHostToDevice);
	if (ret != cudaSuccess)
		return -RAND_ERR_COPY;

	return 0;
}

} // namespace device
