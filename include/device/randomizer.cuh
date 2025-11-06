// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Dus'li

#pragma once

#include <cstddef>
#include <functional>
#include <random>
#include <stdexcept>

#include "pproc.hxx"

namespace device {

#define RAND_ERR_BOUNDS (1)
#define RAND_ERR_COPY   (2)

template <typename C, typename T>
using builder = std::function<C(std::function<T()>)>;

/**
 * Fill a device array of a product type with random values from a range.
 * @tparam C     Type of a single array element.
 * @tparam T     Type from which @a C is constructible.
 * @tparam G     Generator type.
 * @param  dest  Destination buffer (must be a device pointer).
 * @param  n     Number of elements in the buffer.
 * @param  lo    Lower bound for acceptable values.
 * @param  up    Upper bound for acceptable values.
 * @param  fn    Builder function.
 *
 * @throw std::runtime_error If value can't be copied to the device.
 *
 * @see rand_fillf
 * @see rand_filli
 */
template <typename C, typename T, typename G>
int __rand_fill(C *dest, size_t n, T lo, T up, builder<C, T> fn);

template <typename C, typename T>
static inline int _rand_fillf(C *dest, size_t n, T lo, T up, builder<C, T> fn)
{
	using G = std::uniform_real_distribution<T>;
	return __rand_fill<C, T, G>(dest, n, lo, up, fn);
}

template <typename C, typename T>
static inline int _rand_filli(C *dest, size_t n, T lo, T up, builder<C, T> fn)
{
	using G = std::uniform_int_distribution<T>;

	// Unlike uniform_float_distribution(), the integer one produces values
	// from a closed interval, rather than a one-side open one. To keep
	// interface uniform, we will subtract one, but first its good to check
	// if we can do that :DD
	if (up - 1 < lo || up < up - 1)
		return -RAND_ERR_BOUNDS;

	return __rand_fill<C, T, G>(dest, n, lo, up - 1, fn);
}

static inline std::runtime_error rand_err_cons(int code)
{
	return std::runtime_error(/**/
	    code == -RAND_ERR_BOUNDS   ? "Incorrect bounds" :
		code == -RAND_ERR_COPY ? "Failed to copy to device" :
					 "Unknown error");
}

/**
 * Declare a function with custom name that can fill a product type array.
 * @param name  Name of the function.
 * @param be    Backend (@ref _rand_fillf or @ref _rand_filli).
 * @param otype Output product type.
 * @param itype Input type.
 * @param body  Body of statement expr computing an instance of @a otype.
 *
 * The @a body param can be any sequence of statements, ending with an
 * expression that evaluates to the type @a otype. It will be repetedly called
 * to populate elements of the output array. Inside the body one may use an
 * identifier @a next which is a callable type that yields a random value of
 * type @a itype.
 */
#define DECLARE_NAMED_FILLER(name, be, otype, itype, body)                   \
	static inline void name(otype *dest, size_t n, itype lo, itype up)   \
	{                                                                    \
		builder<otype, itype> fn = [](std::function<itype()> next) { \
			return ({ body });                                   \
		};                                                           \
		for (int _r = be(dest, n, lo, up, fn); _r;)                  \
			throw rand_err_cons(_r);                             \
	}

/**
 * Declare a function that can fill a product type array.
 * @see DECLARE_NAMED_FILLER
 */
#define DECLARE_FILLER(be, otype, ...) \
	DECLARE_NAMED_FILLER(CONCAT(rand_fill_, otype), be, otype, __VA_ARGS__)

/**
 * Declare a function that can fill an array with random data.
 * @param be   Backend (@ref _rand_fillf or @ref _rand_filli).
 * @param type Type of the array.
 *
 * This is a special case of @ref DECLARE_FILLER when @a otype and @a itype are
 * the same type.
 */
#define DECLARE_IDENTITY_FILLER(be, type) \
	DECLARE_FILLER(be, type, type, next();)

DECLARE_FILLER(_rand_fillf, float4, float,
    make_float4(next(), next(), next(), next()); /**/
)

DECLARE_NAMED_FILLER(rand_fill_float4_0, _rand_fillf, float4, float,
    make_float4(next(), next(), next(), 0.0f); /**/
)

DECLARE_NAMED_FILLER(rand_fill_float4_1, _rand_fillf, float4, float,
    make_float4(next(), next(), next(), 1.0f); /**/
)

DECLARE_IDENTITY_FILLER(_rand_fillf, float)
DECLARE_IDENTITY_FILLER(_rand_filli, size_t)

} // namespace device

#include "template/randomizer.tpp"
