// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Dus'li

#pragma once

#include <cstddef>

#if defined(__BYTE_ORDER__)
  #if (__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__)
    #define make_rgba(r, g, b, a) make_uchar4(a, b, g, r)
  #else
    #define make_rgba(r, g, b, a) make_uchar4(r, g, b, a)
  #endif
#else // !defined(__BYTE_ORDER__)
  #error __BYTE_ORDER__ must be defined
#endif

#define call_any__ __host__ __device__

namespace device {

template <typename T> struct __VecWrapper;

template <> struct __VecWrapper<float3> {
	static constexpr size_t nmemb = 3;

	static inline call_any__ float &ith(float3 &v, size_t i)
	{
		return ((float *)&v)[i];
	}

	static inline call_any__ const float &ith(const float3 &v, size_t i)
	{
		return ((const float *)&v)[i];
	}
};

template <> struct __VecWrapper<float4> {
	static constexpr size_t nmemb = 4;

	static inline call_any__ float &ith(float4 &v, size_t i)
	{
		return ((float *)&v)[i];
	}

	static inline call_any__ const float &ith(const float4 &v, size_t i)
	{
		return ((const float *)&v)[i];
	}
};

template <typename T> static inline constexpr call_any__ T make_vec()
{
	if constexpr (__VecWrapper<T>::nmemb == 3)
		return make_float3(0, 0, 0);
	else
		return make_float4(0, 0, 0, 0);
}

template <typename T, typename O>
static inline call_any__ T cwise_binary_op(const T &lhs, const T &rhs, O op)
{
	T ret = make_vec<T>();
	for (size_t i = 0; i < __VecWrapper<T>::nmemb; ++i) {
		__VecWrapper<T>::ith(ret, i) = op(__VecWrapper<T>::ith(lhs, i),
		    __VecWrapper<T>::ith(rhs, i));
	}

	return ret;
}

template <typename T, typename O>
static inline call_any__ T scalar_binary_op_lhs(float lhs, const T &rhs, O op)
{
	T ret;
	for (size_t i = 0; i < __VecWrapper<T>::nmemb; ++i) {
		__VecWrapper<T>::ith(ret, i) = op(lhs,
		    __VecWrapper<T>::ith(rhs, i));
	}

	return ret;
}

template <typename T, typename O>
static inline call_any__ T scalar_binary_op_rhs(const T &lhs, float rhs, O op)
{
	T ret;
	for (size_t i = 0; i < __VecWrapper<T>::nmemb; ++i) {
		__VecWrapper<T>::ith(ret, i) = op(__VecWrapper<T>::ith(lhs, i),
		    rhs);
	}

	return ret;
}

#define VECOPS_OPERATORS(X) \
	X(+)                \
	X(-)                \
	X(*)                \
	X(/)

#define vecops_make_cwise_binary_operator(_op)                              \
	template <typename T>                                               \
	static inline call_any__ T operator _op(const T &lhs, const T &rhs) \
	{                                                                   \
		return cwise_binary_op(lhs,                                 \
		    rhs,                                                    \
		    [] call_any__(float l, float r) { return l _op r; });   \
	}

#define vecops_make_scalar_binary_operators(_op)                          \
	template <typename T>                                             \
	static inline call_any__ T operator _op(float lhs, const T &rhs)  \
	{                                                                 \
		return scalar_binary_op_lhs(lhs,                          \
		    rhs,                                                  \
		    [] call_any__(float l, float r) { return l _op r; }); \
	}                                                                 \
	template <typename T>                                             \
	static inline call_any__ T operator _op(const T &lhs, float rhs)  \
	{                                                                 \
		return scalar_binary_op_rhs(lhs,                          \
		    rhs,                                                  \
		    [] call_any__(float l, float r) { return l _op r; }); \
	}

VECOPS_OPERATORS(vecops_make_cwise_binary_operator)
VECOPS_OPERATORS(vecops_make_scalar_binary_operators)

template <typename T>
static inline call_any__ float dot(const T &lhs, const T &rhs)
{
	float ret = 0;
	for (size_t i = 0; i < __VecWrapper<T>::nmemb; ++i) {
		ret += __VecWrapper<T>::ith(lhs, i) *
		    __VecWrapper<T>::ith(rhs, i);
	}

	return ret;
}

template <typename T> static inline call_any__ T normalize(const T &v)
{
	return v * rsqrtf(dot(v, v));
}

template <typename T>
static inline call_any__ T clamp(const T &v, float lo, float up)
{
	T ret = make_vec<T>();
	for (size_t i = 0; i < __VecWrapper<T>::nmemb; ++i) {
		__VecWrapper<T>::ith(ret, i) = fmaxf(lo,
		    fminf(up, __VecWrapper<T>::ith(v, i)));
	}

	return ret;
}

static inline call_any__ float3 f4_xyz(float4 v)
{
	return make_float3(v.x, v.y, v.z);
}

static inline call_any__ float3 cross(float3 lhs, float3 rhs)
{
	return make_float3(lhs.y * rhs.z - lhs.z * rhs.y,
	    lhs.z * rhs.x - lhs.x * rhs.z,
	    lhs.x * rhs.y - lhs.y * rhs.x);
}

} // namespace device
