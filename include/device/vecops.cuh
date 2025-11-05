// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Dus'li

#pragma once

#include "pproc.hxx"

#if defined(__BYTE_ORDER__)
  #if (__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__)
    #define make_rgba(r, g, b, a) make_uchar4(a, b, g, r)
  #else
    #define make_rgba(r, g, b, a) make_uchar4(r, g, b, a)
  #endif
#else // !defined(__BYTE_ORDER__)
  #error __BYTE_ORDER__ must be defined
#endif

static inline __device__ __host__ float3 f4_xyz(float4 v)
{
	return make_float3(v.x, v.y, v.z);
}

static inline __device__ __host__ float3 f3_neg(float3 v)
{
	return make_float3(-v.x, -v.y, -v.z);
}

static inline __device__ __host__ float f3_dot(float3 a, float3 b)
{
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

static inline __device__ __host__ float3 f3_mul(float a, float3 v)
{
	return make_float3(a * v.x, a * v.y, a * v.z);
}

static inline __device__ __host__ float3 f3_norm(float3 v)
{
	float tmp = rsqrtf(f3_dot(v, v));
	return f3_mul(tmp, v);
}

#define f3_cwise_binop__(name, op)                                      \
	static inline __device__ __host__ float3 CONCAT(f3_, name) /**/ \
	    (float3 a, float3 b)                                        \
	{                                                               \
		return make_float3(a.x op b.x, a.y op b.y, a.z op b.z); \
	}

// clang-format off
f3_cwise_binop__(sub, -)
f3_cwise_binop__(add, +)
f3_cwise_binop__(mul_cwise, *)
