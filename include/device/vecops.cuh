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

#define op_add__ +
#define op_mul__ *
#define op_sub__ -

#define f3_scalar_op__(name, op)                                        \
	static inline __device__ __host__ float3 CONCAT(f3_, name) /**/ \
	    (float a, float3 v)                                         \
	{                                                               \
		return make_float3(a op v.x, a op v.y, a op v.z);       \
	}

f3_scalar_op__(mul, op_mul__)
f3_scalar_op__(add_sc, op_add__)

#define f3_cwise_binop__(name, op)                                      \
	static inline __device__ __host__ float3 CONCAT(f3_, name) /**/ \
	    (float3 a, float3 b)                                        \
	{                                                               \
		return make_float3(a.x op b.x, a.y op b.y, a.z op b.z); \
	}

f3_cwise_binop__(sub, op_sub__)
f3_cwise_binop__(add, op_add__)
f3_cwise_binop__(mul_cwise, op_mul__)

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

static inline __device__ __host__ float3 f3_norm(float3 v)
{
	float tmp = rsqrtf(f3_dot(v, v));
	return f3_mul(tmp, v);
}
