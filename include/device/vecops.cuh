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

// clang-format off
/**
 * X macro describing scalar and component-wise operations.
 *
 * This is a table-like structure describing operations on vector types. Each
 * row corresponds to an operation. First column contains the operation name,
 * whereas the second one is the actual operator used in the operation.
 *
 * Using data from this table, a macro-based engine automatically generates
 * scalar and component-wise operations for types float3 and float4.
 */
#define SC_CWISE_OPERATIONS(X)   \
       /*  OP NAME  OPERATOR  */ \
	X(   add,       +    )   \
	X(   sub,       -    )   \
	X(   mul,       *    )
// clang-format on

/** Pseudodecorator denoting that routine is callable both from host and dev. */
#define call_any__ __host__ __device__

#define MAKE_F3_SC_OP_(_name, _op)                                   \
	static inline call_any__ float3 CONCAT(f3_sc_, _name) /**/   \
	    (float a, float3 v)                                      \
	{                                                            \
		return make_float3(a _op v.x, a _op v.y, a _op v.z); \
	}

#define MAKE_F3_CWISE_OP_(_name, _op)                                      \
	static inline call_any__ float3 CONCAT(f3_cwise_, _name) /**/      \
	    (float3 a, float3 b)                                           \
	{                                                                  \
		return make_float3(a.x _op b.x, a.y _op b.y, a.z _op b.z); \
	}

#define MAKE_F4_SC_OP_(_name, _op)                                 \
	static inline call_any__ float4 CONCAT(f4_sc_, _name) /**/ \
	    (float a, float4 v)                                    \
	{                                                          \
		return make_float4(/**/                            \
		    a _op v.x,                                     \
		    a _op v.y,                                     \
		    a _op v.z,                                     \
		    a _op v.w);                                    \
	}

#define MAKE_F4_CWISE_OP_(_name, _op)                                 \
	static inline call_any__ float4 CONCAT(f4_cwise_, _name) /**/ \
	    (float4 a, float4 b)                                      \
	{                                                             \
		return make_float4(/**/                               \
		    a.x _op b.x,                                      \
		    a.y _op b.y,                                      \
		    a.z _op b.z,                                      \
		    a.w _op b.w);                                     \
	}

// Define all operations.
SC_CWISE_OPERATIONS(MAKE_F3_SC_OP_)
SC_CWISE_OPERATIONS(MAKE_F3_CWISE_OP_)
SC_CWISE_OPERATIONS(MAKE_F4_SC_OP_)
SC_CWISE_OPERATIONS(MAKE_F4_CWISE_OP_)

// This could be done by the X macro but at this point automating it would take
// longer than writing those few lines lol.
//
// Could have also been an __attribute__((alias)) if ops were declared with
// C linkage, but apparently NVCC/G++ does not like static inline aliases in
// headers and I AM NOT DEBUGGING THAT!
#define f3_add f3_cwise_add
#define f3_sub f3_cwise_sub
#define f4_add f4_cwise_add
#define f4_sub f4_cwise_sub

static inline call_any__ float3 f4_xyz(float4 v)
{
	return make_float3(v.x, v.y, v.z);
}

static inline call_any__ float3 f3_neg(float3 v)
{
	return make_float3(-v.x, -v.y, -v.z);
}

static inline call_any__ float f3_dot(float3 a, float3 b)
{
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

static inline call_any__ float3 f3_norm(float3 v)
{
	float tmp = rsqrtf(f3_dot(v, v));
	return f3_sc_mul(tmp, v);
}
