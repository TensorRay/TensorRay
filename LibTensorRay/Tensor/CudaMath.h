/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */


/*
 *  This file implements common mathematical operations on vector types
 *  (float3, float4 etc.) since these are not provided as standard by CUDA.
 *
 *  The syntax is modeled on the Cg standard library.
 *
 *  This is part of the Helper library includes
 *
 *    Thanks to Linh Hah for additions and fixes.
 */

typedef unsigned int uint;
typedef unsigned short ushort;

////////////////////////////////////////////////////////////////////////////////
// host implementations of CUDA functions
////////////////////////////////////////////////////////////////////////////////

inline float fminf(float a, float b)
{
    return a < b ? a : b;
}

inline float fmaxf(float a, float b)
{
    return a > b ? a : b;
}

inline int max(int a, int b)
{
    return a > b ? a : b;
}

inline int min(int a, int b)
{
    return a < b ? a : b;
}

inline float rsqrtf(float x)
{
    return 1.0f / sqrtf(x);
}


////////////////////////////////////////////////////////////////////////////////
// constructors
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ float2 make_float2(float s)
{
    return make_float2(s, s);
}
inline __host__ __device__ float2 make_float2(float3 a)
{
    return make_float2(a.x, a.y);
}
inline __host__ __device__ float2 make_float2(int2 a)
{
    return make_float2(float(a.x), float(a.y));
}
inline __host__ __device__ float2 make_float2(uint2 a)
{
    return make_float2(float(a.x), float(a.y));
}

inline __host__ __device__ int2 make_int2(int s)
{
    return make_int2(s, s);
}
inline __host__ __device__ int2 make_int2(int3 a)
{
    return make_int2(a.x, a.y);
}
inline __host__ __device__ int2 make_int2(uint2 a)
{
    return make_int2(int(a.x), int(a.y));
}
inline __host__ __device__ int2 make_int2(float2 a)
{
    return make_int2(int(a.x), int(a.y));
}

inline __host__ __device__ uint2 make_uint2(uint s)
{
    return make_uint2(s, s);
}
inline __host__ __device__ uint2 make_uint2(uint3 a)
{
    return make_uint2(a.x, a.y);
}
inline __host__ __device__ uint2 make_uint2(int2 a)
{
    return make_uint2(uint(a.x), uint(a.y));
}

inline __host__ __device__ float3 make_float3(float s)
{
    return make_float3(s, s, s);
}
inline __host__ __device__ float3 make_float3(float2 a)
{
    return make_float3(a.x, a.y, 0.0f);
}
inline __host__ __device__ float3 make_float3(float2 a, float s)
{
    return make_float3(a.x, a.y, s);
}
inline __host__ __device__ float3 make_float3(float4 a)
{
    return make_float3(a.x, a.y, a.z);
}
inline __host__ __device__ float3 make_float3(int3 a)
{
    return make_float3(float(a.x), float(a.y), float(a.z));
}
inline __host__ __device__ float3 make_float3(uint3 a)
{
    return make_float3(float(a.x), float(a.y), float(a.z));
}

inline __host__ __device__ int3 make_int3(int s)
{
    return make_int3(s, s, s);
}
inline __host__ __device__ int3 make_int3(int2 a)
{
    return make_int3(a.x, a.y, 0);
}
inline __host__ __device__ int3 make_int3(int2 a, int s)
{
    return make_int3(a.x, a.y, s);
}
inline __host__ __device__ int3 make_int3(uint3 a)
{
    return make_int3(int(a.x), int(a.y), int(a.z));
}
inline __host__ __device__ int3 make_int3(float3 a)
{
    return make_int3(int(a.x), int(a.y), int(a.z));
}

inline __host__ __device__ uint3 make_uint3(uint s)
{
    return make_uint3(s, s, s);
}
inline __host__ __device__ uint3 make_uint3(uint2 a)
{
    return make_uint3(a.x, a.y, 0);
}
inline __host__ __device__ uint3 make_uint3(uint2 a, uint s)
{
    return make_uint3(a.x, a.y, s);
}
inline __host__ __device__ uint3 make_uint3(uint4 a)
{
    return make_uint3(a.x, a.y, a.z);
}
inline __host__ __device__ uint3 make_uint3(int3 a)
{
    return make_uint3(uint(a.x), uint(a.y), uint(a.z));
}

inline __host__ __device__ float4 make_float4(float s)
{
    return make_float4(s, s, s, s);
}
inline __host__ __device__ float4 make_float4(float3 a)
{
    return make_float4(a.x, a.y, a.z, 0.0f);
}
inline __host__ __device__ float4 make_float4(float3 a, float w)
{
    return make_float4(a.x, a.y, a.z, w);
}
inline __host__ __device__ float4 make_float4(int4 a)
{
    return make_float4(float(a.x), float(a.y), float(a.z), float(a.w));
}
inline __host__ __device__ float4 make_float4(uint4 a)
{
    return make_float4(float(a.x), float(a.y), float(a.z), float(a.w));
}

inline __host__ __device__ int4 make_int4(int s)
{
    return make_int4(s, s, s, s);
}
inline __host__ __device__ int4 make_int4(int3 a)
{
    return make_int4(a.x, a.y, a.z, 0);
}
inline __host__ __device__ int4 make_int4(int3 a, int w)
{
    return make_int4(a.x, a.y, a.z, w);
}
inline __host__ __device__ int4 make_int4(uint4 a)
{
    return make_int4(int(a.x), int(a.y), int(a.z), int(a.w));
}
inline __host__ __device__ int4 make_int4(float4 a)
{
    return make_int4(int(a.x), int(a.y), int(a.z), int(a.w));
}


inline __host__ __device__ uint4 make_uint4(uint s)
{
    return make_uint4(s, s, s, s);
}
inline __host__ __device__ uint4 make_uint4(uint3 a)
{
    return make_uint4(a.x, a.y, a.z, 0);
}
inline __host__ __device__ uint4 make_uint4(uint3 a, uint w)
{
    return make_uint4(a.x, a.y, a.z, w);
}
inline __host__ __device__ uint4 make_uint4(int4 a)
{
    return make_uint4(uint(a.x), uint(a.y), uint(a.z), uint(a.w));
}

////////////////////////////////////////////////////////////////////////////////
// negate
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ float2 operator-(const float2 &a)
{
    return make_float2(-a.x, -a.y);
}
inline __host__ __device__ int2 operator-(const int2 &a)
{
    return make_int2(-a.x, -a.y);
}
inline __host__ __device__ float3 operator-(const float3 &a)
{
    return make_float3(-a.x, -a.y, -a.z);
}
inline __host__ __device__ int3 operator-(const int3 &a)
{
    return make_int3(-a.x, -a.y, -a.z);
}
inline __host__ __device__ float4 operator-(const float4 &a)
{
    return make_float4(-a.x, -a.y, -a.z, -a.w);
}
inline __host__ __device__ int4 operator-(const int4 &a)
{
    return make_int4(-a.x, -a.y, -a.z, -a.w);
}


////////////////////////////////////////////////////////////////////////////////
// Arithmetic
////////////////////////////////////////////////////////////////////////////////
#define ARITH_OP_TYPE(OP, TYPE)                                                   \
    inline __host__ __device__ TYPE##2 operator OP (TYPE##2 a, TYPE##2 b)         \
    {                                                                             \
        return make_##TYPE##2(a.x OP b.x, a.y OP b.y);                            \
    }                                                                             \
    inline __host__ __device__ TYPE##2 operator OP (TYPE##2 a, TYPE b)            \
    {                                                                             \
        return make_##TYPE##2(a.x OP b, a.y OP b);                                \
    }                                                                             \
    inline __host__ __device__ TYPE##2 operator OP (TYPE b, TYPE##2 a)            \
    {                                                                             \
        return make_##TYPE##2(b OP a.x, b OP a.y);                                \
    }                                                                             \
    inline __host__ __device__ void operator OP##= (TYPE##2 &a, TYPE##2 b)        \
    {                                                                             \
        a.x OP##= b.x;                                                            \
        a.y OP##= b.y;                                                            \
    }                                                                             \
    inline __host__ __device__ void operator OP##= (TYPE##2 &a, TYPE b)           \
    {                                                                             \
        a.x OP##= b;                                                              \
        a.y OP##= b;                                                              \
    }                                                                             \
    inline __host__ __device__ TYPE##3 operator OP (TYPE##3 a, TYPE##3 b)         \
    {                                                                             \
        return make_##TYPE##3(a.x OP b.x, a.y OP b.y, a.z OP b.z);                \
    }                                                                             \
    inline __host__ __device__ TYPE##3 operator OP (TYPE##3 a, TYPE b)            \
    {                                                                             \
        return make_##TYPE##3(a.x OP b, a.y OP b, a.z OP b);                      \
    }                                                                             \
    inline __host__ __device__ TYPE##3 operator OP (TYPE b, TYPE##3 a)            \
    {                                                                             \
        return make_##TYPE##3(b OP a.x, b OP a.y, b OP a.z);                      \
    }                                                                             \
    inline __host__ __device__ void operator OP##= (TYPE##3 &a, TYPE##3 b)        \
    {                                                                             \
        a.x OP##= b.x;                                                            \
        a.y OP##= b.y;                                                            \
        a.z OP##= b.z;                                                            \
    }                                                                             \
    inline __host__ __device__ void operator OP##= (TYPE##3 &a, TYPE b)           \
    {                                                                             \
        a.x OP##= b;                                                              \
        a.y OP##= b;                                                              \
        a.z OP##= b;                                                              \
    }                                                                             \
    inline __host__ __device__ TYPE##4 operator OP (TYPE##4 a, TYPE##4 b)         \
    {                                                                             \
        return make_##TYPE##4(a.x OP b.x, a.y OP b.y, a.z OP b.z,  a.w OP b.w);   \
    }                                                                             \
    inline __host__ __device__ TYPE##4 operator OP (TYPE##4 a, TYPE b)            \
    {                                                                             \
        return make_##TYPE##4(a.x OP b, a.y OP b, a.z OP b, a.w OP b);            \
    }                                                                             \
    inline __host__ __device__ TYPE##4 operator OP (TYPE b, TYPE##4 a)            \
    {                                                                             \
        return make_##TYPE##4(b OP a.x, b OP a.y, b OP a.z, b OP a.w);            \
    }                                                                             \
    inline __host__ __device__ void operator OP##= (TYPE##4 &a, TYPE##4 b)        \
    {                                                                             \
        a.x OP##= b.x;                                                            \
        a.y OP##= b.y;                                                            \
        a.z OP##= b.z;                                                            \
        a.w OP##= b.w;                                                            \
    }                                                                             \
    inline __host__ __device__ void operator OP##= (TYPE##4 &a, TYPE b)           \
    {                                                                             \
        a.x OP##= b;                                                              \
        a.y OP##= b;                                                              \
        a.z OP##= b;                                                              \
        a.w OP##= b;                                                              \
    }                                                                             \


#define ARITH_TYPE(TYPE)           \
    ARITH_OP_TYPE(+, TYPE)         \
    ARITH_OP_TYPE(-, TYPE)         \
    ARITH_OP_TYPE(*, TYPE)         \
    ARITH_OP_TYPE(/, TYPE)         \

ARITH_TYPE(float)
ARITH_TYPE(int)
ARITH_TYPE(uint)
ARITH_TYPE(double)


#define ARITH_OP_TYPE_2(OP, TYPE, TYPE2)                                          \
    inline __host__ __device__ float2 operator OP (TYPE##2 a, TYPE2##2 b)         \
    {                                                                             \
        return make_float2(a.x OP b.x, a.y OP b.y);                               \
    }                                                                             \
    inline __host__ __device__ float2 operator OP (TYPE a, TYPE2##2 b)            \
    {                                                                             \
        return make_float2(a OP b.x, a OP b.y);                                   \
    }                                                                             \
    inline __host__ __device__ float2 operator OP (TYPE##2 a, TYPE2 b)            \
    {                                                                             \
        return make_float2(a.x OP b, a.y OP b);                                   \
    }                                                                             \
    inline __host__ __device__ float3 operator OP (TYPE##3 a, TYPE2##3 b)         \
    {                                                                             \
        return make_float3(a.x OP b.x, a.y OP b.y, a.z OP b.z);                   \
    }                                                                             \
    inline __host__ __device__ float3 operator OP (TYPE a, TYPE2##3 b)            \
    {                                                                             \
        return make_float3(a OP b.x, a OP b.y, a OP b.z);                         \
    }                                                                             \
    inline __host__ __device__ float3 operator OP (TYPE##3 a, TYPE2 b)            \
    {                                                                             \
        return make_float3(a.x OP b, a.y OP b, a.z OP b);                         \
    }                                                                             \
    inline __host__ __device__ float4 operator OP (TYPE##4 a, TYPE2##4 b)         \
    {                                                                             \
        return make_float4(a.x OP b.x, a.y OP b.y, a.z OP b.z, a.w OP b.w);       \
    }                                                                             \
    inline __host__ __device__ float4 operator OP (TYPE a, TYPE2##4 b)            \
    {                                                                             \
        return make_float4(a OP b.x, a OP b.y, a OP b.z, a OP b.w);               \
    }                                                                             \
    inline __host__ __device__ float4 operator OP (TYPE##4 a, TYPE2 b)            \
    {                                                                             \
        return make_float4(a.x OP b, a.y OP b, a.z OP b, a.w OP b);               \
    }                                                                             \

#define ARITH_TYPE_2(TYPE, TYPE2)    \
    ARITH_OP_TYPE_2(+, TYPE, TYPE2)  \
    ARITH_OP_TYPE_2(-, TYPE, TYPE2)  \
    ARITH_OP_TYPE_2(*, TYPE, TYPE2)  \
    ARITH_OP_TYPE_2(/, TYPE, TYPE2)  \
    ARITH_OP_TYPE_2(+, TYPE2, TYPE)  \
    ARITH_OP_TYPE_2(-, TYPE2, TYPE)  \
    ARITH_OP_TYPE_2(*, TYPE2, TYPE)  \
    ARITH_OP_TYPE_2(/, TYPE2, TYPE)  \

ARITH_TYPE_2(float, int)
ARITH_TYPE_2(float, uint)

////////////////////////////////////////////////////////////////////////////////
// Logical operators
////////////////////////////////////////////////////////////////////////////////

#define LOGICAL_OPERATOR(OP, TYPE)                                            \
    inline __host__ __device__ int2 operator OP (TYPE a, TYPE##2 b)        \
    {                                                                      \
        return make_int2(a OP b.x, a OP b.y);                              \
    }                                                                      \
    inline __host__ __device__ int3 operator OP (TYPE a, TYPE##3 b)        \
    {                                                                      \
        return make_int3(a OP b.x, a OP b.y, a OP b.z);                    \
    }                                                                      \
    inline __host__ __device__ int4 operator OP (TYPE a, TYPE##4 b)        \
    {                                                                      \
        return make_int4(a OP b.x, a OP b.y, a OP b.z, a OP b.w);          \
    }                                                                      \
    inline __host__ __device__ int2 operator OP (TYPE##2 a, TYPE b)        \
    {                                                                      \
        return make_int2(a.x OP b, a.y OP b);                              \
    }                                                                      \
    inline __host__ __device__ int3 operator OP (TYPE##3 a, TYPE b)        \
    {                                                                      \
        return make_int3(a.x OP b, a.y OP b, a.z OP b);                    \
    }                                                                      \
    inline __host__ __device__ int4 operator OP (TYPE##4 a, TYPE b)        \
    {                                                                      \
        return make_int4(a.x OP b, a.y OP b, a.z OP b, a.w OP b);          \
    }                                                                      \
    inline __host__ __device__ int2 operator OP (TYPE##2 a, TYPE##2 b)     \
    {                                                                      \
        return make_int2(a.x OP b.x, a.y OP b.y);                          \
    }                                                                      \
    inline __host__ __device__ int3 operator OP (TYPE##3 a, TYPE##3 b)     \
    {                                                                      \
        return make_int3(a.x OP b.x, a.y OP b.y, a.z OP b.z);              \
    }                                                                      \
    inline __host__ __device__ int4 operator OP (TYPE##4 a, TYPE##4 b)     \
    {                                                                      \
        return make_int4(a.x OP b.x, a.y OP b.y, a.z OP b.z, a.w OP b.w);  \
    }                                                                      \

#define LOGICAL_OPERATOR_TYPE(TYPE) \
    LOGICAL_OPERATOR(<, TYPE)       \
    LOGICAL_OPERATOR(<=, TYPE)      \
    LOGICAL_OPERATOR(>, TYPE)       \
    LOGICAL_OPERATOR(>=, TYPE)      \
    LOGICAL_OPERATOR(== , TYPE)     \
    LOGICAL_OPERATOR(!= , TYPE)     \

LOGICAL_OPERATOR_TYPE(float)
LOGICAL_OPERATOR_TYPE(double)
LOGICAL_OPERATOR_TYPE(int)
LOGICAL_OPERATOR_TYPE(uint)

LOGICAL_OPERATOR(&& , int)
LOGICAL_OPERATOR(|| , int)
LOGICAL_OPERATOR(&& , uint)
LOGICAL_OPERATOR(|| , uint)


////////////////////////////////////////////////////////////////////////////////
// min/max
////////////////////////////////////////////////////////////////////////////////

#define MIN_MAX_TYPE(FUNC, TYPE)                                                                     \
    inline __host__ __device__ TYPE##2 FUNC(TYPE##2 a, TYPE##2 b)                                 \
    {                                                                                             \
        return make_##TYPE##2(FUNC(a.x,b.x), FUNC(a.y,b.y));                                      \
    }                                                                                             \
    inline __host__ __device__ TYPE##3 FUNC(TYPE##3 a, TYPE##3 b)                                 \
    {                                                                                             \
        return make_##TYPE##3(FUNC(a.x,b.x), FUNC(a.y,b.y), FUNC(a.z,b.z));                       \
    }                                                                                             \
    inline __host__ __device__ TYPE##4 FUNC(TYPE##4 a, TYPE##4 b)                                 \
    {                                                                                             \
        return make_##TYPE##4(FUNC(a.x,b.x), FUNC(a.y,b.y), FUNC(a.z,b.z), FUNC(a.w,b.w));        \
    }                                                                                             \
                                                                                                  \
    inline __host__ __device__ TYPE##2 FUNC(TYPE##2 a, TYPE b)                                    \
    {                                                                                             \
        return make_##TYPE##2(FUNC(a.x, b), FUNC(a.y, b));                                        \
    }                                                                                             \
    inline __host__ __device__ TYPE##3 FUNC(TYPE##3 a, TYPE b)                                    \
    {                                                                                             \
        return make_##TYPE##3(FUNC(a.x, b), FUNC(a.y, b), FUNC(a.z, b));                          \
    }                                                                                             \
    inline __host__ __device__ TYPE##4 FUNC(TYPE##4 a, TYPE b)                                    \
    {                                                                                             \
        return make_##TYPE##4(FUNC(a.x, b), FUNC(a.y, b), FUNC(a.z, b), FUNC(a.w, b));            \
    }                                                                                             \
                                                                                                  \
    inline __host__ __device__ TYPE##2 FUNC(TYPE a, TYPE##2 b)                                    \
    {                                                                                             \
        return make_##TYPE##2(FUNC(a, b.x), FUNC(a, b.y));                                        \
    }                                                                                             \
    inline __host__ __device__ TYPE##3 FUNC(TYPE a, TYPE##3 b)                                    \
    {                                                                                             \
        return make_##TYPE##3(FUNC(a, b.x), FUNC(a, b.y), FUNC(a, b.z));                          \
    }                                                                                             \
    inline __host__ __device__ TYPE##4 FUNC(TYPE a, TYPE##4 b)                                    \
    {                                                                                             \
        return make_##TYPE##4(FUNC(a, b.x), FUNC(a, b.y), FUNC(a, b.z), FUNC(a, b.w));            \
    }                                                                                             \

MIN_MAX_TYPE(fmaxf, float)
MIN_MAX_TYPE(fminf, float)
MIN_MAX_TYPE(max, int)
MIN_MAX_TYPE(min, int)
MIN_MAX_TYPE(max, uint)
MIN_MAX_TYPE(min, uint)
MIN_MAX_TYPE(max, double)
MIN_MAX_TYPE(min, double)


////////////////////////////////////////////////////////////////////////////////
// lerp
// - linear interpolation between a and b, based on value t in [0, 1] range
////////////////////////////////////////////////////////////////////////////////

inline __device__ __host__ float lerp(float a, float b, float t)
{
    return a + t*(b-a);
}
inline __device__ __host__ float2 lerp(float2 a, float2 b, float t)
{
    return a + t*(b-a);
}
inline __device__ __host__ float3 lerp(float3 a, float3 b, float t)
{
    return a + t*(b-a);
}
inline __device__ __host__ float4 lerp(float4 a, float4 b, float t)
{
    return a + t*(b-a);
}

////////////////////////////////////////////////////////////////////////////////
// clamp
// - clamp the value v to be in the range [a, b]
////////////////////////////////////////////////////////////////////////////////

inline __device__ __host__ float clamp(float f, float a, float b)
{
    return fminf(a, fminf(f, b));
}
inline __device__ __host__ int clamp(int f, int a, int b)
{
    return max(a, min(f, b));
}
inline __device__ __host__ uint clamp(uint f, uint a, uint b)
{
    return max(a, min(f, b));
}

inline __device__ __host__ float2 clamp(float2 v, float a, float b)
{
    return make_float2(clamp(v.x, a, b), clamp(v.y, a, b));
}
inline __device__ __host__ float2 clamp(float2 v, float2 a, float2 b)
{
    return make_float2(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y));
}
inline __device__ __host__ float3 clamp(float3 v, float a, float b)
{
    return make_float3(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b));
}
inline __device__ __host__ float3 clamp(float3 v, float3 a, float3 b)
{
    return make_float3(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z));
}
inline __device__ __host__ float4 clamp(float4 v, float a, float b)
{
    return make_float4(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b), clamp(v.w, a, b));
}
inline __device__ __host__ float4 clamp(float4 v, float4 a, float4 b)
{
    return make_float4(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z), clamp(v.w, a.w, b.w));
}

inline __device__ __host__ int2 clamp(int2 v, int a, int b)
{
    return make_int2(clamp(v.x, a, b), clamp(v.y, a, b));
}
inline __device__ __host__ int2 clamp(int2 v, int2 a, int2 b)
{
    return make_int2(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y));
}
inline __device__ __host__ int3 clamp(int3 v, int a, int b)
{
    return make_int3(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b));
}
inline __device__ __host__ int3 clamp(int3 v, int3 a, int3 b)
{
    return make_int3(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z));
}
inline __device__ __host__ int4 clamp(int4 v, int a, int b)
{
    return make_int4(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b), clamp(v.w, a, b));
}
inline __device__ __host__ int4 clamp(int4 v, int4 a, int4 b)
{
    return make_int4(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z), clamp(v.w, a.w, b.w));
}

inline __device__ __host__ uint2 clamp(uint2 v, uint a, uint b)
{
    return make_uint2(clamp(v.x, a, b), clamp(v.y, a, b));
}
inline __device__ __host__ uint2 clamp(uint2 v, uint2 a, uint2 b)
{
    return make_uint2(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y));
}
inline __device__ __host__ uint3 clamp(uint3 v, uint a, uint b)
{
    return make_uint3(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b));
}
inline __device__ __host__ uint3 clamp(uint3 v, uint3 a, uint3 b)
{
    return make_uint3(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z));
}
inline __device__ __host__ uint4 clamp(uint4 v, uint a, uint b)
{
    return make_uint4(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b), clamp(v.w, a, b));
}
inline __device__ __host__ uint4 clamp(uint4 v, uint4 a, uint4 b)
{
    return make_uint4(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z), clamp(v.w, a.w, b.w));
}

////////////////////////////////////////////////////////////////////////////////
// dot product
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ float dot(float2 a, float2 b)
{
    return a.x * b.x + a.y * b.y;
}
inline __host__ __device__ float dot(float3 a, float3 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
inline __host__ __device__ float dot(float4 a, float4 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

inline __host__ __device__ int dot(int2 a, int2 b)
{
    return a.x * b.x + a.y * b.y;
}
inline __host__ __device__ int dot(int3 a, int3 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
inline __host__ __device__ int dot(int4 a, int4 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

inline __host__ __device__ uint dot(uint2 a, uint2 b)
{
    return a.x * b.x + a.y * b.y;
}
inline __host__ __device__ uint dot(uint3 a, uint3 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
inline __host__ __device__ uint dot(uint4 a, uint4 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

////////////////////////////////////////////////////////////////////////////////
// length
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ float length(float2 v)
{
    return sqrtf(dot(v, v));
}
inline __host__ __device__ float length(float3 v)
{
    return sqrtf(dot(v, v));
}
inline __host__ __device__ float length(float4 v)
{
    return sqrtf(dot(v, v));
}

////////////////////////////////////////////////////////////////////////////////
// normalize
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ float2 normalize(float2 v)
{
    float invLen = rsqrtf(dot(v, v));
    return v * invLen;
}
inline __host__ __device__ float3 normalize(float3 v)
{
    float invLen = rsqrtf(dot(v, v));
    return v * invLen;
}
inline __host__ __device__ float4 normalize(float4 v)
{
    float invLen = rsqrtf(dot(v, v));
    return v * invLen;
}

////////////////////////////////////////////////////////////////////////////////
// floor
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ float2 floorf(float2 v)
{
    return make_float2(floorf(v.x), floorf(v.y));
}
inline __host__ __device__ float3 floorf(float3 v)
{
    return make_float3(floorf(v.x), floorf(v.y), floorf(v.z));
}
inline __host__ __device__ float4 floorf(float4 v)
{
    return make_float4(floorf(v.x), floorf(v.y), floorf(v.z), floorf(v.w));
}

////////////////////////////////////////////////////////////////////////////////
// frac - returns the fractional portion of a scalar or each vector component
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ float fracf(float v)
{
    return v - floorf(v);
}
inline __host__ __device__ float2 fracf(float2 v)
{
    return make_float2(fracf(v.x), fracf(v.y));
}
inline __host__ __device__ float3 fracf(float3 v)
{
    return make_float3(fracf(v.x), fracf(v.y), fracf(v.z));
}
inline __host__ __device__ float4 fracf(float4 v)
{
    return make_float4(fracf(v.x), fracf(v.y), fracf(v.z), fracf(v.w));
}

////////////////////////////////////////////////////////////////////////////////
// fmod
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ float2 fmodf(float2 a, float2 b)
{
    return make_float2(fmodf(a.x, b.x), fmodf(a.y, b.y));
}
inline __host__ __device__ float3 fmodf(float3 a, float3 b)
{
    return make_float3(fmodf(a.x, b.x), fmodf(a.y, b.y), fmodf(a.z, b.z));
}
inline __host__ __device__ float4 fmodf(float4 a, float4 b)
{
    return make_float4(fmodf(a.x, b.x), fmodf(a.y, b.y), fmodf(a.z, b.z), fmodf(a.w, b.w));
}

////////////////////////////////////////////////////////////////////////////////
// absolute value
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ float2 fabs(float2 v)
{
    return make_float2(fabs(v.x), fabs(v.y));
}
inline __host__ __device__ float3 fabs(float3 v)
{
    return make_float3(fabs(v.x), fabs(v.y), fabs(v.z));
}
inline __host__ __device__ float4 fabs(float4 v)
{
    return make_float4(fabs(v.x), fabs(v.y), fabs(v.z), fabs(v.w));
}

inline __host__ __device__ int2 abs(int2 v)
{
    return make_int2(abs(v.x), abs(v.y));
}
inline __host__ __device__ int3 abs(int3 v)
{
    return make_int3(abs(v.x), abs(v.y), abs(v.z));
}
inline __host__ __device__ int4 abs(int4 v)
{
    return make_int4(abs(v.x), abs(v.y), abs(v.z), abs(v.w));
}

////////////////////////////////////////////////////////////////////////////////
// reflect
// - returns reflection of incident ray I around surface normal N
// - N should be normalized, reflected vector's length is equal to length of I
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ float3 reflect(float3 i, float3 n)
{
    return i - 2.0f * n * dot(n,i);
}

////////////////////////////////////////////////////////////////////////////////
// cross product
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ float3 cross(float3 a, float3 b)
{
    return make_float3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x);
}

////////////////////////////////////////////////////////////////////////////////
// smoothstep
// - returns 0 if x < a
// - returns 1 if x > b
// - otherwise returns smooth interpolation between 0 and 1 based on x
////////////////////////////////////////////////////////////////////////////////

inline __device__ __host__ float smoothstep(float a, float b, float x)
{
    float y = clamp((x - a) / (b - a), 0.0f, 1.0f);
    return (y*y*(3.0f - (2.0f*y)));
}
inline __device__ __host__ float2 smoothstep(float2 a, float2 b, float2 x)
{
    float2 y = clamp((x - a) / (b - a), 0.0f, 1.0f);
    return (y*y*(make_float2(3.0f) - (make_float2(2.0f)*y)));
}
inline __device__ __host__ float3 smoothstep(float3 a, float3 b, float3 x)
{
    float3 y = clamp((x - a) / (b - a), 0.0f, 1.0f);
    return (y*y*(make_float3(3.0f) - (make_float3(2.0f)*y)));
}
inline __device__ __host__ float4 smoothstep(float4 a, float4 b, float4 x)
{
    float4 y = clamp((x - a) / (b - a), 0.0f, 1.0f);
    return (y*y*(make_float4(3.0f) - (make_float4(2.0f)*y)));
}

// Math utils
inline __device__ __host__ float squaref(float v)
{
    return v * v;
}
inline __device__ __host__ float2 squaref(float2 v)
{
    return v * v;
}
inline __device__ __host__ float3 squaref(float3 v)
{
    return v * v;
}
inline __device__ __host__ float4 squaref(float4 v)
{
    return v * v;
}

inline __device__ __host__ float2 sqrtf(float2 v)
{
    return make_float2(sqrtf(v.x), sqrtf(v.y));
}
inline __device__ __host__ float3 sqrtf(float3 v)
{
    return make_float3(sqrtf(v.x), sqrtf(v.y), sqrtf(v.z));
}
inline __device__ __host__ float4 sqrtf(float4 v)
{
    return make_float4(sqrtf(v.x), sqrtf(v.y), sqrtf(v.z), sqrtf(v.w));
}

inline __device__ __host__ float2 powf(float2 v, float exp)
{
    return make_float2(powf(v.x, exp), powf(v.y, exp));
}
inline __device__ __host__ float3 powf(float3 v, float exp)
{
    return make_float3(powf(v.x, exp), powf(v.y, exp), powf(v.z, exp));
}
inline __device__ __host__ float4 powf(float4 v, float exp)
{
    return make_float4(powf(v.x, exp), powf(v.y, exp), powf(v.z, exp), powf(v.w, exp));
}

inline __device__ __host__ float2 cosf(float2 v)
{
    return make_float2(cosf(v.x), cosf(v.y));
}
inline __device__ __host__ float3 cosf(float3 v)
{
    return make_float3(cosf(v.x), cosf(v.y), cosf(v.z));
}
inline __device__ __host__ float4 cosf(float4 v)
{
    return make_float4(cosf(v.x), cosf(v.y), cosf(v.z), cosf(v.w));
}

inline __device__ __host__ float2 sinf(float2 v)
{
    return make_float2(sinf(v.x), sinf(v.y));
}
inline __device__ __host__ float3 sinf(float3 v)
{
    return make_float3(sinf(v.x), sinf(v.y), sinf(v.z));
}
inline __device__ __host__ float4 sinf(float4 v)
{
    return make_float4(sinf(v.x), sinf(v.y), sinf(v.z), sinf(v.w));
}

#define TENARY_OPERATOR_TYPE(TYPE)                                                  \
    inline __device__ __host__ TYPE where(bool b, TYPE t, TYPE f)                   \
    {                                                                               \
        return b ? t : f;                                                           \
    }                                                                               \
    inline __device__ __host__ TYPE##2 where(bool b, TYPE##2 t, TYPE##2 f)          \
    {                                                                               \
        return b ? t : f;                                                           \
    }                                                                               \
    inline __device__ __host__ TYPE##3 where(bool b, TYPE##3 t, TYPE##3 f)          \
    {                                                                               \
        return b ? t : f;                                                           \
    }                                                                               \
    inline __device__ __host__ TYPE##4 where(bool b, TYPE##4 t, TYPE##4 f)          \
    {                                                                               \
        return b ? t : f;                                                           \
    }                                                                               \
    inline __device__ __host__ TYPE##2 where(bool b, TYPE##2 t, TYPE f)             \
    {                                                                               \
        return make_##TYPE##2(                                                      \
            b ? t.x : f,                                                            \
            b ? t.y : f                                                             \
        );                                                                          \
    }                                                                               \
    inline __device__ __host__ TYPE##3 where(bool b, TYPE##3 t, TYPE f)             \
    {                                                                               \
        return make_##TYPE##3(                                                      \
            b ? t.x : f,                                                            \
            b ? t.y : f,                                                            \
            b ? t.z : f                                                             \
        );                                                                          \
    }                                                                               \
    inline __device__ __host__ TYPE##4 where(bool b, TYPE##4 t, TYPE f)             \
    {                                                                               \
        return make_##TYPE##4(                                                      \
            b ? t.x : f,                                                            \
            b ? t.y : f,                                                            \
            b ? t.z : f,                                                            \
            b ? t.w : f                                                             \
        );                                                                          \
    }                                                                               \
    inline __device__ __host__ TYPE##2 where(bool b, TYPE t, TYPE##2 f)             \
    {                                                                               \
        return make_##TYPE##2(                                                      \
            b ? t : f.x,                                                            \
            b ? t : f.y                                                             \
        );                                                                          \
    }                                                                               \
    inline __device__ __host__ TYPE##3 where(bool b, TYPE t, TYPE##3 f)             \
    {                                                                               \
        return make_##TYPE##3(                                                      \
            b ? t : f.x,                                                            \
            b ? t : f.y,                                                            \
            b ? t : f.z                                                             \
        );                                                                          \
    }                                                                               \
    inline __device__ __host__ TYPE##4 where(bool b, TYPE t, TYPE##4 f)             \
    {                                                                               \
        return make_##TYPE##4(                                                      \
            b ? t : f.x,                                                            \
            b ? t : f.y,                                                            \
            b ? t : f.z,                                                            \
            b ? t : f.w                                                             \
        );                                                                          \
    }                                                                               \
    inline __device__ __host__ TYPE##2 where(int2 b, TYPE##2 t, TYPE f)             \
    {                                                                               \
        return make_##TYPE##2(                                                      \
            b.x ? t.x : f,                                                          \
            b.y ? t.y : f                                                           \
        );                                                                          \
    }                                                                               \
    inline __device__ __host__ TYPE##3 where(int3 b, TYPE##3 t, TYPE f)             \
    {                                                                               \
        return make_##TYPE##3(                                                      \
            b.x ? t.x : f,                                                          \
            b.y ? t.y : f,                                                          \
            b.z ? t.z : f                                                           \
        );                                                                          \
    }                                                                               \
    inline __device__ __host__ TYPE##4 where(int4 b, TYPE##4 t, TYPE f)             \
    {                                                                               \
        return make_##TYPE##4(                                                      \
            b.x ? t.x : f,                                                          \
            b.y ? t.y : f,                                                          \
            b.z ? t.z : f,                                                          \
            b.w ? t.w : f                                                           \
        );                                                                          \
    }                                                                               \
    inline __device__ __host__ TYPE##2 where(int2 b, TYPE t, TYPE##2 f)             \
    {                                                                               \
        return make_##TYPE##2(                                                      \
            b.x ? t : f.x,                                                          \
            b.y ? t : f.y                                                           \
        );                                                                          \
    }                                                                               \
    inline __device__ __host__ TYPE##3 where(int3 b, TYPE t, TYPE##3 f)             \
    {                                                                               \
        return make_##TYPE##3(                                                      \
            b.x ? t : f.x,                                                          \
            b.y ? t : f.y,                                                          \
            b.z ? t : f.z                                                           \
        );                                                                          \
    }                                                                               \
    inline __device__ __host__ TYPE##4 where(int4 b, TYPE t, TYPE##4 f)             \
    {                                                                               \
        return make_##TYPE##4(                                                      \
            b.x ? t : f.x,                                                          \
            b.y ? t : f.y,                                                          \
            b.z ? t : f.z,                                                          \
            b.w ? t : f.w                                                           \
        );                                                                          \
    }                                                                               \
    inline __device__ __host__ TYPE##2 where(int2 b, TYPE##2 t, TYPE##2 f)          \
    {                                                                               \
        return make_##TYPE##2(                                                      \
            b.x ? t.x : f.x,                                                        \
            b.y ? t.y : f.y                                                         \
        );                                                                          \
    }                                                                               \
    inline __device__ __host__ TYPE##3 where(int3 b, TYPE##3 t, TYPE##3 f)          \
    {                                                                               \
        return make_##TYPE##3(                                                      \
            b.x ? t.x : f.x,                                                        \
            b.y ? t.y : f.y,                                                        \
            b.z ? t.z : f.z                                                         \
        );                                                                          \
    }                                                                               \
    inline __device__ __host__ TYPE##4 where(int4 b, TYPE##4 t, TYPE##4 f)          \
    {                                                                               \
        return make_##TYPE##4(                                                      \
            b.x ? t.x : f.x,                                                        \
            b.y ? t.y : f.y,                                                        \
            b.z ? t.z : f.z,                                                        \
            b.w ? t.w : f.w                                                         \
        );                                                                          \
    }                                                                               \
    inline __device__ __host__ TYPE##2 where(int2 b, TYPE t, TYPE f)                \
    {                                                                               \
        return make_##TYPE##2(                                                      \
            b.x ? t : f,                                                            \
            b.y ? t : f                                                             \
        );                                                                          \
    }                                                                               \
    inline __device__ __host__ TYPE##3 where(int3 b, TYPE t, TYPE f)                \
    {                                                                               \
        return make_##TYPE##3(                                                      \
            b.x ? t : f,                                                            \
            b.y ? t : f,                                                            \
            b.z ? t : f                                                             \
        );                                                                          \
    }                                                                               \
    inline __device__ __host__ TYPE##4 where(int4 b, TYPE t, TYPE f)                \
    {                                                                               \
        return make_##TYPE##4(                                                      \
            b.x ? t : f,                                                            \
            b.y ? t : f,                                                            \
            b.z ? t : f,                                                            \
            b.w ? t : f                                                             \
        );                                                                          \
    }                                                                               \

TENARY_OPERATOR_TYPE(float)
TENARY_OPERATOR_TYPE(int)
TENARY_OPERATOR_TYPE(uint)
TENARY_OPERATOR_TYPE(double)