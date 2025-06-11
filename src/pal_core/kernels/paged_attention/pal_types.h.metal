// pal_types.h.metal
// Metal shader header for PAL types.
//
// Copyright 2025 The Proxy Company. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ============================================================================

#pragma once

#include <metal_stdlib>

#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/bf16_math.h"
#include "mlx/backend/metal/kernels/complex.h"
#include "mlx/backend/metal/kernels/defines.h"

using namespace metal;

#define MAX(a, b) ((a) > (b) ? (a) : (b));
#define MIN(a, b) ((a) < (b) ? (a) : (b));
#define ALIGN16(ptr) (((uintptr_t)(ptr) + 15) & ~15)

// ============================================================================
// Composite vector types for extended width support
// ============================================================================

// 8-wide float vector
struct Float8 {
    float4 lo;
    float4 hi;
};

// 8-wide half vector
struct Half8 {
    half4 lo;
    half4 hi;
};

// Bfloat16 vector types - using structs for compatibility
struct Bfloat2 {
    bfloat16_t x;
    bfloat16_t y;

    // Conversion to float2
    operator float2() const {
        return float2(static_cast<float>(x), static_cast<float>(y));
    }
};

struct Bfloat4 {
    bfloat16_t x;
    bfloat16_t y;
    bfloat16_t z;
    bfloat16_t w;

    // Conversion to float4
    operator float4() const {
        return float4(static_cast<float>(x), static_cast<float>(y),
                      static_cast<float>(z), static_cast<float>(w));
    }
};

struct Bfloat8 {
    Bfloat4 lo;
    Bfloat4 hi;
};

// ============================================================================
// Vector type mapping template
// ============================================================================

template <typename T, int SIZE>
struct Vec {};

// Float vector specializations
template <>
struct Vec<float, 1> {
    using Type = float;
};

template <>
struct Vec<float, 2> {
    using Type = float2;
};

template <>
struct Vec<float, 4> {
    using Type = float4;
};

template <>
struct Vec<float, 8> {
    using Type = Float8;
};

// Half vector specializations
template <>
struct Vec<half, 1> {
    using Type = half;
};

template <>
struct Vec<half, 2> {
    using Type = half2;
};

template <>
struct Vec<half, 4> {
    using Type = half4;
};

template <>
struct Vec<half, 8> {
    using Type = Half8;
};

// Bfloat16 vector specializations
template <>
struct Vec<bfloat16_t, 1> {
    using Type = bfloat16_t;
};

template <>
struct Vec<bfloat16_t, 2> {
    using Type = Bfloat2;
};

template <>
struct Vec<bfloat16_t, 4> {
    using Type = Bfloat4;
};

template <>
struct Vec<bfloat16_t, 8> {
    using Type = Bfloat8;
};

// ============================================================================
// Float accumulator type mapping
// ============================================================================

template <typename T>
struct FloatVec {};

// Float8 accumulator specialization (custom struct needs mapping)
template <>
struct FloatVec<Float8> {
    using Type = Float8;
};

// Half to float accumulator mapping
template <>
struct FloatVec<half> {
    using Type = float;
};

template <>
struct FloatVec<half2> {
    using Type = float2;
};

template <>
struct FloatVec<half4> {
    using Type = float4;
};

template <>
struct FloatVec<Half8> {
    using Type = Float8;
};

// Bfloat16 to float accumulator mapping
template <>
struct FloatVec<bfloat16_t> {
    using Type = float;
};

template <>
struct FloatVec<Bfloat2> {
    using Type = float2;
};

template <>
struct FloatVec<Bfloat4> {
    using Type = float4;
};

template <>
struct FloatVec<Bfloat8> {
    using Type = Float8;
};

// ============================================================================
// Type conversion utilities
// ============================================================================

// Float8 conversions (custom struct)
inline void from_float(thread Float8& dst, Float8 src) {
    dst = src;
}

// Half conversions
inline void from_float(thread half& dst, float src) {
    dst = static_cast<half>(src);
}

inline void from_float(thread half2& dst, float2 src) {
    dst = half2(src);
}

inline void from_float(thread half4& dst, float4 src) {
    dst = half4(src);
}

inline void from_float(thread Half8& dst, Float8 src) {
    dst.lo = half4(src.lo);
    dst.hi = half4(src.hi);
}

// Bfloat16 conversions
inline void from_float(thread bfloat16_t& dst, float src) {
    dst = static_cast<bfloat16_t>(src);
}

inline void from_float(thread Bfloat2& dst, float2 src) {
    dst.x = static_cast<bfloat16_t>(src.x);
    dst.y = static_cast<bfloat16_t>(src.y);
}

inline void from_float(thread Bfloat4& dst, float4 src) {
    dst.x = static_cast<bfloat16_t>(src.x);
    dst.y = static_cast<bfloat16_t>(src.y);
    dst.z = static_cast<bfloat16_t>(src.z);
    dst.w = static_cast<bfloat16_t>(src.w);
}

inline void from_float(thread Bfloat8& dst, Float8 src) {
    from_float(dst.lo, src.lo);
    from_float(dst.hi, src.hi);
}

// ============================================================================
// Multiplication operations
// ============================================================================

template <typename Acc, typename A, typename B>
inline Acc mul(A a, B b);

// Float8 multiplication (custom struct)
template <>
inline Float8 mul<Float8, Float8, Float8>(Float8 a, Float8 b) {
    Float8 c;
    c.lo = a.lo * b.lo;
    c.hi = a.hi * b.hi;
    return c;
}

// Half multiplications with float accumulation
template <>
inline float mul<float, half, half>(half a, half b) {
    return static_cast<float>(a) * static_cast<float>(b);
}

template <>
inline float2 mul<float2, half2, half2>(half2 a, half2 b) {
    return float2(a) * float2(b);
}

template <>
inline float4 mul<float4, half4, half4>(half4 a, half4 b) {
    return float4(a) * float4(b);
}

template <>
inline Float8 mul<Float8, Half8, Half8>(Half8 a, Half8 b) {
    Float8 c;
    c.lo = mul<float4, half4, half4>(a.lo, b.lo);
    c.hi = mul<float4, half4, half4>(a.hi, b.hi);
    return c;
}

// Bfloat16 multiplications with float accumulation
template <>
inline float mul<float, bfloat16_t, bfloat16_t>(bfloat16_t a, bfloat16_t b) {
    return static_cast<float>(a) * static_cast<float>(b);
}

template <>
inline float2 mul<float2, Bfloat2, Bfloat2>(Bfloat2 a, Bfloat2 b) {
    float2 a_f(static_cast<float>(a.x), static_cast<float>(a.y));
    float2 b_f(static_cast<float>(b.x), static_cast<float>(b.y));
    return a_f * b_f;
}

template <>
inline float4 mul<float4, Bfloat4, Bfloat4>(Bfloat4 a, Bfloat4 b) {
    float4 a_f(static_cast<float>(a.x), static_cast<float>(a.y),
               static_cast<float>(a.z), static_cast<float>(a.w));
    float4 b_f(static_cast<float>(b.x), static_cast<float>(b.y),
               static_cast<float>(b.z), static_cast<float>(b.w));
    return a_f * b_f;
}

template <>
inline Float8 mul<Float8, Bfloat8, Bfloat8>(Bfloat8 a, Bfloat8 b) {
    Float8 c;
    c.lo = mul<float4, Bfloat4, Bfloat4>(a.lo, b.lo);
    c.hi = mul<float4, Bfloat4, Bfloat4>(a.hi, b.hi);
    return c;
}

// ============================================================================
// Fused multiply-add operations
// ============================================================================

// Float8 FMA operations (custom struct)
inline Float8 fma(Float8 a, Float8 b, Float8 c) {
    Float8 res;
    res.lo = a.lo * b.lo + c.lo;
    res.hi = a.hi * b.hi + c.hi;
    return res;
}

// Half FMA with float accumulation
inline float fma(half a, half b, float c) {
    return static_cast<float>(a) * static_cast<float>(b) + c;
}

inline float2 fma(half2 a, half2 b, float2 c) {
    return float2(a) * float2(b) + c;
}

inline float4 fma(half4 a, half4 b, float4 c) {
    return float4(a) * float4(b) + c;
}

inline Float8 fma(Half8 a, Half8 b, Float8 c) {
    Float8 res;
    res.lo = fma(a.lo, b.lo, c.lo);
    res.hi = fma(a.hi, b.hi, c.hi);
    return res;
}

// Bfloat16 FMA with float accumulation
inline float fma(bfloat16_t a, bfloat16_t b, float c) {
    return static_cast<float>(a) * static_cast<float>(b) + c;
}

inline float2 fma(Bfloat2 a, Bfloat2 b, float2 c) {
    float2 a_f(static_cast<float>(a.x), static_cast<float>(a.y));
    float2 b_f(static_cast<float>(b.x), static_cast<float>(b.y));
    return a_f * b_f + c;
}

inline float4 fma(Bfloat4 a, Bfloat4 b, float4 c) {
    float4 a_f(static_cast<float>(a.x), static_cast<float>(a.y),
               static_cast<float>(a.z), static_cast<float>(a.w));
    float4 b_f(static_cast<float>(b.x), static_cast<float>(b.y),
               static_cast<float>(b.z), static_cast<float>(b.w));
    return a_f * b_f + c;
}

inline Float8 fma(Bfloat8 a, Bfloat8 b, Float8 c) {
    Float8 res;
    res.lo = fma(a.lo, b.lo, c.lo);
    res.hi = fma(a.hi, b.hi, c.hi);
    return res;
}

// ============================================================================
// Sum reduction operations
// ============================================================================

template <typename T>
inline float sum(T v);

template <>
inline float sum<float4>(float4 v) {
    return v.x + v.y + v.z + v.w;
}

// Float8 sum operations (custom struct)
template <>
inline float sum<Float8>(Float8 v) {
    return sum<float4>(v.lo) + sum<float4>(v.hi);
}

// Half sum operations
template <>
inline float sum<half>(half v) {
    return static_cast<float>(v);
}

template <>
inline float sum<half2>(half2 v) {
    return static_cast<float>(v.x) + static_cast<float>(v.y);
}

template <>
inline float sum<half4>(half4 v) {
    float4 v_f = float4(v);
    return v_f.x + v_f.y + v_f.z + v_f.w;
}

template <>
inline float sum<Half8>(Half8 v) {
    return sum<half4>(v.lo) + sum<half4>(v.hi);
}

// Bfloat16 sum operations
template <>
inline float sum<bfloat16_t>(bfloat16_t v) {
    return static_cast<float>(v);
}

template <>
inline float sum<Bfloat2>(Bfloat2 v) {
    return static_cast<float>(v.x) + static_cast<float>(v.y);
}

template <>
inline float sum<Bfloat4>(Bfloat4 v) {
    return static_cast<float>(v.x) + static_cast<float>(v.y) +
           static_cast<float>(v.z) + static_cast<float>(v.w);
}

template <>
inline float sum<Bfloat8>(Bfloat8 v) {
    return sum<Bfloat4>(v.lo) + sum<Bfloat4>(v.hi);
}

// ============================================================================
// Dot product utilities
// ============================================================================

template <typename T>
inline float dot(T a, T b) {
    return sum(mul<T, T, T>(a, b));
}

template <typename A, typename T>
inline float dot(T a, T b) {
    return sum(mul<A, T, T>(a, b));
}

// ============================================================================
// SIMD reduction helpers
// ============================================================================

template <typename T>
inline T simd_sum(T v, uint simd_size) {
    #pragma unroll
    for (uint off = simd_size >> 1; off > 0; off >>= 1)
        v += simd_shuffle_xor(v, off);
    return v;
}

template <typename T>
inline T simd_max(T v, uint simd_size) {
    #pragma unroll
    for (uint off = simd_size >> 1; off > 0; off >>= 1)
        v = max(v, simd_shuffle_xor(v, off));
    return v;
}

// ============================================================================
// Attention-specific utilities
// borrowed from:
// https://github.com/EricLBuehler/mistral.rs/blob/58df07e2abb758f7c1d4de8f26f24803d7dbee1f/mistralrs-paged-attn/src/metal/kernels/pagedattention.metal
// ============================================================================

// Optimized Q*K dot product with SIMD reduction
template <int SIMD_SIZE, typename Vec, int N>
inline float qk_dot(const threadgroup Vec (&q)[N], const thread Vec (&k)[N]) {
    // Compute parallel products for Q*K^T
    using AccVec = typename FloatVec<Vec>::Type;
    AccVec qk_vec = mul<AccVec, Vec, Vec>(q[0], k[0]);

    #pragma unroll
    for (int i = 1; i < N; ++i) {
        qk_vec = fma(q[i], k[i], qk_vec);
    }

    // Reduce across vector lanes
    float qk = sum(qk_vec);

    // Reduce across SIMD lanes
    return simd_sum(qk, SIMD_SIZE);
}

// Block-wide sum for softmax computation
template <int NUM_SIMD_GROUPS, int SIMD_SIZE>
inline float block_sum(threadgroup float* shared_mem, float local_sum,
                      uint simd_group_id, uint simd_lane_id) {
    // First reduce within each SIMD group
    local_sum = simd_sum(local_sum, SIMD_SIZE);

    // SIMD group leaders write to shared memory
    if (simd_lane_id == 0) {
        shared_mem[simd_group_id] = local_sum;
    }

    // Synchronize to ensure all groups have written
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Final reduction across SIMD groups
    if (simd_lane_id < NUM_SIMD_GROUPS) {
        local_sum = shared_mem[simd_lane_id];
    } else {
        local_sum = 0.0f;
    }

    // Reduce across the SIMD groups
    local_sum = simd_sum(local_sum, SIMD_SIZE);

    // Broadcast result to all threads
    return simd_broadcast(local_sum, 0);
}
