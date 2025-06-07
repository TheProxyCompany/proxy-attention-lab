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

#include "mlx/backend/metal/kernels/metal_3_1/bf16.h"
#include "mlx/backend/metal/kernels/bf16_math.h"
#include "mlx/backend/metal/kernels/complex.h"
#include "mlx/backend/metal/kernels/defines.h"

using namespace metal;

// Generic vector type definition
template <typename T, int SIZE>
struct Vec {};

// Specialization for half4
template <>
struct Vec<half, 4> {
  using Type = half4;
};

// Specialization for bfloat16_t requires a custom struct
// to group 4 bfloat16_t values, as bfloat4 is not a built-in Metal type.
struct bfloat4_ {
  bfloat16_t x, y, z, w;
};

template <>
struct Vec<bfloat16_t, 4> {
  using Type = bfloat4_;
};

// Convert half4 to float4
inline float4 to_float4(half4 v) {
    return float4(v);
}

// Convert our custom bfloat4_ struct to float4
inline float4 to_float4(bfloat4_ v) {
    return float4((float)v.x, (float)v.y, (float)v.z, (float)v.w);
}

// Convert float4 to a target type T's vector representation
template <typename T>
inline typename Vec<T, 4>::Type from_float4(float4 v);

// Specialization for converting float4 to half4
template <>
inline half4 from_float4<half>(float4 v) {
    return half4(v);
}

// Specialization for converting float4 to our custom bfloat4_ struct
template <>
inline bfloat4_ from_float4<bfloat16_t>(float4 v) {
    bfloat4_ result;
    result.x = bfloat16_t(v.x);
    result.y = bfloat16_t(v.y);
    result.z = bfloat16_t(v.z);
    result.w = bfloat16_t(v.w);
    return result;
}
