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
