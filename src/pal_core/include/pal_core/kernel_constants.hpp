#pragma once
// kernel_constants.hpp
// Centralized constants for all PAL kernels
//
// Copyright 2024 The Proxy Company. All Rights Reserved.

#include <cstdint>

namespace pal::cpp::kernels {

// Common constants used across multiple kernels
constexpr uint32_t DEFAULT_THREADS_PER_GROUP = 64;
constexpr size_t MEMORY_ALIGNMENT = 64;
constexpr size_t SIMD_WIDTH = 32; // Default SIMD width for Metal

// Kernel-specific namespaces for organization
namespace paged_attention {
    // Mathematical constants
    constexpr float kLogFp16DenormMinVal = -88.0f;

    // Prefill pass configuration
    constexpr uint32_t PREFILL_PASS1_Q_HEAD_BLOCK_SIZE = 8;
    constexpr uint32_t PREFILL_PASS2_TOKEN_BLOCK_SIZE = 64;
    constexpr uint32_t PREFILL_PASS2_QHEAD_BLOCK_SIZE = 8;

    // Memory and tiling constraints
    constexpr uint32_t TILE_SIZE_ALIGNMENT = 4;
    constexpr uint32_t MIN_TILE_SIZE_SOFT = 8;
    constexpr uint32_t MAX_TILE_SIZE_PRACTICAL = 256;

    // Vectorization requirements
    constexpr uint32_t HEAD_DIM_VECTORIZATION = 4;
}

// Future kernels can add their namespaces here
// namespace future_kernel {
//     constexpr uint32_t SOME_CONSTANT = 42;
// }

} // namespace pal::cpp::kernels
