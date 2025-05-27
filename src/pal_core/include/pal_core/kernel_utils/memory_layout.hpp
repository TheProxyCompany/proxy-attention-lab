#pragma once
// memory_layout.hpp
// Shared utilities for calculating threadgroup memory layouts for various kernels
//
// Copyright 2024 The Proxy Company. All Rights Reserved.

#include <cstddef>
#include <cstdint>
#include "shaders/paged_attention_types.h"

namespace pal::cpp::kernel_utils {

// Common memory alignment constants
constexpr size_t kMemoryAlignmentBytes = 64;
constexpr size_t kMemoryAlignmentMask = kMemoryAlignmentBytes - 1;
constexpr size_t kFinalMemoryPaddingGuardBytes = 32;

// Base structure for threadgroup memory layout
struct ThreadgroupMemoryLayout {
    size_t total_bytes{0};

    // Align a size to the next memory boundary
    static size_t align_size(size_t size) {
        return (size + kMemoryAlignmentMask) & ~kMemoryAlignmentMask;
    }
};

// Memory layout specific to attention-style kernels
struct AttentionMemoryLayout : ThreadgroupMemoryLayout {
    size_t q_shmem_bytes{0};
    size_t partial_reduce_scratch_bytes{0};
    size_t simd_reduced_maxes_bytes{0};
    size_t simd_reduced_adjusted_sum_exps_bytes{0};
    size_t global_stats_bytes{0};
    size_t s_global_compensation_bytes{0};
    size_t simd_v_chunk_sums_bytes{0};
    size_t k_tile_bytes{0};
    size_t v_tile_bytes{0};
    size_t page_table_slice_bytes{0};
    size_t final_guard_bytes{0};
};

// Calculate number of SIMD groups for a given thread count
inline uint32_t calculate_simd_groups(size_t threads_per_group, size_t simd_lanes_per_group) {
    return (threads_per_group + simd_lanes_per_group - 1) / simd_lanes_per_group;
}

AttentionMemoryLayout calculate_attention_memory_layout(
    const PagedAttentionParams& params,
    size_t threads_per_group,
    size_t actual_simd_lanes_per_group,
    bool is_prefill = false
);

} // namespace pal::cpp::kernel_utils
