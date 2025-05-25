// memory_layout.cpp
// Implementation of shared memory layout utilities
//
// Copyright 2024 The Proxy Company. All Rights Reserved.

#include "pal_core/kernel_utils/memory_layout.hpp"
#include <spdlog/spdlog.h>

// Define half type for memory calculations (matching Metal's half type)
using half = short;

namespace pal::cpp::kernel_utils {

AttentionMemoryLayout calculate_attention_memory_layout(
    const PagedAttentionParams& params,
    size_t threads_per_group,
    size_t actual_simd_lanes_per_group
) {
    AttentionMemoryLayout layout;
    uintptr_t tg_mem_current_offset_bytes = 0;

    const uint32_t num_simd_groups = calculate_simd_groups(threads_per_group, actual_simd_lanes_per_group);

    // 1. q_shmem: head_dim floats
    layout.q_shmem_bytes = params.head_dim * sizeof(float);
    tg_mem_current_offset_bytes += layout.q_shmem_bytes;
    tg_mem_current_offset_bytes = AttentionMemoryLayout::align_size(tg_mem_current_offset_bytes);

    // 2. G_partial_max_scores: threads_per_group floats
    layout.partial_reduce_scratch_bytes = threads_per_group * sizeof(float);
    tg_mem_current_offset_bytes += layout.partial_reduce_scratch_bytes;

    // 3. G_simd_reduced_maxes: num_simd_groups floats
    tg_mem_current_offset_bytes = AttentionMemoryLayout::align_size(tg_mem_current_offset_bytes);
    layout.simd_reduced_maxes_bytes = num_simd_groups * sizeof(float);
    tg_mem_current_offset_bytes += layout.simd_reduced_maxes_bytes;

    // 4. G_simd_reduced_adjusted_sum_exps: num_simd_groups floats
    tg_mem_current_offset_bytes = AttentionMemoryLayout::align_size(tg_mem_current_offset_bytes);
    layout.simd_reduced_adjusted_sum_exps_bytes = num_simd_groups * sizeof(float);
    tg_mem_current_offset_bytes += layout.simd_reduced_adjusted_sum_exps_bytes;

    // 5. g_global_stats (float2 for m_global, s_global)
    tg_mem_current_offset_bytes = AttentionMemoryLayout::align_size(tg_mem_current_offset_bytes);
    layout.global_stats_bytes = 2 * sizeof(float);
    tg_mem_current_offset_bytes += layout.global_stats_bytes;

    // 6. g_s_global_compensation (for Kahan summation)
    tg_mem_current_offset_bytes = AttentionMemoryLayout::align_size(tg_mem_current_offset_bytes);
    layout.s_global_compensation_bytes = 1 * sizeof(float);
    tg_mem_current_offset_bytes += layout.s_global_compensation_bytes;

    // 7. G_simd_group_v_sums (float4 per SIMD group)
    tg_mem_current_offset_bytes = AttentionMemoryLayout::align_size(tg_mem_current_offset_bytes);
    layout.simd_v_chunk_sums_bytes = num_simd_groups * sizeof(float) * 4;
    tg_mem_current_offset_bytes += layout.simd_v_chunk_sums_bytes;

    // 8. K Tile memory
    tg_mem_current_offset_bytes = AttentionMemoryLayout::align_size(tg_mem_current_offset_bytes);
    layout.k_tile_bytes = params.tile_size_T_runtime * params.head_dim * sizeof(half);
    tg_mem_current_offset_bytes += layout.k_tile_bytes;

    // 9. V Tile memory
    tg_mem_current_offset_bytes = AttentionMemoryLayout::align_size(tg_mem_current_offset_bytes);
    layout.v_tile_bytes = params.tile_size_T_runtime * params.head_dim * sizeof(half);
    tg_mem_current_offset_bytes += layout.v_tile_bytes;

    // 10. Per-sequence page-table slice
    tg_mem_current_offset_bytes = AttentionMemoryLayout::align_size(tg_mem_current_offset_bytes);
    layout.page_table_slice_bytes = params.max_logical_blocks_per_seq * sizeof(uint32_t);
    tg_mem_current_offset_bytes += layout.page_table_slice_bytes;

    // Final padding guard
    layout.final_guard_bytes = kFinalMemoryPaddingGuardBytes;
    tg_mem_current_offset_bytes += layout.final_guard_bytes;

    // Ensure final size is aligned
    layout.total_bytes = AttentionMemoryLayout::align_size(tg_mem_current_offset_bytes);
    spdlog::debug("[Memory Layout] Total calculated tg_memory_bytes: {} bytes", layout.total_bytes);

    return layout;
}

} // namespace pal::cpp::kernel_utils
