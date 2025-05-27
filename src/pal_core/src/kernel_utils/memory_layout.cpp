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
    size_t actual_simd_lanes_per_group,
    bool is_prefill
) {
    AttentionMemoryLayout layout;
    uintptr_t current_offset_bytes = 0;

    // Number of query heads per kv group
    uint32_t query_heads_per_kv_group = 1;
    if (params.num_kv_heads > 0) {
        query_heads_per_kv_group = std::max(1u, params.num_q_heads / params.num_kv_heads);
    }
    // This num_simd_groups is based on actual launched threads_per_group,
    // which should be query_heads_per_kv_group * actual_simd_lanes_per_group for prefill.
    const uint32_t num_simd_groups_in_tg = calculate_simd_groups(threads_per_group, actual_simd_lanes_per_group);

    // Scratch for TG-wide reductions (e.g. finding max score)
    size_t partial_reduce_scratch = threads_per_group * sizeof(float);
    // Scratch for per-SIMD-group results before final reduction
    size_t simd_reduced_maxes_scratch = num_simd_groups_in_tg * sizeof(float);
    size_t simd_reduced_sum_exps_scratch = num_simd_groups_in_tg * sizeof(float);
    // Space for {m_global, s_global} like stats (decode uses this for online softmax over K/V tiles)
    // For prefill, each Q-instance (from Q-block vs K-tile) computes its own page_max/page_sum.
    size_t global_stats_scratch = num_simd_groups_in_tg * 2 * sizeof(float); // Max 2 floats per SIMD group for M/L
    size_t kahan_comp_scratch = num_simd_groups_in_tg * sizeof(float); // Kahan compensation per SIMD group
    // Scratch for V-sum reduction (decode uses this for reducing partial V sums from SIMD groups)
    size_t simd_v_sums_scratch = num_simd_groups_in_tg * params.head_dim * sizeof(float);; // Assuming V is reduced in float4 chunks per SIMD group

    // --- Conditional Logic for Prefill vs. Decode ---
    if (is_prefill) {
        // --- PREFILL PATH (Symmetric QKV Tiling) ---
        // params.tokens_per_page is D_s (symmetric depth)
        // params.tile_size_T_runtime is also D_s

        // 1. Q_shmem_block: Stores D_s Q-vectors, each for query_heads_per_kv_group heads, as float.
        layout.q_shmem_bytes = params.tokens_per_page * query_heads_per_kv_group * params.head_dim * sizeof(float);
        current_offset_bytes = AttentionMemoryLayout::align_size(layout.q_shmem_bytes);

        // 2. K_tile: Stores D_s K-vectors as half.
        layout.k_tile_bytes = params.tokens_per_page * params.head_dim * sizeof(half);
        current_offset_bytes = AttentionMemoryLayout::align_size(current_offset_bytes + layout.k_tile_bytes);

        // 3. V_tile: Stores D_s V-vectors as half.
        layout.v_tile_bytes = params.tokens_per_page * params.head_dim * sizeof(half);
        current_offset_bytes = AttentionMemoryLayout::align_size(current_offset_bytes + layout.v_tile_bytes);

        // Component 1: V-sum accumulator space (matches Component 1 from _eval_gpu_prefill)
        size_t v_sum_accum_bytes_tgm = query_heads_per_kv_group * params.head_dim * sizeof(float);
        current_offset_bytes = AttentionMemoryLayout::align_size(current_offset_bytes + v_sum_accum_bytes_tgm);

        // Component 2: M/L stats space (matches Component 2 from _eval_gpu_prefill)
        size_t ml_stats_bytes_tgm = query_heads_per_kv_group * 2 * sizeof(float);
        current_offset_bytes = AttentionMemoryLayout::align_size(current_offset_bytes + ml_stats_bytes_tgm);

        // Component 3: General reduction scratch (matches Component 3 from _eval_gpu_prefill)
        layout.partial_reduce_scratch_bytes = partial_reduce_scratch;
        current_offset_bytes = AttentionMemoryLayout::align_size(current_offset_bytes + layout.partial_reduce_scratch_bytes);

        // Zero out other decode-specific items
        layout.simd_reduced_maxes_bytes = 0;
        layout.simd_reduced_adjusted_sum_exps_bytes = 0;
        layout.global_stats_bytes = 0;
        layout.s_global_compensation_bytes = 0;
        layout.simd_v_chunk_sums_bytes = 0;
        layout.page_table_slice_bytes = 0;

    } else {
        // --- DECODE PATH (Original Logic) ---
        // params.tokens_per_page is the TPP_Opt from PIE for decode cache structure.
        // params.tile_size_T_runtime is the calculated K/V tile depth for decode kernel.

        // 1. Q_shmem: For one Q-vector, as float.
        layout.q_shmem_bytes = params.head_dim * sizeof(float);
        current_offset_bytes = AttentionMemoryLayout::align_size(layout.q_shmem_bytes);

        // 2. K_tile: Uses params.tile_size_T_runtime for depth.
        layout.k_tile_bytes = params.tile_size_T_runtime * params.head_dim * sizeof(half);
        current_offset_bytes = AttentionMemoryLayout::align_size(current_offset_bytes + layout.k_tile_bytes);

        // 3. V_tile: Uses params.tile_size_T_runtime for depth.
        layout.v_tile_bytes = params.tile_size_T_runtime * params.head_dim * sizeof(half);
        current_offset_bytes = AttentionMemoryLayout::align_size(current_offset_bytes + layout.v_tile_bytes);

        // 4. Fixed Scratch components for Decode (as originally designed for it)
        layout.partial_reduce_scratch_bytes = partial_reduce_scratch;
        current_offset_bytes = AttentionMemoryLayout::align_size(current_offset_bytes + layout.partial_reduce_scratch_bytes);

        layout.simd_reduced_maxes_bytes = simd_reduced_maxes_scratch;
        current_offset_bytes = AttentionMemoryLayout::align_size(current_offset_bytes + layout.simd_reduced_maxes_bytes);

        layout.simd_reduced_adjusted_sum_exps_bytes = simd_reduced_sum_exps_scratch;
        current_offset_bytes = AttentionMemoryLayout::align_size(current_offset_bytes + layout.simd_reduced_adjusted_sum_exps_bytes);

        // For decode, num_simd_groups_in_tg is likely 1 or 2. global_stats and s_global_comp are small.
        layout.global_stats_bytes = 2 * sizeof(float); // Per TG, not per SIMD group for decode's online softmax
        current_offset_bytes = AttentionMemoryLayout::align_size(current_offset_bytes + layout.global_stats_bytes);

        layout.s_global_compensation_bytes = 1 * sizeof(float); // Per TG
        current_offset_bytes = AttentionMemoryLayout::align_size(current_offset_bytes + layout.s_global_compensation_bytes);

        layout.simd_v_chunk_sums_bytes = simd_v_sums_scratch;
        current_offset_bytes = AttentionMemoryLayout::align_size(current_offset_bytes + layout.simd_v_chunk_sums_bytes);

        layout.page_table_slice_bytes = params.max_logical_blocks_per_seq * sizeof(uint32_t);
        current_offset_bytes = AttentionMemoryLayout::align_size(current_offset_bytes + layout.page_table_slice_bytes);
    }

    // --- Final Guard and Total ---
    layout.final_guard_bytes = kFinalMemoryPaddingGuardBytes;
    current_offset_bytes = AttentionMemoryLayout::align_size(current_offset_bytes + layout.final_guard_bytes);

    layout.total_bytes = current_offset_bytes; // current_offset_bytes is already aligned at each step of accumulation

    spdlog::debug("[Memory Layout {}] Q_shmem: {}, K_tile: {}, V_tile: {}, ReduceScratch: {}, OtherFixed: ..., Total: {} bytes",
                  is_prefill ? "Prefill" : "Decode",
                  layout.q_shmem_bytes, layout.k_tile_bytes, layout.v_tile_bytes,
                  layout.partial_reduce_scratch_bytes, layout.total_bytes);

    return layout;
}


} // namespace pal::cpp::kernel_utils
