// paged_attention_prefill.metal
// Metal shader implementation for paged attention operations with tiled V accumulation.
//
// Copyright 2024 The Proxy Company. All Rights Reserved.
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

#include <metal_stdlib>
#include "paged_attention.h.metal"

using namespace metal;

/**
 * paged_attn_prefill_kernel
 * -----------------
 * Pass 1 of the new page-centric prefill architecture.
 * Each threadgroup processes:
 * - One active KV page (from tg_pos_in_grid.x)
 * - A block of Q heads (from tg_pos_in_grid.y)
 * The kernel receives "Relevant Query Map" data to identify which query tokens
 * have history on its assigned page.
 */
[[kernel]] void paged_attn_prefill_kernel(
    device      const half* queries_in              [[buffer(0)]],
    device      const half* k_cache_pool_in         [[buffer(1)]],
    device      const half* v_cache_pool_in         [[buffer(2)]],
    device      const uint* page_table_in           [[buffer(3)]],
    device      const int*  sequence_lengths_in     [[buffer(4)]],
    device      const int*  query_to_seq_map_in     [[buffer(5)]],
    device      const int*  query_token_offset_in   [[buffer(6)]],
    constant    const PagedAttentionParams& params  [[buffer(7)]],
    device      half* output_buffer                 [[buffer(8)]],
    // New Pass 1 "Relevant Query Map" input buffers
    device      const uint* relevant_query_indices  [[buffer(9)]],   // Flat array of query indices
    device      const uint* relevant_history_starts [[buffer(10)]],  // History start offsets on page
    device      const uint* relevant_history_counts [[buffer(11)]],  // Number of history tokens on page
    device      const uint* page_offsets            [[buffer(12)]],  // Offsets into flat arrays per page
    device      const uint* active_pages            [[buffer(13)]],  // List of active page IDs
    // New Pass 1 output buffers for intermediate results
    device      float* m_locals_pass1_out           [[buffer(14)]],  // Local max scores
    device      float* s_locals_pass1_out           [[buffer(15)]],  // Local sum-exponentials
    device      half*  o_partials_pass1_out         [[buffer(16)]],  // Unnormalized partial V-accumulations
    uint actual_simd_width                          [[threads_per_simdgroup]],
    threadgroup float* tg_mem                       [[threadgroup(0)]],
    uint3       tg_pos_in_grid                      [[threadgroup_position_in_grid]],
    uint3       tg_dim                              [[threads_per_threadgroup]],
    uint        local_idx_in_tg                     [[thread_index_in_threadgroup]],
    uint        simd_lane_id                        [[thread_index_in_simdgroup]],
    uint        simd_group_id                       [[simdgroup_index_in_threadgroup]]
) {

    // Early exit for degenerate case where head_dim is zero
    if (params.head_dim == 0) {
        return;
    }

    // Define Pass 1 constants
    constexpr uint PREFILL_PASS1_Q_HEAD_BLOCK_SIZE_CONST = 8;
    constexpr uintptr_t kAlignmentBytes = 64;
    constexpr uintptr_t kAlignmentMask = kAlignmentBytes - 1;

    // --- Step 2: Threadgroup Role Identification ---
    // Determine assigned Global_KV_Page_ID from X dimension
    uint assigned_page_index = tg_pos_in_grid.x;  // Index into active_pages array
    uint assigned_global_kv_page_id = active_pages[assigned_page_index];

    // Determine Q-Head Block Responsibility from Y dimension
    uint q_head_block_idx = tg_pos_in_grid.y;
    uint q_head_start_index_in_model = q_head_block_idx * PREFILL_PASS1_Q_HEAD_BLOCK_SIZE_CONST;
    uint num_q_heads_for_this_block = min(PREFILL_PASS1_Q_HEAD_BLOCK_SIZE_CONST,
                                          params.num_q_heads - q_head_start_index_in_model);

    // Access "Relevant Query Map" Data
    // Find the range of relevance entries for this page
    uint relevance_start_idx = page_offsets[assigned_page_index];
    uint relevance_end_idx = page_offsets[assigned_page_index + 1];
    uint num_relevant_queries_for_page = relevance_end_idx - relevance_start_idx;

    uint local_thread_idx = local_idx_in_tg;    // Thread ID within this group
    const uint num_simd_groups = max(1u, (tg_dim.x + actual_simd_width - 1) / actual_simd_width);

    // --- 5/10: Threadgroup Memory Carving ---
    // Layout for reductions, statistics and vector caching.
    // q_shmem for query vectors
    // tg_partial_reduce_scratch, tg_simd_reduce_scratch, tg_simd_exp_sums_scratch for softmax reductions
    // tg_global_stats, tg_s_global_comp for Kahan summation
    // tg_simd_v_chunk_sums for final output reduction

    threadgroup float* q_shmem = tg_mem;

    // Align subsequent threadgroup memory sections to kAlignmentBytes (e.g., 64 bytes).
    uintptr_t current_offset = (uintptr_t)(q_shmem + params.head_dim);
    current_offset = (current_offset + kAlignmentMask) & ~kAlignmentMask;
    threadgroup float* tg_partial_reduce_scratch = (threadgroup float*)current_offset;  // threads_per_tg floats

    current_offset = (uintptr_t)(tg_partial_reduce_scratch + tg_dim.x);
    current_offset = (current_offset + kAlignmentMask) & ~kAlignmentMask;
    threadgroup float* tg_simd_reduce_scratch = (threadgroup float*)current_offset; // num_simd_groups floats

    current_offset = (uintptr_t)(tg_simd_reduce_scratch + num_simd_groups);
    current_offset = (current_offset + kAlignmentMask) & ~kAlignmentMask;
    threadgroup float* tg_simd_exp_sums_scratch = (threadgroup float*)current_offset; // num_simd_groups floats

    current_offset = (uintptr_t)(tg_simd_exp_sums_scratch + num_simd_groups);
    current_offset = (current_offset + kAlignmentMask) & ~kAlignmentMask;
    threadgroup float2* tg_global_stats = (threadgroup float2*)current_offset; // {m_global, s_global}

    current_offset = (uintptr_t)(tg_global_stats + 1);
    current_offset = (current_offset + kAlignmentMask) & ~kAlignmentMask;
    threadgroup float* tg_s_global_comp = (threadgroup float*)current_offset; // Kahan summation compensation

    current_offset = (uintptr_t)(tg_s_global_comp + 1);
    current_offset = (current_offset + kAlignmentMask) & ~kAlignmentMask;
    threadgroup float4* tg_simd_v_chunk_sums = (threadgroup float4*)current_offset; // num_simd_groups float4s (for Pass 2 o_tile reduction)

    // K_tile for caching K-vectors in threadgroup memory - after SIMD v_chunk_sums
    current_offset = (uintptr_t)(tg_simd_v_chunk_sums + num_simd_groups);
    current_offset = (current_offset + kAlignmentMask) & ~kAlignmentMask;

    // K_tile and V_tile for spatial KV tiling
    // These will hold PAGE_SUB_TILE_TOKEN_COUNT tokens for all unique KV heads needed by this Q-head block
    threadgroup half* K_tile = (threadgroup half*)current_offset;
    current_offset += params.tile_size_T_runtime * params.head_dim * sizeof(half);
    current_offset = (current_offset + kAlignmentMask) & ~kAlignmentMask;

    threadgroup half* V_tile = (threadgroup half*)current_offset;
    current_offset += params.tile_size_T_runtime * params.head_dim * sizeof(half);
    current_offset = (current_offset + kAlignmentMask) & ~kAlignmentMask;

    // Threadgroup buffer for current sequence's page-table slice
    // For Pass 1, we'll use a simplified page table since we're processing a single page
    threadgroup uint* tg_page_table_slice = (threadgroup uint*)current_offset;
    current_offset += params.max_logical_blocks_per_seq * sizeof(uint);
    current_offset = (current_offset + kAlignmentMask) & ~kAlignmentMask;

    // Final padding guard (mirrors host-side calculation)
    current_offset += 32;
    current_offset = (current_offset + kAlignmentMask) & ~kAlignmentMask;

    // --- Step 3: Identify unique KV heads for this Q-head block ---
    // This is done once before the sub-tiling loop
    uint unique_kv_heads[PREFILL_PASS1_Q_HEAD_BLOCK_SIZE_CONST];  // Max possible unique KV heads
    uint num_unique_kv_heads = 0;

    // // Find unique KV heads mapped from our Q heads
    // for (uint q_idx = 0; q_idx < num_q_heads_for_this_block; ++q_idx) {
    //     uint global_q_head = q_head_start_index_in_model + q_idx;
    //     uint target_kv_head = map_q_to_kv_head(global_q_head, params.num_q_heads, params.num_kv_heads);

    //     // Check if this KV head is already in our unique list
    //     bool already_exists = false;
    //     for (uint i = 0; i < num_unique_kv_heads; ++i) {
    //         if (unique_kv_heads[i] == target_kv_head) {
    //             already_exists = true;
    //             break;
    //         }
    //     }

    //     if (!already_exists) {
    //         unique_kv_heads[num_unique_kv_heads++] = target_kv_head;
    //     }
    // }

    // // Calculate number of sub-tiles needed to process the entire page
    // uint num_sub_tiles_in_page = (params.tokens_per_page + PAGE_SUB_TILE_TOKEN_COUNT - 1) / PAGE_SUB_TILE_TOKEN_COUNT;

    // // Calculate tile stride for indexing into K_tile/V_tile
    // // Layout: [unique_kv_head_idx][token_in_sub_tile][dim]
    // const uint tile_stride_per_kv_head = PAGE_SUB_TILE_TOKEN_COUNT * params.head_dim;
    // const uint chunks_per_row = params.head_dim / 4;  // Vectorized loading

    // // First, collect unique sequence indices from relevant queries
    // // We'll process queries in groups by sequence to minimize page table loads
    // const uint MAX_SEQUENCES_PER_PAGE = 32;  // Reasonable limit for on-chip tracking
    // int unique_sequences[MAX_SEQUENCES_PER_PAGE];
    // uint num_unique_sequences = 0;

    // // Scan relevant queries to find unique sequences
    // for (uint relevance_idx = relevance_start_idx; relevance_idx < relevance_end_idx; ++relevance_idx) {
    //     uint query_token_global_idx = relevant_query_indices[relevance_idx];
    //     int original_sequence_idx = query_to_seq_map_in[query_token_global_idx];

    //     // Check if we already have this sequence
    //     bool found = false;
    //     for (uint i = 0; i < num_unique_sequences; ++i) {
    //         if (unique_sequences[i] == original_sequence_idx) {
    //             found = true;
    //             break;
    //         }
    //     }

    //     if (!found && num_unique_sequences < MAX_SEQUENCES_PER_PAGE) {
    //         unique_sequences[num_unique_sequences++] = original_sequence_idx;
    //     }
    // }

    // // Process queries by sequence
    // for (uint seq_iter = 0; seq_iter < num_unique_sequences; ++seq_iter) {
    //     int current_sequence_idx = unique_sequences[seq_iter];
    //     // Load the page table for this sequence ONCE
    //     for (uint blk = local_thread_idx; blk < params.max_logical_blocks_per_seq; blk += tg_dim.x) {
    //         uint flat_idx_in_global_pt = current_sequence_idx * params.max_logical_blocks_per_seq + blk;
    //         tg_page_table_slice[blk] = page_table_in[flat_idx_in_global_pt];
    //     }
    //     threadgroup_barrier(mem_flags::mem_threadgroup);

    //     // TODO: Implement this
    // } // End sequence loop

} // End of kernel
