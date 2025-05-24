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

// --- Step 1: Page Sub-Tiling Constants ---
// Number of tokens from the assigned page processed per internal K/V load
// This value is tuned for HD=128 with Q-head block of 8 (mapping to 4 unique KV-heads in GQA)
static constant uint PAGE_SUB_TILE_TOKEN_COUNT = 12;

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

    // Find unique KV heads mapped from our Q heads
    for (uint q_idx = 0; q_idx < num_q_heads_for_this_block; ++q_idx) {
        uint global_q_head = q_head_start_index_in_model + q_idx;
        uint target_kv_head = map_q_to_kv_head(global_q_head, params.num_q_heads, params.num_kv_heads);

        // Check if this KV head is already in our unique list
        bool already_exists = false;
        for (uint i = 0; i < num_unique_kv_heads; ++i) {
            if (unique_kv_heads[i] == target_kv_head) {
                already_exists = true;
                break;
            }
        }

        if (!already_exists) {
            unique_kv_heads[num_unique_kv_heads++] = target_kv_head;
        }
    }

    // Set up simplified page table for fetch_kv_pointer
    // Since we're processing a single page, create a minimal page table
    if (local_thread_idx == 0) {
        for (uint i = 0; i < params.max_logical_blocks_per_seq; ++i) {
            // For the tokens on our assigned page, set the physical page ID
            // For simplicity, assume our page covers logical blocks starting from 0
            tg_page_table_slice[i] = (i == 0) ? assigned_global_kv_page_id : 0;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // --- Step 2: Page Sub-Tiling Loop ---
    // Calculate number of sub-tiles needed to process the entire page
    uint num_sub_tiles_in_page = (params.tokens_per_page + PAGE_SUB_TILE_TOKEN_COUNT - 1) / PAGE_SUB_TILE_TOKEN_COUNT;

    // Main loop iterating through sub-tiles of the assigned page
    for (uint sub_tile_iter = 0; sub_tile_iter < num_sub_tiles_in_page; ++sub_tile_iter) {
        // Calculate token range for current sub-tile
        uint current_sub_tile_start_token_offset_in_page = sub_tile_iter * PAGE_SUB_TILE_TOKEN_COUNT;
        uint current_sub_tile_actual_len = min(PAGE_SUB_TILE_TOKEN_COUNT,
                                               params.tokens_per_page - current_sub_tile_start_token_offset_in_page);

        // --- Step 3: Spatial K/V Tile Loading ---
        // Load K and V data for all unique KV heads for this sub-tile

        // Calculate tile stride for indexing into K_tile/V_tile
        // Layout: [unique_kv_head_idx][token_in_sub_tile][dim]
        const uint tile_stride_per_kv_head = PAGE_SUB_TILE_TOKEN_COUNT * params.head_dim;
        const uint chunks_per_row = params.head_dim / 4;  // Vectorized loading

        // Load K vectors for all unique KV heads
        for (uint unique_kv_idx = 0; unique_kv_idx < num_unique_kv_heads; ++unique_kv_idx) {
            uint kv_head_idx = unique_kv_heads[unique_kv_idx];

            // Each SIMD group loads tokens cooperatively
            for (uint tok_idx = simd_group_id; tok_idx < current_sub_tile_actual_len; tok_idx += num_simd_groups) {
                // Calculate absolute token position on the page
                uint token_pos_on_page = current_sub_tile_start_token_offset_in_page + tok_idx;

                // Get K vector pointer
                device const half* k_vec_ptr = fetch_kv_pointer(
                    true,  // is_k_vector
                    token_pos_on_page,
                    kv_head_idx,
                    k_cache_pool_in,
                    v_cache_pool_in,
                    tg_page_table_slice,
                    params
                );

                // Calculate destination in K_tile
                threadgroup half* k_tile_dest = K_tile + (unique_kv_idx * tile_stride_per_kv_head +
                                                          tok_idx * params.head_dim);
                threadgroup half4* k_tile_dest_h4 = reinterpret_cast<threadgroup half4*>(k_tile_dest);

                if (k_vec_ptr != nullptr) {
                    // Vectorized load by SIMD lanes
                    device const half4* k_vec_h4 = reinterpret_cast<device const half4*>(k_vec_ptr);
                    for (uint chunk_idx = simd_lane_id; chunk_idx < chunks_per_row; chunk_idx += actual_simd_width) {
                        k_tile_dest_h4[chunk_idx] = k_vec_h4[chunk_idx];
                    }
                } else {
                    // Zero fill if pointer is null
                    for (uint chunk_idx = simd_lane_id; chunk_idx < chunks_per_row; chunk_idx += actual_simd_width) {
                        k_tile_dest_h4[chunk_idx] = half4(0.0h);
                    }
                }
            }
        }

        // Ensure K vectors are loaded before loading V vectors
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Load V vectors for all unique KV heads (similar pattern)
        for (uint unique_kv_idx = 0; unique_kv_idx < num_unique_kv_heads; ++unique_kv_idx) {
            uint kv_head_idx = unique_kv_heads[unique_kv_idx];

            // Each SIMD group loads tokens cooperatively
            for (uint tok_idx = simd_group_id; tok_idx < current_sub_tile_actual_len; tok_idx += num_simd_groups) {
                // Calculate absolute token position on the page
                uint token_pos_on_page = current_sub_tile_start_token_offset_in_page + tok_idx;

                // Get V vector pointer
                device const half* v_vec_ptr = fetch_kv_pointer(
                    false,  // is_k_vector = false for V
                    token_pos_on_page,
                    kv_head_idx,
                    k_cache_pool_in,
                    v_cache_pool_in,
                    tg_page_table_slice,
                    params
                );

                // Calculate destination in V_tile
                threadgroup half* v_tile_dest = V_tile + (unique_kv_idx * tile_stride_per_kv_head +
                                                          tok_idx * params.head_dim);
                threadgroup half4* v_tile_dest_h4 = reinterpret_cast<threadgroup half4*>(v_tile_dest);

                if (v_vec_ptr != nullptr) {
                    // Vectorized load by SIMD lanes
                    device const half4* v_vec_h4 = reinterpret_cast<device const half4*>(v_vec_ptr);
                    for (uint chunk_idx = simd_lane_id; chunk_idx < chunks_per_row; chunk_idx += actual_simd_width) {
                        v_tile_dest_h4[chunk_idx] = v_vec_h4[chunk_idx];
                    }
                } else {
                    // Zero fill if pointer is null
                    for (uint chunk_idx = simd_lane_id; chunk_idx < chunks_per_row; chunk_idx += actual_simd_width) {
                        v_tile_dest_h4[chunk_idx] = half4(0.0h);
                    }
                }
            }
        }

        // Ensure all K/V data is loaded before processing
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // --- Step 4: Verification Output ---
        // For the first sub-tile of the first threadgroup, output verification data
        if (sub_tile_iter == 0 && local_thread_idx == 0 && tg_pos_in_grid.x == 0 && tg_pos_in_grid.y == 0) {
            // Write debug info about loaded K/V data
            m_locals_pass1_out[0] = float(assigned_global_kv_page_id);
            m_locals_pass1_out[1] = float(num_unique_kv_heads);
            m_locals_pass1_out[2] = float(current_sub_tile_actual_len);

            // Write first few values from K_tile for verification
            if (num_unique_kv_heads > 0 && current_sub_tile_actual_len > 0) {
                // First element of K vector for first unique KV head, first token
                m_locals_pass1_out[3] = float(K_tile[0]);
                // Second element
                if (params.head_dim > 1) {
                    m_locals_pass1_out[4] = float(K_tile[1]);
                }
                // First element of V vector for first unique KV head, first token
                m_locals_pass1_out[5] = float(V_tile[0]);
            }
        }

        // --- Step 1: Loop Through Relevant Query Tokens ---
        // Iterate through relevant queries for this page and check overlap with current sub-tile
        for (uint relevance_idx = relevance_start_idx; relevance_idx < relevance_end_idx; ++relevance_idx) {
            uint query_token_global_idx = relevant_query_indices[relevance_idx];
            uint history_start_on_page = relevant_history_starts[relevance_idx];
            uint history_count_on_page = relevant_history_counts[relevance_idx];

            // Calculate history range for this query on this page
            uint history_end_on_page = history_start_on_page + history_count_on_page; // Exclusive

            // Check if this query's history overlaps with current sub-tile
            uint sub_tile_start = current_sub_tile_start_token_offset_in_page;
            uint sub_tile_end = sub_tile_start + current_sub_tile_actual_len; // Exclusive

            // Skip if no overlap
            if (history_end_on_page <= sub_tile_start || history_start_on_page >= sub_tile_end) {
                continue;
            }

            // Calculate actual overlap range
            uint overlap_start = max(history_start_on_page, sub_tile_start);
            uint overlap_end = min(history_end_on_page, sub_tile_end);

            // --- Step 2: Q-Head Block Processing Loop ---
            for (uint q_head_offset_in_block = 0; q_head_offset_in_block < num_q_heads_for_this_block; ++q_head_offset_in_block) {
                uint current_global_q_head_idx = q_head_start_index_in_model + q_head_offset_in_block;

                // Load Q vector for this query token and Q head
                // Calculate Q vector pointer
                device const half* q_vector_ptr;
                if (params.num_q_heads > 1) {
                    // 3D queries: [num_tokens, num_q_heads, head_dim]
                    ulong q_offset = (ulong)query_token_global_idx * params.num_q_heads * params.head_dim +
                                     (ulong)current_global_q_head_idx * params.head_dim;
                    q_vector_ptr = queries_in + q_offset;
                } else {
                    // 2D queries: [num_tokens, head_dim]
                    q_vector_ptr = queries_in + (query_token_global_idx * params.head_dim);
                }

                // Load Q vector into shared memory with scaling
                threadgroup float4* q_vec_f4 = reinterpret_cast<threadgroup float4*>(q_shmem);
                for (uint chunk = local_thread_idx; chunk < params.head_dim / 4; chunk += tg_dim.x) {
                    uint base_offset = chunk * 4;
                    half qh0 = q_vector_ptr[base_offset + 0];
                    half qh1 = q_vector_ptr[base_offset + 1];
                    half qh2 = q_vector_ptr[base_offset + 2];
                    half qh3 = q_vector_ptr[base_offset + 3];

                    float4 q_float_chunk = float4(qh0, qh1, qh2, qh3) * params.inv_sqrt_head_dim;
                    q_vec_f4[chunk] = q_float_chunk;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                // --- Step 3: Per-Q-Head Accumulator Initialization ---
                // Initialize accumulators for this Q-head's interaction with current sub-tile
                float m_sub_tile = -INFINITY;
                float s_sub_tile = 0.0f;
                float o_partial_sub_tile[kMaxHeadDimMetal];
                for (uint d = 0; d < params.head_dim; ++d) {
                    o_partial_sub_tile[d] = 0.0f;
                }

                // Determine which KV head this Q head maps to
                uint target_kv_head_idx = map_q_to_kv_head(current_global_q_head_idx,
                                                           params.num_q_heads,
                                                           params.num_kv_heads);

                // Find which unique KV head index this maps to in our loaded data
                uint unique_kv_idx = 0;
                for (uint i = 0; i < num_unique_kv_heads; ++i) {
                    if (unique_kv_heads[i] == target_kv_head_idx) {
                        unique_kv_idx = i;
                        break;
                    }
                }

                // --- Step 4: Innermost Loop - History Tokens in Sub-Tile ---
                for (uint hist_idx_in_sub_tile = 0; hist_idx_in_sub_tile < current_sub_tile_actual_len; ++hist_idx_in_sub_tile) {
                    uint actual_hist_token_pos_on_page = current_sub_tile_start_token_offset_in_page + hist_idx_in_sub_tile;

                    // Check if this history position is within the valid range for this query
                    if (actual_hist_token_pos_on_page < overlap_start || actual_hist_token_pos_on_page >= overlap_end) {
                        continue;
                    }

                    // Get K vector from K_tile
                    threadgroup const half* k_vector_ptr = K_tile +
                        (unique_kv_idx * tile_stride_per_kv_head + hist_idx_in_sub_tile * params.head_dim);

                    // Compute QK^T dot product
                    float score = dot_product_qk(q_shmem, k_vector_ptr, params);

                    // Update running max
                    if (score > m_sub_tile) {
                        // Rescale existing sum
                        float scale_factor = exp(m_sub_tile - score);
                        s_sub_tile *= scale_factor;

                        // Rescale partial output accumulator
                        for (uint d = 0; d < params.head_dim; ++d) {
                            o_partial_sub_tile[d] *= scale_factor;
                        }

                        m_sub_tile = score;
                    }

                    // Compute exp(score - m_sub_tile) and accumulate
                    float exp_score = exp(max(score - m_sub_tile, params.log_exp_min_clamp));
                    s_sub_tile += exp_score;

                    // Get V vector from V_tile and accumulate weighted V
                    threadgroup const half* v_vector_ptr = V_tile +
                        (unique_kv_idx * tile_stride_per_kv_head + hist_idx_in_sub_tile * params.head_dim);

                    for (uint d = 0; d < params.head_dim; d += 4) {
                        float4 v_chunk = float4(*((threadgroup const half4*)(v_vector_ptr + d)));
                        float4 weighted_v = v_chunk * exp_score;

                        o_partial_sub_tile[d] += weighted_v.x;
                        if (d + 1 < params.head_dim) o_partial_sub_tile[d + 1] += weighted_v.y;
                        if (d + 2 < params.head_dim) o_partial_sub_tile[d + 2] += weighted_v.z;
                        if (d + 3 < params.head_dim) o_partial_sub_tile[d + 3] += weighted_v.w;
                    }
                }

                // --- Step 5: Store Sub-Tile Results (Temporary) ---
                // For verification, write results for first query, first Q-head, first sub-tile
                if (local_thread_idx == 0 && sub_tile_iter == 0 &&
                    q_head_offset_in_block == 0 && relevance_idx == relevance_start_idx) {
                    // Write to debug locations in output buffers
                    m_locals_pass1_out[7] = m_sub_tile;
                    s_locals_pass1_out[7] = s_sub_tile;

                    // Write first few elements of partial output
                    for (uint d = 0; d < min(4u, params.head_dim); ++d) {
                        o_partials_pass1_out[7 * params.head_dim + d] = (half)o_partial_sub_tile[d];
                    }
                }

                // TODO: In the next phase, aggregate these sub-tile results across all sub-tiles
                // to produce final m_local, s_local, o_partial for this (query, q_head) pair
            } // End Q-head loop
        } // End relevant query loop
    }

    // Temporary: write dummy values to output buffers
    if (local_thread_idx == 0) {
        // Calculate output indices for this threadgroup
        uint output_base_idx = assigned_page_index * (params.num_q_heads / PREFILL_PASS1_Q_HEAD_BLOCK_SIZE_CONST + 1) * PREFILL_PASS1_Q_HEAD_BLOCK_SIZE_CONST +
                              q_head_block_idx * PREFILL_PASS1_Q_HEAD_BLOCK_SIZE_CONST;

        // Write dummy values for each Q head in this block
        for (uint q_idx = 0; q_idx < num_q_heads_for_this_block; ++q_idx) {
            m_locals_pass1_out[output_base_idx + q_idx] = -999.0f;  // Dummy max score
            s_locals_pass1_out[output_base_idx + q_idx] = 0.001f;   // Dummy sum exponential

            // Write dummy partial output values
            uint o_base = (output_base_idx + q_idx) * params.head_dim;
            for (uint d = 0; d < params.head_dim; ++d) {
                o_partials_pass1_out[o_base + d] = (half)0.0f;
            }
        }
    }

} // End of kernel
