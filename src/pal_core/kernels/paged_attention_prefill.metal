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

    // Calculate number of sub-tiles needed to process the entire page
    uint num_sub_tiles_in_page = (params.tokens_per_page + PAGE_SUB_TILE_TOKEN_COUNT - 1) / PAGE_SUB_TILE_TOKEN_COUNT;

    // --- RESTRUCTURED: Sub-tile loop is now outermost ---
    for (uint sub_tile_iter = 0; sub_tile_iter < num_sub_tiles_in_page; ++sub_tile_iter) {
        // Calculate token range for current sub-tile
        uint current_sub_tile_start_token_offset_in_page = sub_tile_iter * PAGE_SUB_TILE_TOKEN_COUNT;
        uint current_sub_tile_actual_len = min(PAGE_SUB_TILE_TOKEN_COUNT,
                                               params.tokens_per_page - current_sub_tile_start_token_offset_in_page);

        // --- Spatial K/V Tile Loading (moved here, outside query loops) ---
        // Load K and V data for all unique KV heads for this sub-tile

        // Calculate tile stride for indexing into K_tile/V_tile
        // Layout: [unique_kv_head_idx][token_in_sub_tile][dim]
        const uint tile_stride_per_kv_head = PAGE_SUB_TILE_TOKEN_COUNT * params.head_dim;
        const uint chunks_per_row = params.head_dim / 4;  // Vectorized loading

        // We need to determine which logical block this page represents
        // This will be the same for all queries, so we can do it once
        // For now, we'll determine it per sequence when we load page tables

        // First, collect unique sequence indices from relevant queries
        // We'll process queries in groups by sequence to minimize page table loads
        const uint MAX_SEQUENCES_PER_PAGE = 32;  // Reasonable limit for on-chip tracking
        int unique_sequences[MAX_SEQUENCES_PER_PAGE];
        uint num_unique_sequences = 0;

        // Scan relevant queries to find unique sequences
        for (uint relevance_idx = relevance_start_idx; relevance_idx < relevance_end_idx; ++relevance_idx) {
            uint query_token_global_idx = relevant_query_indices[relevance_idx];
            uint history_start_on_page = relevant_history_starts[relevance_idx];
            uint history_count_on_page = relevant_history_counts[relevance_idx];
            uint history_end_on_page = history_start_on_page + history_count_on_page;

            // Check if this query's history overlaps with current sub-tile
            uint sub_tile_start = current_sub_tile_start_token_offset_in_page;
            uint sub_tile_end = sub_tile_start + current_sub_tile_actual_len;

            if (history_end_on_page <= sub_tile_start || history_start_on_page >= sub_tile_end) {
                continue;  // No overlap with this sub-tile
            }

            int original_sequence_idx = query_to_seq_map_in[query_token_global_idx];

            // Check if we already have this sequence
            bool found = false;
            for (uint i = 0; i < num_unique_sequences; ++i) {
                if (unique_sequences[i] == original_sequence_idx) {
                    found = true;
                    break;
                }
            }

            if (!found && num_unique_sequences < MAX_SEQUENCES_PER_PAGE) {
                unique_sequences[num_unique_sequences++] = original_sequence_idx;
            }
        }

        // Now iterate through unique sequences
        for (uint seq_iter = 0; seq_iter < num_unique_sequences; ++seq_iter) {
            int current_sequence_idx = unique_sequences[seq_iter];

            // Load the page table for this sequence ONCE
            for (uint blk = local_thread_idx; blk < params.max_logical_blocks_per_seq; blk += tg_dim.x) {
                uint flat_idx_in_global_pt = current_sequence_idx * params.max_logical_blocks_per_seq + blk;
                tg_page_table_slice[blk] = page_table_in[flat_idx_in_global_pt];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Find which logical block our assigned page corresponds to
            uint logical_block_for_page = 0;
            bool found_logical_block = false;
            for (uint blk = 0; blk < params.max_logical_blocks_per_seq; ++blk) {
                if (tg_page_table_slice[blk] == assigned_global_kv_page_id) {
                    logical_block_for_page = blk;
                    found_logical_block = true;
                    break;
                }
            }

            // Calculate the base token position in the sequence for tokens on this page
            uint page_base_token_in_seq = logical_block_for_page * params.tokens_per_page;

            // Load K/V vectors for this sub-tile if not already loaded
            // We only need to load once per sub-tile, not per sequence
            if (seq_iter == 0) {
                // Load K vectors for all unique KV heads
                for (uint unique_kv_idx_load = 0; unique_kv_idx_load < num_unique_kv_heads; ++unique_kv_idx_load) {
                    uint kv_head_idx = unique_kv_heads[unique_kv_idx_load];

                    // Each SIMD group loads tokens cooperatively
                    for (uint tok_idx = simd_group_id; tok_idx < current_sub_tile_actual_len; tok_idx += num_simd_groups) {
                        // Calculate absolute token position on the page
                        uint token_pos_on_page = current_sub_tile_start_token_offset_in_page + tok_idx;

                        // Convert to absolute position in the sequence
                        uint token_pos_in_seq = page_base_token_in_seq + token_pos_on_page;

                        // Get K vector pointer
                        device const half* k_vec_ptr = fetch_kv_pointer_optimized(
                            true,  // is_k_vector
                            token_pos_in_seq,
                            kv_head_idx,
                            k_cache_pool_in,
                            v_cache_pool_in,
                            tg_page_table_slice,
                            params
                        );

                        // Calculate destination in K_tile
                        threadgroup half* k_tile_dest = K_tile + (unique_kv_idx_load * tile_stride_per_kv_head +
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
                for (uint unique_kv_idx_load = 0; unique_kv_idx_load < num_unique_kv_heads; ++unique_kv_idx_load) {
                    uint kv_head_idx = unique_kv_heads[unique_kv_idx_load];

                    // Each SIMD group loads tokens cooperatively
                    for (uint tok_idx = simd_group_id; tok_idx < current_sub_tile_actual_len; tok_idx += num_simd_groups) {
                        // Calculate absolute token position on the page
                        uint token_pos_on_page = current_sub_tile_start_token_offset_in_page + tok_idx;

                        // Convert to absolute position in the sequence
                        uint token_pos_in_seq = page_base_token_in_seq + token_pos_on_page;

                        // Get V vector pointer
                        device const half* v_vec_ptr = fetch_kv_pointer_optimized(
                            false,  // is_k_vector = false for V
                            token_pos_in_seq,
                            kv_head_idx,
                            k_cache_pool_in,
                            v_cache_pool_in,
                            tg_page_table_slice,
                            params
                        );

                        // Calculate destination in V_tile
                        threadgroup half* v_tile_dest = V_tile + (unique_kv_idx_load * tile_stride_per_kv_head +
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
            }

            // Now process all queries that belong to this sequence
            for (uint relevance_idx = relevance_start_idx; relevance_idx < relevance_end_idx; ++relevance_idx) {
                uint query_token_global_idx = relevant_query_indices[relevance_idx];
                int query_sequence_idx = query_to_seq_map_in[query_token_global_idx];

                // Skip if not for current sequence
                if (query_sequence_idx != current_sequence_idx) {
                    continue;
                }

                uint history_start_on_page = relevant_history_starts[relevance_idx];
                uint history_count_on_page = relevant_history_counts[relevance_idx];
                uint history_end_on_page = history_start_on_page + history_count_on_page;

                // Check if this query's history overlaps with current sub-tile
                uint sub_tile_start = current_sub_tile_start_token_offset_in_page;
                uint sub_tile_end = sub_tile_start + current_sub_tile_actual_len;

                if (history_end_on_page <= sub_tile_start || history_start_on_page >= sub_tile_end) {
                    continue;  // No overlap with this sub-tile
                }

                // Calculate actual overlap range
                uint overlap_start = max(history_start_on_page, sub_tile_start);
                uint overlap_end = min(history_end_on_page, sub_tile_end);

                // --- Loop over Q-heads in this TG's assigned block ---
                for (uint q_head_offset_in_block = 0; q_head_offset_in_block < num_q_heads_for_this_block; ++q_head_offset_in_block) {
                    uint current_global_q_head_idx = q_head_start_index_in_model + q_head_offset_in_block;

                    // --- Initialize Sub-Tile Accumulators ---
                    float m_sub_tile = -INFINITY;
                    float s_sub_tile = 0.0f;
                    float o_partial_sub_tile[kMaxHeadDimMetal];
                    for (uint d = 0; d < params.head_dim; ++d) {
                        o_partial_sub_tile[d] = 0.0f;
                    }

                    // Load Q vector for this query token and Q head
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

                    // --- Innermost History Loop ---
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
                            float scale_factor = precise::exp(m_sub_tile - score);
                            s_sub_tile *= scale_factor;

                            // Rescale partial output accumulator
                            for (uint d = 0; d < params.head_dim; ++d) {
                                o_partial_sub_tile[d] *= scale_factor;
                            }

                            m_sub_tile = score;
                        }

                        // Compute exp(score - m_sub_tile) and accumulate
                        float exp_score = precise::exp(max(score - m_sub_tile, params.log_exp_min_clamp));
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

                    // --- Write Sub-Tile Results to Global Memory ---
                    // Since we restructured to process sub-tiles first, we write results per sub-tile
                    // Calculate output index based on C++ allocation pattern: [TotalQueryTokens][NumTotalQHeads][NumActivePages]
                    uint output_idx_for_m_s = query_token_global_idx * params.num_q_heads * params.num_active_pages_in_batch +
                                              current_global_q_head_idx * params.num_active_pages_in_batch +
                                              assigned_page_index;
                    uint output_idx_for_o_base = output_idx_for_m_s * params.head_dim;

                    // Write sub-tile results
                    // Note: With the restructured loops, we're accumulating across sub-tiles differently
                    // For now, write the sub-tile results directly (Pass 2 will handle aggregation)
                    if (local_thread_idx == 0) {
                        // Use atomic max for m_locals to handle multiple sub-tiles
                        atomic_store_explicit((device atomic<float>*)&m_locals_pass1_out[output_idx_for_m_s],
                                            m_sub_tile, memory_order_relaxed);

                        // For s_locals and o_partials, we need atomic add since multiple sub-tiles contribute
                        atomic_fetch_add_explicit((device atomic<float>*)&s_locals_pass1_out[output_idx_for_m_s],
                                                s_sub_tile, memory_order_relaxed);

                        // Write partial output vector with atomic add
                        for (uint d = 0; d < params.head_dim; ++d) {
                            atomic_fetch_add_explicit((device atomic<float>*)&o_partials_pass1_out[output_idx_for_o_base + d],
                                                    o_partial_sub_tile[d], memory_order_relaxed);
                        }
                    }

                } // End Q-head loop
            } // End query loop for this sequence
        } // End sequence loop
    } // End sub-tile loop

} // End of kernel
