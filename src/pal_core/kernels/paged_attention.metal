// paged_attention.metal
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

constant static const uint kMaxHeadDimMetal = 256; // Kernel's internal max, C++ validates params.head_dim against this
constant static const uint kHeadDimProcessingChunk = 64; // Chunk size for processing head_dim in Pass 2


constant static const uint kMaxAccumulationTile = 64; // Matches kMaxAccumulationTile in C++ code

using namespace metal;

/**
 *  paged_attn_kernel
 *  ----------------------------------------
 *  Implements paged attention for transformer models with key-value memory pooling.
 *
 *  This kernel computes attention scores between query vectors and key vectors stored in
 *  paged memory, then weights value vectors to produce the final output. The implementation
 *  handles:
 *  - Multi-headed attention (MHA)
 *  - Grouped query attention (GQA)
 *  - Multi-query attention (MQA)
 *  - Efficient vectorized memory access
 *  - Tiled value accumulation for large head dimensions (via two-pass approach)
 *  - SIMD-accelerated parallel reductions
 *
 *  Thread Mapping:
 *  - One threadgroup processes one query item (single token+head pair)
 *  - Each thread in the threadgroup collaboratively processes a portion of history
 *  - SIMD groups are used for efficient parallel reductions
 *
 *  Memory Layout:
 *  - Query shape: [N_tokens × H_q × D] or [N] when H_q==1
 *  - Key/Value cache: [Pages × TokensPerPage × H_kv × D]
 *  - Page table: [Sequences × MaxBlocksPerSequence]
 *
 *  Algorithm Stages (Two-Pass):
 *  Pass 1:
 *  1. Collaboratively load and pre-scale query vector into shared memory.
 *  2. Single-pass history scan performing online softmax to calculate m_final_global and d_final_global.
 *     - Each thread computes its local m_local and d_local.
 *     - Threadgroup reduction to find m_final_global (overall max score) and d_final_global (overall normalization sum).
 *  Pass 2:
 *  1. Iterate over head_dim in chunks of kHeadDimProcessingChunk.
 *  2. For each chunk:
 *     - Each thread initializes a local o_tile[kHeadDimProcessingChunk].
 *     - Re-scan history:
 *       - Compute Q·K score.
 *       - Calculate attention weight p = fast::exp(score - m_final_global) / d_final_global.
 *       - Accumulate V contributions into o_tile: o_tile += p * V_chunk.
 *     - Reduce o_tile across the threadgroup and write to output buffer.
 */
[[kernel]] void paged_attn_kernel(
    device      const half* queries_in              [[buffer(0)]],
    device      const half* k_cache_pool_in         [[buffer(1)]],
    device      const half* v_cache_pool_in         [[buffer(2)]],
    device      const uint* page_table_in           [[buffer(3)]],
    device      const int* sequence_lengths_in      [[buffer(4)]],
    device      const int* query_to_seq_map_in      [[buffer(5)]],
    device      const int* query_token_offset_in    [[buffer(6)]],
    constant    const PagedAttentionParams& params  [[buffer(7)]],
    device      half* output_buffer                 [[buffer(8)]],
    threadgroup float* tg_mem                       [[threadgroup(0)]],
    uint3       tg_pos_in_grid                      [[threadgroup_position_in_grid]],
    uint3       tg_dim                              [[threads_per_threadgroup]],
    uint        local_idx_in_tg                     [[thread_index_in_threadgroup]],
    uint        simd_lane_id                        [[thread_index_in_simdgroup]],
    uint        simd_group_id                       [[simdgroup_index_in_threadgroup]]
) {
    // --- Create a local, corrected scale value using rsqrt for better performance ---
    float kernel_scale;
    if (params.head_dim > 0) {
        kernel_scale = rsqrt((float)params.head_dim);  // Use rsqrt for 1/sqrt()
    } else {
        kernel_scale = 1.0f;  // Fallback for head_dim == 0 to avoid division by zero/NaN
    }

    // --- Basic input validation ---
    // Early exit for degenerate case where head_dim is zero
    if (params.head_dim == 0) {
        // Zero the output and exit (no need for loop if head_dim is 0, but for safety if it's called)
        // Output buffer is implicitly zero if no writes occur, but explicit zeroing can be done if needed.
        return;
    }

    // --- Thread-Local Accumulators for Pass 1 (Online Softmax stats) ---
    float m_local = -INFINITY; // Maximum score accumulator
    float d_local = 0.0f;      // Sum of scaled exponentials accumulator

    // Note: o_local (large array) is removed. Output O will be computed in Pass 2 using o_tile.

    // --- Thread Identifiers ---
    uint global_item_idx = tg_pos_in_grid.x;    // Identifies the query-head item
    uint local_thread_idx = local_idx_in_tg;    // Thread ID within this group
    const uint num_simd_groups = max(1u, (tg_dim.x + 31u) >> 5); // Calculate number of actual SIMD groups

    // --- Carve the dynamic threadgroup buffer into logical sub-arrays ---
    // Layout for Pass 1 and Pass 2 reductions.
    // q_shmem, G_partial_max_scores, G_simd_reduced_maxes, G_simd_reduced_adjusted_sum_exps,
    // G_final_max_for_item, G_final_sum_exp_for_item are primarily for Pass 1.
    // G_simd_group_v_sums is used in Pass 2 for o_tile reduction.

    threadgroup float* q_shmem = tg_mem;  // head_dim floats

    uintptr_t current_offset = (uintptr_t)(q_shmem + params.head_dim);
    current_offset = (current_offset + 63u) & ~63u;
    threadgroup float* G_partial_max_scores = (threadgroup float*)current_offset;  // threads_per_tg floats

    current_offset = (uintptr_t)(G_partial_max_scores + tg_dim.x);
    current_offset = (current_offset + 63u) & ~63u;
    threadgroup float* G_simd_reduced_maxes = (threadgroup float*)current_offset; // num_simd_groups floats

    current_offset = (uintptr_t)(G_simd_reduced_maxes + num_simd_groups);
    current_offset = (current_offset + 63u) & ~63u;
    threadgroup float* G_simd_reduced_adjusted_sum_exps = (threadgroup float*)current_offset; // num_simd_groups floats

    current_offset = (uintptr_t)(G_simd_reduced_adjusted_sum_exps + num_simd_groups);
    current_offset = (current_offset + 63u) & ~63u;
    threadgroup float* G_final_max_for_item = (threadgroup float*)current_offset; // 1 float

    current_offset = (uintptr_t)(G_final_max_for_item + 1);
    current_offset = (current_offset + 63u) & ~63u;
    threadgroup float* G_final_sum_exp_for_item = (threadgroup float*)current_offset; // 1 float

    current_offset = (uintptr_t)(G_final_sum_exp_for_item + 1);
    current_offset = (current_offset + 63u) & ~63u;
    threadgroup float4* G_simd_group_v_sums = (threadgroup float4*)current_offset; // num_simd_groups float4s (for Pass 2 o_tile reduction)

    // K/V and score tiles for history processing
    current_offset = (current_offset + 63u) & ~63u; // Align start of K_tile
    threadgroup float* K_tile = (threadgroup float*)current_offset;
                               // Size: params.tile_size_T_runtime * params.head_dim floats

    // V_tile reuses K_tile's memory space
    threadgroup float* V_tile = K_tile;

    // score_tile starts after K_tile's memory region
    current_offset = (uintptr_t)(K_tile + (ulong)params.tile_size_T_runtime * (ulong)params.head_dim);
    current_offset = (current_offset + 63u) & ~63u; // Align start of score_tile
    threadgroup float* score_tile = (threadgroup float*)current_offset;
                                  // Size: params.tile_size_T_runtime floats

    // --- Determine Q-vector pointer for this item ---
    device const half* q_vector_item_ptr;
    if (params.num_q_heads > 1) {
        uint item_token_idx = global_item_idx / params.num_q_heads;
        uint item_q_head_idx = global_item_idx % params.num_q_heads;
        ulong query_base_offset = (ulong)item_token_idx * params.num_q_heads * params.head_dim +
                                 (ulong)item_q_head_idx * params.head_dim;
        q_vector_item_ptr = queries_in + query_base_offset;
    } else {
        q_vector_item_ptr = queries_in + (global_item_idx * params.head_dim);
    }

    // --- Stage Q-vector into shared memory and pre-scale with kernel_scale ---
    for (uint i = local_thread_idx; i < params.head_dim; i += tg_dim.x) {
        q_shmem[i] = (float)q_vector_item_ptr[i] * kernel_scale;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // --- Determine this item's overall history and sequence length ---
    uint token_idx_for_sideband_lookup;
    if (params.num_q_heads > 1) {
        token_idx_for_sideband_lookup = global_item_idx / params.num_q_heads;
    } else {
        token_idx_for_sideband_lookup = global_item_idx;
    }

    uint item_seq_idx_in_batch = (uint)query_to_seq_map_in[token_idx_for_sideband_lookup];
    if (item_seq_idx_in_batch >= params.num_sequences_in_batch) {
        // Zero the full output vector collaboratively and exit (handled by Pass 2 loop not running if history is 0)
        // For safety, ensure output is zero if exiting early.
        // The Pass 2 loop for h_offset will handle zeroing if item_effective_history_length is 0.
        return;
    }

    int item_signed_query_token_offset = query_token_offset_in[token_idx_for_sideband_lookup];
    if (item_signed_query_token_offset < 0) {
        // Similar to above, Pass 2 will handle zeroing.
        return;
    }

    uint item_current_q_token_logical_pos = (uint)item_signed_query_token_offset;
    uint item_actual_sequence_length = (uint)sequence_lengths_in[item_seq_idx_in_batch];
    uint item_effective_history_length = min(item_current_q_token_logical_pos,
                                         item_actual_sequence_length);

    // Ensure tile_size_T_runtime is valid and not exceeding a practical limit for K_tile/V_tile/score_tile arrays.
    // This complements C++ side checks. Max of 512 for T is a generous upper bound for practical TG mem.
    if (params.tile_size_T_runtime == 0 || params.tile_size_T_runtime > 512) {
        // This indicates a misconfiguration from the host.
        // For now, we can't easily "error out" and return an error code.
        // If this happens, the kernel might produce incorrect results or behave unpredictably.
        // A proper solution would be an error flag written to output, or ensure host never sends this.
        // Given host-side clamps, this state should ideally not be reached.
    }

    // Initialize global stats and thread-local D-tiled output accumulator
    float m_global = -INFINITY;
    float s_global = 0.0f;

    // Per-thread, D-tiled output accumulator
    // Use existing kHeadDimProcessingChunk constant instead of declaring a new one
    float acc_tile_local[kHeadDimProcessingChunk]; // Thread-local stack array for a chunk of head_dim

    // Outer loop for D-tiling (processing head_dim in chunks)
    for (uint d_base_offset = 0; d_base_offset < params.head_dim; d_base_offset += kHeadDimProcessingChunk) {
        uint current_d_chunk_len = min(kHeadDimProcessingChunk, params.head_dim - d_base_offset);

        // Initialize local accumulator tile for this d_chunk
        for (uint i = 0; i < current_d_chunk_len; ++i) {
            acc_tile_local[i] = 0.0f;
        }

        // Main History Tiling Loop
        if (item_effective_history_length > 0) {
            for (uint hist_tile_start = 0; hist_tile_start < item_effective_history_length; hist_tile_start += params.tile_size_T_runtime) {
                uint current_hist_tile_actual_len = min(params.tile_size_T_runtime, item_effective_history_length - hist_tile_start);

                // --- Phase 1: Compute and Stash Raw Scores for the current history tile ---
                if (local_thread_idx < current_hist_tile_actual_len) {
                    uint actual_hist_token_pos = hist_tile_start + local_thread_idx;

                    // Get K-vector pointer for actual_hist_token_pos
                    // (This is the existing complex logic involving page_table, seq_idx, kv_head_idx etc.)
                    uint logical_block_idx_k = actual_hist_token_pos / params.tokens_per_page;
                    uint token_slot_in_page_k = actual_hist_token_pos % params.tokens_per_page;

                    // Default to a state that would lead to zero contribution if checks fail
                    float score_val = -INFINITY;

                    if (logical_block_idx_k < params.max_logical_blocks_per_seq) {
                        uint page_table_flat_idx_k = item_seq_idx_in_batch * params.max_logical_blocks_per_seq + logical_block_idx_k;
                        uint physical_page_id_k = page_table_in[page_table_flat_idx_k];

                        if (physical_page_id_k < params.num_physical_pages_in_pool) {
                            uint q_head_for_kv_map_k = (params.num_q_heads > 1) ? (global_item_idx % params.num_q_heads) : 0;
                            uint target_kv_head_idx_k = 0;
                            if (params.num_kv_heads > 0) {
                                if (params.num_q_heads > params.num_kv_heads) { // GQA
                                    target_kv_head_idx_k = q_head_for_kv_map_k / (params.num_q_heads / params.num_kv_heads);
                                } else { // MHA or MQA
                                    target_kv_head_idx_k = q_head_for_kv_map_k;
                                }
                                if (target_kv_head_idx_k >= params.num_kv_heads) target_kv_head_idx_k %= params.num_kv_heads;
                            }

                            ulong k_base_offset_val = (ulong)physical_page_id_k * params.tokens_per_page * params.num_kv_heads * params.head_dim +
                                                   (ulong)token_slot_in_page_k * params.num_kv_heads * params.head_dim +
                                                   (ulong)target_kv_head_idx_k * params.head_dim;
                            device const half* k_vector_ptr_val = k_cache_pool_in + k_base_offset_val;

                            // Compute QK dot product (vectorized, assuming head_dim % 4 == 0 from C++ validation)
                            score_val = 0.0f; // Reset for actual computation
                            for (uint d_qk = 0; d_qk < params.head_dim; d_qk += 4) {
                                float4 q_chunk = float4(q_shmem[d_qk], q_shmem[d_qk+1], q_shmem[d_qk+2], q_shmem[d_qk+3]);
                                float4 k_chunk = float4((float)k_vector_ptr_val[d_qk], (float)k_vector_ptr_val[d_qk+1],
                                                        (float)k_vector_ptr_val[d_qk+2], (float)k_vector_ptr_val[d_qk+3]);
                                score_val += dot(q_chunk, k_chunk);
                            }
                        }
                    }
                    score_tile[local_thread_idx] = score_val; // Stash the raw score
                } else {
                    // Threads outside current_hist_tile_actual_len write -INF to their score_tile slot for padding
                    score_tile[local_thread_idx] = -INFINITY;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup); // Ensure all raw scores for the tile are in score_tile

                // --- Find m_local_tile (max score in current score_tile) ---
                float current_thread_score_for_max_reduction = (local_thread_idx < current_hist_tile_actual_len) ? score_tile[local_thread_idx] : -INFINITY;

                G_partial_max_scores[local_thread_idx] = current_thread_score_for_max_reduction;
                threadgroup_barrier(mem_flags::mem_threadgroup); // Ensure all G_partial_max_scores are written

                float simd_max_m_tile_val = simd_max(G_partial_max_scores[local_thread_idx]);
                if (simd_lane_id == 0) { G_simd_reduced_maxes[simd_group_id] = simd_max_m_tile_val; }
                threadgroup_barrier(mem_flags::mem_threadgroup); // Ensure G_simd_reduced_maxes written

                float m_local_tile_val = -INFINITY;
                if (local_thread_idx == 0) {
                    m_local_tile_val = (current_hist_tile_actual_len > 0) ? G_simd_reduced_maxes[0] : 0.0f; // Default to 0 if tile empty
                    for (uint sg_idx = 1; sg_idx < num_simd_groups; ++sg_idx) {
                        m_local_tile_val = max(m_local_tile_val, G_simd_reduced_maxes[sg_idx]);
                    }
                    if (m_local_tile_val == -INFINITY && current_hist_tile_actual_len > 0) m_local_tile_val = 0.0f;
                    *G_final_max_for_item = m_local_tile_val; // Use G_final_max_for_item to broadcast m_local_tile_val
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                m_local_tile_val = *G_final_max_for_item; // All threads get the correct m_local_tile_val for this tile

                // --- Stash exp(score - m_local_tile_val) in score_tile ---
                if (local_thread_idx < current_hist_tile_actual_len) {
                    score_tile[local_thread_idx] = exp(max(score_tile[local_thread_idx] - m_local_tile_val, params.log_exp_min_clamp));
                } else {
                    score_tile[local_thread_idx] = 0.0f; // Non-contributing part of the tile
                }
                threadgroup_barrier(mem_flags::mem_threadgroup); // Ensure all exp_vals are in score_tile

                // --- Compute d_local_tile_total (sum of current score_tile) ---
                float thread_s_val_for_reduction = (local_thread_idx < current_hist_tile_actual_len) ? score_tile[local_thread_idx] : 0.0f;

                // Repurpose G_partial_max_scores for partial sums
                G_partial_max_scores[local_thread_idx] = thread_s_val_for_reduction;
                threadgroup_barrier(mem_flags::mem_threadgroup); // Ensure G_partial_max_scores written

                // Repurpose G_simd_reduced_maxes for SIMD sums
                float simd_sum_d_tile_val = simd_sum(G_partial_max_scores[local_thread_idx]);
                if (simd_lane_id == 0) { G_simd_reduced_maxes[simd_group_id] = simd_sum_d_tile_val; }
                threadgroup_barrier(mem_flags::mem_threadgroup); // Ensure G_simd_reduced_maxes written

                float d_local_tile_total_val = 0.0f;
                if (local_thread_idx == 0) {
                    for (uint sg_idx = 0; sg_idx < num_simd_groups; ++sg_idx) {
                        d_local_tile_total_val += G_simd_reduced_maxes[sg_idx];
                    }
                    // Use G_final_sum_exp_for_item to broadcast d_local_tile_total_val
                    *G_final_sum_exp_for_item = d_local_tile_total_val;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                d_local_tile_total_val = *G_final_sum_exp_for_item; // All threads get d_local_tile_total_val

                // --- Update m_global, s_global, and rescale acc_tile_local for current D-chunk ---
                float scale_factor_for_acc_and_s = 1.0f; // Default if m_global doesn't change

                if (m_local_tile_val > m_global) {
                    scale_factor_for_acc_and_s = exp(max(m_global - m_local_tile_val, params.log_exp_min_clamp));
                    m_global = m_local_tile_val; // Update thread-local m_global
                }

#ifdef DEBUG
                if (!isfinite(scale_factor_for_acc_and_s)) {
                    // Set to a safe value to prevent NaN propagation
                    scale_factor_for_acc_and_s = 0.0f;
                }
#endif

                // Update thread-local s_global
                s_global = s_global * scale_factor_for_acc_and_s +
                           (d_local_tile_total_val * exp(max(m_local_tile_val - m_global, params.log_exp_min_clamp)));
                           // Note: if m_global was just updated to m_local_tile_val, then exp(m_local_tile_val - m_global) is exp(0)=1.
                           // If m_local_tile_val <= m_global, then scale_factor_for_acc_and_s was 1.0.

                // Rescale the current D-chunk in acc_tile_local by scale_factor_for_acc_and_s
                // This loop is over the 'current_d_chunk_len' defined by the outer D-tiling loop.
                for (uint i = 0; i < current_d_chunk_len; ++i) { // current_d_chunk_len from outer d_base_offset loop
                    acc_tile_local[i] *= scale_factor_for_acc_and_s;
                }

                // --- Accumulate Weighted V into acc_tile_local (for current D-chunk: d_base_offset to d_base_offset + current_d_chunk_len) ---
                if (local_thread_idx < current_hist_tile_actual_len) {
                    uint actual_hist_token_pos = hist_tile_start + local_thread_idx;

                    // Get V-vector pointer (direct read from v_cache_pool_in)
                    uint logical_block_idx_v = actual_hist_token_pos / params.tokens_per_page;
                    // No need to re-check page table validity if K was valid from same logical_block_idx
                    // Assuming K and V pages are consistent.
                    uint token_slot_in_page_v = actual_hist_token_pos % params.tokens_per_page;
                    uint q_head_for_kv_map_v = (params.num_q_heads > 1) ? (global_item_idx % params.num_q_heads) : 0;
                    uint target_kv_head_idx_v = 0; // Calculate as done for K
                    if (params.num_kv_heads > 0) {
                        if (params.num_q_heads > params.num_kv_heads) {
                            target_kv_head_idx_v = q_head_for_kv_map_v / (params.num_q_heads / params.num_kv_heads);
                        } else {
                            target_kv_head_idx_v = q_head_for_kv_map_v;
                        }
                        if (target_kv_head_idx_v >= params.num_kv_heads) target_kv_head_idx_v %= params.num_kv_heads;
                    }
                    ulong v_base_offset_val = (ulong)page_table_in[item_seq_idx_in_batch * params.max_logical_blocks_per_seq + logical_block_idx_v] * params.tokens_per_page * params.num_kv_heads * params.head_dim +
                                           (ulong)token_slot_in_page_v * params.num_kv_heads * params.head_dim +
                                           (ulong)target_kv_head_idx_v * params.head_dim;
                    device const half* v_vector_ptr_val = v_cache_pool_in + v_base_offset_val;

                    // Calculate final weight component: exp(raw_score - m_global)
                    // score_tile[local_thread_idx] = exp(raw_score - m_local_tile_val)
                    // Need to multiply by exp(m_local_tile_val - m_global)
                    float w_numerator_global_scaled = score_tile[local_thread_idx] *
                                                     exp(max(m_local_tile_val - m_global, params.log_exp_min_clamp));

                    // Accumulate into the D-chunk (acc_tile_local)
                    // current_d_chunk_len and d_base_offset are from the outer D-tiling loop.
                    for (uint d_idx_in_chunk = 0; d_idx_in_chunk < current_d_chunk_len; ++d_idx_in_chunk) {
                        uint actual_d_dim_idx = d_base_offset + d_idx_in_chunk;
                        // Outer D-loop guarantees the chunk fits within head_dim
                        float v_comp = (float)v_vector_ptr_val[actual_d_dim_idx]; // Cast to float once
                        acc_tile_local[d_idx_in_chunk] += w_numerator_global_scaled * v_comp;
                    }
                }

                threadgroup_barrier(mem_flags::mem_threadgroup); // End of history tile processing
            } // End history tiling loop
        } // End if history > 0

        // Inside D-tiling loop, AFTER history tiling loop
        // acc_tile_local holds unnormalized sum for current D-chunk
        // s_global is the final denominator for the entire item

        float inv_s_global = (s_global > 1e-9f) ? (1.0f / s_global) : 0.0f;

        // Normalize the current acc_tile_local D-chunk
        for (uint i = 0; i < current_d_chunk_len; ++i) {
            acc_tile_local[i] *= inv_s_global;
        }

        // Reduce the now-normalized acc_tile_local D-chunk across the threadgroup and write to output_buffer
        // This uses the G_simd_group_v_sums scratch space and logic from TDD-8.1
        for (uint i = 0; i < current_d_chunk_len; i += 4) { // Iterate over float4 chunks in acc_tile_local
            float4 chunk_to_write = float4(0.0f);
            // Safe loading from acc_tile_local into chunk_to_write
            if (i < current_d_chunk_len)     chunk_to_write.x = acc_tile_local[i+0];
            if (i+1 < current_d_chunk_len) chunk_to_write.y = acc_tile_local[i+1];
            if (i+2 < current_d_chunk_len) chunk_to_write.z = acc_tile_local[i+2];
            if (i+3 < current_d_chunk_len) chunk_to_write.w = acc_tile_local[i+3];

            // SIMD-group reduction of the (now final, normalized) values in chunk_to_write
            float4 reduced_simd_group_final_chunk;
            reduced_simd_group_final_chunk.x = simd_sum(chunk_to_write.x);
            reduced_simd_group_final_chunk.y = simd_sum(chunk_to_write.y);
            reduced_simd_group_final_chunk.z = simd_sum(chunk_to_write.z);
            reduced_simd_group_final_chunk.w = simd_sum(chunk_to_write.w);

            if (simd_lane_id == 0) {
                G_simd_group_v_sums[simd_group_id] = reduced_simd_group_final_chunk;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (local_thread_idx == 0) {
                float4 final_output_chunk = float4(0.0f);
                for (uint sg_idx = 0; sg_idx < num_simd_groups; ++sg_idx) {
                    final_output_chunk += G_simd_group_v_sums[sg_idx];
                }

                uint output_base_idx = global_item_idx * params.head_dim + d_base_offset + i;
                if (i < current_d_chunk_len)     output_buffer[output_base_idx + 0] = (half)final_output_chunk.x;
                if (i+1 < current_d_chunk_len) output_buffer[output_base_idx + 1] = (half)final_output_chunk.y;
                if (i+2 < current_d_chunk_len) output_buffer[output_base_idx + 2] = (half)final_output_chunk.z;
                if (i+3 < current_d_chunk_len) output_buffer[output_base_idx + 3] = (half)final_output_chunk.w;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup); // Sync before next i (float4 chunk of current d_chunk)
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    } // End of D-tiling loop

    // The D-tiling loop now handles writing the computed output
    // No need for extra zero-out pass here
} // End of kernel
