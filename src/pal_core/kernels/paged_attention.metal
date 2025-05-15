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
 *  - Kahan summation for improved numerical stability with long sequences
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
 *  1. Iterate over head_dim in chunks of params.d_chunk_size_runtime.
 *  2. For each chunk:
 *     - Each thread initializes a local acc_tile_local[kDefaultAccTileChunkSize].
 *     - Re-scan history:
 *       - Compute Q·K score.
 *       - Calculate attention weight p = fast::exp(score - m_final_global) / d_final_global.
 *       - Accumulate V contributions into acc_tile_local: acc_tile_local += p * V_chunk.
 *     - Reduce acc_tile_local across the threadgroup and write to output buffer.
 */
[[kernel]] void paged_attn_kernel(
    device      const half* queries_in              [[buffer(0)]],
    device      const half* k_cache_pool_in         [[buffer(1)]],
    device      const half* v_cache_pool_in         [[buffer(2)]],
    device      const uint* page_table_in           [[buffer(3)]],
    device      const int*  sequence_lengths_in     [[buffer(4)]],
    device      const int*  query_to_seq_map_in     [[buffer(5)]],
    device      const int*  query_token_offset_in   [[buffer(6)]],
    constant    const PagedAttentionParams& params  [[buffer(7)]],
    device      half* output_buffer                 [[buffer(8)]],
    threadgroup float* tg_mem                       [[threadgroup(0)]],
    uint3       tg_pos_in_grid                      [[threadgroup_position_in_grid]],
    uint3       tg_dim                              [[threads_per_threadgroup]],
    uint        local_idx_in_tg                     [[thread_index_in_threadgroup]],
    uint        simd_lane_id                        [[thread_index_in_simdgroup]],
    uint        simd_group_id                       [[simdgroup_index_in_threadgroup]]
) {
    // --- 1/10: Calculate Inverse Sqrt Head Dim ---
    float inv_sqrt_head_dim_val = calculate_inv_sqrt_head_dim(params.head_dim);

    // --- 2/10: Basic Input Validation ---
    // Early exit for degenerate case where head_dim is zero
    if (params.head_dim == 0) {
        return;
    }

    // --- 3/10: Thread-Local Accumulators for Online Softmax ---
    float m_local = -INFINITY; // Maximum score accumulator
    float d_local = 0.0f;      // Sum of scaled exponentials accumulator

    // --- 4/10: Thread Identifiers & SIMD Group Info ---
    uint global_item_idx = tg_pos_in_grid.x;    // Identifies the query-head item
    uint local_thread_idx = local_idx_in_tg;    // Thread ID within this group
    const uint num_simd_groups = max(1u, (tg_dim.x + kSimdLanesPerGroup - 1) / kSimdLanesPerGroup); // Calculate number of actual SIMD groups

    // --- 4.1/10: KV Head Mapping (constant per item) ---
    const uint q_head_for_kv_map     = (params.num_q_heads > 1)
                                     ? (global_item_idx % params.num_q_heads)
                                     : 0;
    const uint target_kv_head_idx_item =
        map_q_to_kv_head(q_head_for_kv_map,
                         params.num_q_heads,
                         params.num_kv_heads);

    // --- 5/10: Threadgroup Memory Carving ---
    // Layout for Pass 1 and Pass 2 reductions.
    // q_shmem, G_partial_max_scores, G_simd_reduced_maxes, G_simd_reduced_adjusted_sum_exps,
    // G_final_max_for_item, G_final_sum_exp_for_item are primarily for Pass 1.
    // G_simd_group_v_sums is used in Pass 2 for o_tile reduction.

    threadgroup float* q_shmem = tg_mem;  // head_dim floats

    // Align subsequent threadgroup memory sections to kAlignmentBytes (e.g., 64 bytes).
    // This helps align with cache line sizes on Apple GPUs and can prevent
    // performance issues like false sharing or split transactions for buffers
    // accessed by multiple threads or SIMD groups.
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

    // Score tile for history processing - directly aligned after the previous section
    current_offset = (current_offset + kAlignmentMask) & ~kAlignmentMask; // Align start of score_tile region
    threadgroup float* score_tile_base_all = (threadgroup float*)current_offset;
                                  // Size: params.tile_size_T_runtime floats + padding

    // K_tile and V_tile pointers are defined but not used for tg memory caching
    // These are kept as placeholders for future optimization where we might
    // implement cooperative loading of K/V into threadgroup memory
    threadgroup float* K_tile = nullptr; // Not actually allocated in tg memory
    threadgroup float* V_tile = nullptr; // Not actually allocated in tg memory

    // --- 6/10: Q-Vector Pointer Calculation & Staging ---
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

    // --- 6.1/10: Stage Q-Vector into Shared Memory ---
    for (uint i = local_thread_idx; i < params.head_dim; i += tg_dim.x) {
        q_shmem[i] = (float)q_vector_item_ptr[i] * inv_sqrt_head_dim_val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // --- 7/10: History & Sequence Length Setup ---
    uint token_idx_for_sideband_lookup;
    if (params.num_q_heads > 1) {
        token_idx_for_sideband_lookup = global_item_idx / params.num_q_heads;
    } else {
        token_idx_for_sideband_lookup = global_item_idx;
    }

    uint item_seq_idx_in_batch = (uint)query_to_seq_map_in[token_idx_for_sideband_lookup];
    if (item_seq_idx_in_batch >= params.num_sequences_in_batch) {
        // Zero the output for this item and exit
        if (local_thread_idx == 0) {
            zero_output_vector_for_item(global_item_idx, output_buffer, params);
        }
        return;
    }

    int item_signed_query_token_offset = query_token_offset_in[token_idx_for_sideband_lookup];
    if (item_signed_query_token_offset < 0) {
        // Zero the output for this item and exit
        if (local_thread_idx == 0) {
            zero_output_vector_for_item(global_item_idx, output_buffer, params);
        }
        return;
    }

    uint item_current_q_token_logical_pos = (uint)item_signed_query_token_offset;
    uint item_actual_sequence_length = (uint)sequence_lengths_in[item_seq_idx_in_batch];
    uint item_effective_history_length = min(item_current_q_token_logical_pos,
                                         item_actual_sequence_length);

    // --- 8/10: Initialize Global Softmax Stats ---
    if (local_thread_idx == 0) {
        (*tg_global_stats).x = -INFINITY; // m_global
        (*tg_global_stats).y = 0.0f; // s_global
        (*tg_s_global_comp) = 0.0f; // Kahan summation compensation term
    }
    threadgroup_barrier(mem_flags::mem_threadgroup); // Ensure initialized before use

    // --- 9/10: Setup Output Accumulator ---
    // Use a fixed-size buffer, but the actual processing will be based on params.d_chunk_size_runtime
    float acc_tile_local[kDefaultAccTileChunkSize]; // Thread-local stack array for a chunk of head_dim

    // --- 10/10: Main Attention Computation (D-Tiling) ---
    for (uint d_base_offset = 0; d_base_offset < params.head_dim; d_base_offset += params.d_chunk_size_runtime) {
        uint current_d_chunk_len = min(min(params.d_chunk_size_runtime, kDefaultAccTileChunkSize),
                                       params.head_dim - d_base_offset);

        // Initialize local accumulator tile for this d_chunk
        for (uint i = 0; i < current_d_chunk_len; ++i) {
            acc_tile_local[i] = 0.0f;
        }

        // --- 10.1/10: History Tiling Loop ---
        if (item_effective_history_length > 0) {
            for (uint hist_tile_start = 0; hist_tile_start < item_effective_history_length; hist_tile_start += params.tile_size_T_runtime) {
                uint current_hist_tile_actual_len = min(params.tile_size_T_runtime, item_effective_history_length - hist_tile_start);

                /* Per-SIMD-group 32-byte (kSimdPaddingFloats-float) lane-shift for score_tile */
                threadgroup float* score_tile = score_tile_base_all + simd_group_id * kSimdPaddingFloats;

                // --- 10.1.1/10: History Tile - Score Calculation & Stashing ---
                if (local_thread_idx < current_hist_tile_actual_len) {
                    uint actual_hist_token_pos = hist_tile_start + local_thread_idx;

                    // Default to a state that would lead to zero contribution if checks fail
                    float score_val = -INFINITY;

                    // KV-head index is constant for this item (computed once in section 4.1)
                    uint target_kv_head_idx_k = target_kv_head_idx_item;


                    // Fetch K-vector pointer using helper
                    device const half* k_vector_ptr_val = fetch_kv_pointer(
                        true, // is_k_vector
                        actual_hist_token_pos,
                        target_kv_head_idx_k,
                        k_cache_pool_in,
                        v_cache_pool_in,
                        page_table_in,
                        item_seq_idx_in_batch,
                        params
                    );

                    if (k_vector_ptr_val != nullptr) {
                        // Compute QK dot product using helper function
                        score_val = dot_product_qk(q_shmem, k_vector_ptr_val, params);
                    }
                    score_tile[local_thread_idx] = score_val; // Stash the raw score
                } else {
                    // Threads outside current_hist_tile_actual_len write -INF to their score_tile slot for padding
                    score_tile[local_thread_idx] = -INFINITY;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup); // Ensure all raw scores for the tile are in score_tile

                // --- 10.1.2/10: History Tile - Local Max (m_local_tile) Reduction ---
                float current_thread_score_for_max_reduction = (local_thread_idx < current_hist_tile_actual_len) ? score_tile[local_thread_idx] : -INFINITY;

                tg_partial_reduce_scratch[local_thread_idx] = current_thread_score_for_max_reduction;
                threadgroup_barrier(mem_flags::mem_threadgroup); // Ensure all G_partial_max_scores are written

                float simd_max_m_tile_val = simd_max(tg_partial_reduce_scratch[local_thread_idx]);
                if (simd_lane_id == 0) { tg_simd_reduce_scratch[simd_group_id] = simd_max_m_tile_val; }
                threadgroup_barrier(mem_flags::mem_threadgroup); // Ensure G_simd_reduced_maxes written

                float m_local_tile_val = -INFINITY;
                if (local_thread_idx == 0) {
                    m_local_tile_val = (current_hist_tile_actual_len > 0) ? tg_simd_reduce_scratch[0] : 0.0f; // Default to 0 if tile empty
                    for (uint sg_idx = 1; sg_idx < num_simd_groups; ++sg_idx) {
                        m_local_tile_val = max(m_local_tile_val, tg_simd_reduce_scratch[sg_idx]);
                    }
                    if (m_local_tile_val == -INFINITY && current_hist_tile_actual_len > 0) m_local_tile_val = 0.0f;
                    tg_simd_reduce_scratch[0] = m_local_tile_val; // Use tg_simd_reduce_scratch[0] to broadcast m_local_tile_val
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                m_local_tile_val = tg_simd_reduce_scratch[0]; // All threads get the correct m_local_tile_val for this tile

                // --- 10.1.3/10: History Tile - Update Score Tile with Exponentiated Values ---
                if (local_thread_idx < current_hist_tile_actual_len) {
                    score_tile[local_thread_idx] = exp(max(score_tile[local_thread_idx] - m_local_tile_val, params.log_exp_min_clamp));
                } else {
                    score_tile[local_thread_idx] = 0.0f; // Non-contributing part of the tile
                }
                threadgroup_barrier(mem_flags::mem_threadgroup); // Ensure all exp_vals are in score_tile

                // --- 10.1.4/10: History Tile - Local Sum (d_local_tile) Reduction ---
                float thread_s_val_for_reduction = (local_thread_idx < current_hist_tile_actual_len) ? score_tile[local_thread_idx] : 0.0f;

                // Perform SIMD reduction directly on register values (no barrier needed)
                float simd_sum_d_tile_val = simd_sum(thread_s_val_for_reduction);

                // Write results to G_simd_reduced_maxes for final reduction by thread0
                if (simd_lane_id == 0) { tg_simd_reduce_scratch[simd_group_id] = simd_sum_d_tile_val; }
                threadgroup_barrier(mem_flags::mem_threadgroup); // Ensure G_simd_reduced_maxes written

                float d_local_tile_total_val = 0.0f;
                if (local_thread_idx == 0) {
                    for (uint sg_idx = 0; sg_idx < num_simd_groups; ++sg_idx) {
                        d_local_tile_total_val += tg_simd_reduce_scratch[sg_idx];
                    }
                    // Use tg_simd_exp_sums_scratch[0] to broadcast d_local_tile_total_val
                    tg_simd_exp_sums_scratch[0] = d_local_tile_total_val;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                d_local_tile_total_val = tg_simd_exp_sums_scratch[0]; // All threads get d_local_tile_total_val

                // --- 10.1.5/10: History Tile - Update Global Stats & Rescale Accumulator ---

                // Thread 0 will handle the update of m_global and s_global in a single atomic operation
                if (local_thread_idx == 0) {
                    update_softmax_stats_kahan(
                        tg_global_stats,
                        tg_s_global_comp,
                        m_local_tile_val,
                        d_local_tile_total_val,
                        tg_simd_reduce_scratch,
                        params
                    );
                }
                threadgroup_barrier(mem_flags::mem_threadgroup); // Sync shared g_stats and scale_factor

                // All threads read the consistent m_global and scale factor for this iteration
                float m_global_current_iter_atomic = (*tg_global_stats).x;
                float scale_for_acc_iter_atomic = tg_simd_reduce_scratch[0];

                // All threads rescale their local acc_tile_local
                for (uint i = 0; i < current_d_chunk_len; ++i) {
                    acc_tile_local[i] *= scale_for_acc_iter_atomic;
                }


                // --- 10.1.6/10: History Tile - Weighted V Accumulation ---
                if (local_thread_idx < current_hist_tile_actual_len) {
                    uint actual_hist_token_pos = hist_tile_start + local_thread_idx;

                    uint target_kv_head_idx_v = target_kv_head_idx_item;

                    // Use the helper to fetch V-vector pointer
                    device const half* v_vector_ptr_val = fetch_kv_pointer(
                        false, // Not K-vector, but V-vector
                        actual_hist_token_pos,
                        target_kv_head_idx_v,
                        k_cache_pool_in,
                        v_cache_pool_in,
                        page_table_in,
                        item_seq_idx_in_batch,
                        params
                    );

                    // Only continue if V pointer is valid
                    if (v_vector_ptr_val != nullptr) {
                        // Calculate final weight component: exp(raw_score - m_global)
                        // score_tile[local_thread_idx] = exp(raw_score - m_local_tile_val)
                        // Need to multiply by exp(m_local_tile_val - m_global_current_iter_atomic)
                        // Using m_global_current_iter_atomic that was read after the sync barrier
                        float weight_term = score_tile[local_thread_idx]; // Already float
                        float exp_term = exp(max(m_local_tile_val - m_global_current_iter_atomic, params.log_exp_min_clamp));
                        float final_p_attn_weight_numerator = weight_term * exp_term; // float * float = float

                        // Accumulate into the D-chunk (acc_tile_local)
                        // current_d_chunk_len and d_base_offset are from the outer D-tiling loop.
                        for (uint d_idx_in_chunk = 0; d_idx_in_chunk < current_d_chunk_len; ++d_idx_in_chunk) {
                            uint actual_d_dim_idx = d_base_offset + d_idx_in_chunk;
                            // Outer D-loop guarantees the chunk fits within head_dim
                            // Metal implicitly widens half to float in mixed expressions
                            acc_tile_local[d_idx_in_chunk] += final_p_attn_weight_numerator * v_vector_ptr_val[actual_d_dim_idx];
                        }
                    }
                }

                threadgroup_barrier(mem_flags::mem_threadgroup); // End of history tile processing
            } // End history tiling loop
        } // End if history > 0

        // --- 10.2/10: D-Chunk - Final Normalization & Output Write ---
        // acc_tile_local holds unnormalized sum for current D-chunk
        // s_global is the final denominator for the entire item

        // Use the threadgroup shared final s_global value from g_global_stats_ptr
        float s_global_final = (*tg_global_stats).y;
        float inv_s_global = (s_global_final > kEpsilonForZeroGuard) ? (1.0f / s_global_final) : 0.0f;

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
                tg_simd_v_chunk_sums[simd_group_id] = reduced_simd_group_final_chunk;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (local_thread_idx == 0) {
                float4 final_output_chunk = float4(0.0f);
                for (uint sg_idx = 0; sg_idx < num_simd_groups; ++sg_idx) {
                    final_output_chunk += tg_simd_v_chunk_sums[sg_idx];
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

} // End of kernel
