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
 *  - K/V-vector caching in threadgroup memory
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
 *  Algorithm Stages (Fused Single-Pass for head_dim <= kMaxHeadDimMetal):
 *  1. Collaboratively load and pre-scale query vector into shared memory.
 *  2. Initialize full head_dim accumulator (acc_tile_local[kMaxHeadDimMetal]).
 *  3. Process history tiles in a single pass:
 *     - Load K/V-vectors into threadgroup memory (K_tile, V_tile).
 *     - Compute Q·K scores and store in thread-local variables.
 *     - Perform online softmax with thread-local scores to update global stats.
 *     - Calculate attention weights using thread-local values.
 *     - Directly accumulate weighted V-vectors from V_tile into full acc_tile_local.
 *     - Rescale accumulator based on global softmax updates.
 *  4. After processing all history:
 *     - Normalize the full accumulator using the final softmax denominator.
 *     - Reduce across the threadgroup and write the entire head_dim to output buffer.
 *
 *  The fused approach eliminates the need for score_tile storage and the D-chunking outer loop,
 *  processing the entire head_dim in one pass for better efficiency on typical model dimensions.
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

    // Hoisted: Calculate padded head dimension to avoid bank conflicts
    // This is used for K_tile and V_tile row strides.
    const uint padded_head_dim_hoisted = params.head_dim + params.pad_floats_per_row;

    // Guard path for large head_dim that's not supported by the fused path
    if (params.head_dim > kMaxHeadDimMetal) {
        // TODO: Implement or call the non-fused (original D-chunking) kernel logic here.
        // For now, to prevent execution with unsupported head_dim for the fused path:
        if (local_idx_in_tg == 0) { // Only one thread needs to zero
            zero_output_vector_for_item(tg_pos_in_grid.x, output_buffer, params);
        }
        return; // Exit if head_dim is too large for the fused path we are building
    }

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
    // Layout for reductions, statistics and vector caching.
    // q_shmem for query vectors
    // tg_partial_reduce_scratch, tg_simd_reduce_scratch, tg_simd_exp_sums_scratch for softmax reductions
    // tg_global_stats, tg_s_global_comp for Kahan summation
    // tg_simd_v_chunk_sums for final output reduction

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

    // K_tile for caching K-vectors in threadgroup memory - after SIMD v_chunk_sums
    current_offset = (uintptr_t)(tg_simd_v_chunk_sums + num_simd_groups);
    current_offset = (current_offset + kAlignmentMask) & ~kAlignmentMask;

    threadgroup float* K_tile = (threadgroup float*)current_offset;

    // Update current_offset for the next section (V_tile) using padded_head_dim_hoisted for K_tile's size
    current_offset += params.tile_size_T_runtime * padded_head_dim_hoisted * sizeof(float);
    current_offset = (current_offset + kAlignmentMask) & ~kAlignmentMask;

    // V_tile for caching V-vectors in threadgroup memory
    threadgroup float* V_tile = (threadgroup float*)current_offset;

    // Update current_offset for final guard bytes - using padded_head_dim_hoisted for V_tile's size
    current_offset += params.tile_size_T_runtime * padded_head_dim_hoisted * sizeof(float);
    current_offset = (current_offset + kAlignmentMask) & ~kAlignmentMask;

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
    // Use a full-size buffer for the entire head dimension in the fused path
    float acc_tile_local[kMaxHeadDimMetal]; // Thread-local stack array for full head_dim

    // Initialize local accumulator for the full head dimension
    for (uint i = 0; i < params.head_dim; ++i) {
        acc_tile_local[i] = 0.0f;
    }

    // --- 10/10: Main Attention Computation (Fused Pass) ---
    // No D-tiling loop in the fused path

        // --- 10.1/10: History Tiling Loop ---
        if (item_effective_history_length > 0) {
            for (uint hist_tile_start = 0; hist_tile_start < item_effective_history_length; hist_tile_start += params.tile_size_T_runtime) {
                uint current_hist_tile_actual_len = min(params.tile_size_T_runtime, item_effective_history_length - hist_tile_start);

                // --- Load K-vectors into K_tile ---
                if (local_thread_idx < current_hist_tile_actual_len) {
                    uint actual_hist_token_pos_for_k_load = hist_tile_start + local_thread_idx;
                    uint target_kv_head_idx_for_k_load = target_kv_head_idx_item;

                    // Fetch K-vector pointer using helper
                    device const half* k_vector_global_ptr = fetch_kv_pointer(
                        true, // is_k_vector
                        actual_hist_token_pos_for_k_load,
                        target_kv_head_idx_for_k_load,
                        k_cache_pool_in,
                        v_cache_pool_in,
                        page_table_in,
                        item_seq_idx_in_batch,
                        params
                    );

                    // Use padded head dimension for tile row stride to avoid bank conflicts
                    // uint padded_head_dim = params.head_dim + kPaddingFloatsPerRow; // Replaced by hoisted version

                    if (k_vector_global_ptr != nullptr) {
                        // Each thread loads its K-vector into the K_tile
                        // K_tile is now [tile_size_T_runtime][padded_head_dim] to avoid bank conflicts
                        threadgroup float* k_tile_entry_ptr = K_tile + (local_thread_idx * padded_head_dim_hoisted);

                        // Load the entire K-vector using float4 chunks
                        // Note: We only fill the actual head_dim elements, leaving padding untouched
                        device const half4* __attribute__((aligned(8))) k_vec_global_ptr_h4 = reinterpret_cast<device const half4*>(k_vector_global_ptr);
                        for (uint d_chunk = 0; d_chunk < params.head_dim / 4; ++d_chunk) {
                            half4 k_val_h4 = k_vec_global_ptr_h4[d_chunk];
                            float4 k_val_f4 = float4(k_val_h4);
                            k_tile_entry_ptr[d_chunk * 4 + 0] = k_val_f4.x;
                            k_tile_entry_ptr[d_chunk * 4 + 1] = k_val_f4.y;
                            k_tile_entry_ptr[d_chunk * 4 + 2] = k_val_f4.z;
                            k_tile_entry_ptr[d_chunk * 4 + 3] = k_val_f4.w;
                        }
                    } else {
                        // Handle null K-vector pointer: fill corresponding K_tile entry with zeros
                        threadgroup float* k_tile_entry_ptr = K_tile + (local_thread_idx * padded_head_dim_hoisted);
                        for (uint d_chunk = 0; d_chunk < params.head_dim / 4; ++d_chunk) {
                            k_tile_entry_ptr[d_chunk * 4 + 0] = 0.0f;
                            k_tile_entry_ptr[d_chunk * 4 + 1] = 0.0f;
                            k_tile_entry_ptr[d_chunk * 4 + 2] = 0.0f;
                            k_tile_entry_ptr[d_chunk * 4 + 3] = 0.0f;
                        }
                    }
                } else {
                    // Thread is outside the current actual history tile length, zero out its K_tile row
                    threadgroup float* k_tile_entry_ptr = K_tile + (local_thread_idx * padded_head_dim_hoisted);
                    for (uint d_chunk = 0; d_chunk < params.head_dim / 4; ++d_chunk) {
                        k_tile_entry_ptr[d_chunk * 4 + 0] = 0.0f;
                        k_tile_entry_ptr[d_chunk * 4 + 1] = 0.0f;
                        k_tile_entry_ptr[d_chunk * 4 + 2] = 0.0f;
                        k_tile_entry_ptr[d_chunk * 4 + 3] = 0.0f;
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup); // Ensure K_tile is fully populated before use

                // --- Load V-vectors into V_tile ---
                if (local_thread_idx < current_hist_tile_actual_len) {
                    uint actual_hist_token_pos_for_v_load = hist_tile_start + local_thread_idx;
                    uint target_kv_head_idx_for_v_load = target_kv_head_idx_item;

                    // Fetch V-vector pointer using helper
                    device const half* v_vector_global_ptr = fetch_kv_pointer(
                        false, // is_k_vector = false for V
                        actual_hist_token_pos_for_v_load,
                        target_kv_head_idx_for_v_load,
                        k_cache_pool_in,
                        v_cache_pool_in,
                        page_table_in,
                        item_seq_idx_in_batch,
                        params
                    );

                    // Use padded head dimension for tile row stride to avoid bank conflicts
                    // uint padded_head_dim = params.head_dim + kPaddingFloatsPerRow; // Replaced by hoisted version

                    // Each thread loads its V-vector into the V_tile
                    threadgroup float* v_tile_entry_ptr = V_tile + (local_thread_idx * padded_head_dim_hoisted);

                    if (v_vector_global_ptr != nullptr) {
                        // Load the entire V-vector using float4 chunks
                        // Note: We only fill the actual head_dim elements, leaving padding untouched
                        device const half4* __attribute__((aligned(8))) v_vec_global_ptr_h4 = reinterpret_cast<device const half4*>(v_vector_global_ptr);
                        for (uint d_chunk = 0; d_chunk < params.head_dim / 4; ++d_chunk) {
                            half4 v_val_h4 = v_vec_global_ptr_h4[d_chunk];
                            float4 v_val_f4 = float4(v_val_h4);
                            v_tile_entry_ptr[d_chunk * 4 + 0] = v_val_f4.x;
                            v_tile_entry_ptr[d_chunk * 4 + 1] = v_val_f4.y;
                            v_tile_entry_ptr[d_chunk * 4 + 2] = v_val_f4.z;
                            v_tile_entry_ptr[d_chunk * 4 + 3] = v_val_f4.w;
                        }
                    } else {
                        // Handle null V-vector pointer: fill corresponding V_tile entry with zeros
                        for (uint d_chunk = 0; d_chunk < params.head_dim / 4; ++d_chunk) {
                            v_tile_entry_ptr[d_chunk * 4 + 0] = 0.0f;
                            v_tile_entry_ptr[d_chunk * 4 + 1] = 0.0f;
                            v_tile_entry_ptr[d_chunk * 4 + 2] = 0.0f;
                            v_tile_entry_ptr[d_chunk * 4 + 3] = 0.0f;
                        }
                    }
                } else {
                    // Thread is outside the current actual history tile length, zero out its V_tile row
                    threadgroup float* v_tile_entry_ptr = V_tile + (local_thread_idx * padded_head_dim_hoisted);
                    for (uint d_chunk = 0; d_chunk < params.head_dim / 4; ++d_chunk) {
                        v_tile_entry_ptr[d_chunk * 4 + 0] = 0.0f;
                        v_tile_entry_ptr[d_chunk * 4 + 1] = 0.0f;
                        v_tile_entry_ptr[d_chunk * 4 + 2] = 0.0f;
                        v_tile_entry_ptr[d_chunk * 4 + 3] = 0.0f;
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup); // Ensure V_tile is fully populated before use

                // --- 10.1.1/10: History Tile - Score Calculation (no stashing in fused path) ---
                float thread_score_val = -INFINITY; // Default to a state that would lead to zero contribution

                if (local_thread_idx < current_hist_tile_actual_len) {
                    // Use padded head dimension for tile row stride to avoid bank conflicts
                    // uint padded_head_dim = params.head_dim + kPaddingFloatsPerRow; // Replaced by hoisted version

                    // Now use the K_tile entry for this thread instead of fetching from global memory again
                    // K-vector for this history token is already loaded in the K_tile with padded stride
                    threadgroup const float* k_vector_from_tile = K_tile + (local_thread_idx * padded_head_dim_hoisted);

                    // Compute QK dot product using the modified helper function that works with K_tile
                    thread_score_val = dot_product_qk(q_shmem, k_vector_from_tile, params);
                }

                // --- 10.1.2/10: History Tile - Local Max (m_local_tile) Reduction ---
                float current_thread_score_for_max_reduction = thread_score_val;

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

                // --- 10.1.3/10: History Tile - Compute Exponentiated Values (no Score Tile) ---
                // Calculate the exponentiated score value in a thread-local variable
                float thread_exp_val = 0.0f;
                if (local_thread_idx < current_hist_tile_actual_len) {
                    thread_exp_val = exp(thread_score_val - m_local_tile_val);
                }

                // --- 10.1.4/10: History Tile - Local Sum (d_local_tile) Reduction ---
                float thread_s_val_for_reduction = thread_exp_val;

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
                        tg_simd_reduce_scratch
                    );
                }
                threadgroup_barrier(mem_flags::mem_threadgroup); // Full barrier to ensure tg_global_stats and tg_simd_reduce_scratch are visible to all threads

                // All threads read the consistent m_global and scale factor for this iteration
                float m_global_current_iter_atomic = (*tg_global_stats).x;
                float scale_for_acc_iter_atomic = tg_simd_reduce_scratch[0];

                // All threads rescale their local acc_tile_local (process in float4 chunks for efficiency)
                if (scale_for_acc_iter_atomic != 1.0f) { // Optimization: only multiply if needed
                    for (uint d = 0; d < params.head_dim; d += 4) {
                        float4 acc_chunk = float4(acc_tile_local[d],
                                                (d + 1 < params.head_dim) ? acc_tile_local[d+1] : 0.0f,
                                                (d + 2 < params.head_dim) ? acc_tile_local[d+2] : 0.0f,
                                                (d + 3 < params.head_dim) ? acc_tile_local[d+3] : 0.0f);
                        acc_chunk *= scale_for_acc_iter_atomic;
                        acc_tile_local[d] = acc_chunk.x;
                        if (d + 1 < params.head_dim) acc_tile_local[d+1] = acc_chunk.y;
                        if (d + 2 < params.head_dim) acc_tile_local[d+2] = acc_chunk.z;
                        if (d + 3 < params.head_dim) acc_tile_local[d+3] = acc_chunk.w;
                    }
                }


                // --- 10.1.6/10: History Tile - Weighted V Accumulation (Fused Path) ---
                if (local_thread_idx < current_hist_tile_actual_len) {
                    // Use padded head dimension for tile row stride to avoid bank conflicts
                    // uint padded_head_dim = params.head_dim + kPaddingFloatsPerRow; // Replaced by hoisted version

                    // Get the V-vector from the V_tile instead of fetching from global memory
                    // V-vector for this history token is loaded in the V_tile with padded stride
                    threadgroup const float* v_vector_from_tile = V_tile + (local_thread_idx * padded_head_dim_hoisted);

                    // Calculate final weight component: exp(raw_score - m_global)
                    // thread_exp_val = exp(raw_score - m_local_tile_val)
                    // Need to multiply by exp(m_local_tile_val - m_global_current_iter_atomic)
                    // Using m_global_current_iter_atomic that was read after the sync barrier
                    // This effectively gives us: exp(raw_score - m_global_current_iter_atomic)
                    float weight_term = thread_exp_val; // Already float
                    float exp_term = exp(max(m_local_tile_val - m_global_current_iter_atomic, params.log_exp_min_clamp));
                    float final_p_attn_weight_numerator = weight_term * exp_term; // float * float = float

                    // Accumulate into the full acc_tile_local in float4 chunks for efficiency
                    // In fused path, we process the entire head_dim at once
                    for (uint d_idx = 0; d_idx < params.head_dim; d_idx += 4) {
                        float4 v_chunk;
                        v_chunk.x = v_vector_from_tile[d_idx];
                        v_chunk.y = (d_idx + 1 < params.head_dim) ? v_vector_from_tile[d_idx + 1] : 0.0f;
                        v_chunk.z = (d_idx + 2 < params.head_dim) ? v_vector_from_tile[d_idx + 2] : 0.0f;
                        v_chunk.w = (d_idx + 3 < params.head_dim) ? v_vector_from_tile[d_idx + 3] : 0.0f;

                        // Weighted contribution from this token's V-vector
                        v_chunk *= final_p_attn_weight_numerator;

                        // Accumulate into acc_tile_local
                        acc_tile_local[d_idx] += v_chunk.x;
                        if (d_idx + 1 < params.head_dim) acc_tile_local[d_idx + 1] += v_chunk.y;
                        if (d_idx + 2 < params.head_dim) acc_tile_local[d_idx + 2] += v_chunk.z;
                        if (d_idx + 3 < params.head_dim) acc_tile_local[d_idx + 3] += v_chunk.w;
                    }
                }

                threadgroup_barrier(mem_flags::mem_threadgroup); // End of history tile processing
            } // End history tiling loop
        } // End if history > 0

        // --- 10.2/10: Final Normalization & Output Write (Fused Path) ---
        // acc_tile_local holds unnormalized sum for the full head_dim
        // s_global is the final denominator for the entire item

        // Use the threadgroup shared final s_global value from g_global_stats_ptr
        float s_global_final = (*tg_global_stats).y;
        float inv_s_global = (s_global_final > kEpsilonForZeroGuard) ? (1.0f / s_global_final) : 0.0f;

        // Normalize the full acc_tile_local
        for (uint i = 0; i < params.head_dim; ++i) {
            acc_tile_local[i] *= inv_s_global;
        }

        // Reduce the now-normalized acc_tile_local across the threadgroup and write to output_buffer
        // Process in float4 chunks for efficiency
        for (uint i = 0; i < params.head_dim; i += 4) { // Iterate over float4 chunks in acc_tile_local
            float4 chunk_to_write = float4(0.0f);
            // Safe loading from acc_tile_local into chunk_to_write
            if (i < params.head_dim)     chunk_to_write.x = acc_tile_local[i+0];
            if (i+1 < params.head_dim) chunk_to_write.y = acc_tile_local[i+1];
            if (i+2 < params.head_dim) chunk_to_write.z = acc_tile_local[i+2];
            if (i+3 < params.head_dim) chunk_to_write.w = acc_tile_local[i+3];

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

                uint output_base_idx = global_item_idx * params.head_dim + i;
                if (i < params.head_dim)     output_buffer[output_base_idx + 0] = (half)final_output_chunk.x;
                if (i+1 < params.head_dim) output_buffer[output_base_idx + 1] = (half)final_output_chunk.y;
                if (i+2 < params.head_dim) output_buffer[output_base_idx + 2] = (half)final_output_chunk.z;
                if (i+3 < params.head_dim) output_buffer[output_base_idx + 3] = (half)final_output_chunk.w;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup); // Sync before next chunk
        }

} // End of kernel
