// paged_attention_decoding.metal
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
 * paged_attn_decode_kernel
 * -----------------
 * Fused paged attention for transformer models. Supports MHA, GQA and MQA with
 * vectorized loads, K/V caching and SIMD reductions. One threadgroup handles a
 * single query-head item and performs: load & scale Q, tile K/V history, online
 * softmax, accumulate V and final normalization.
 * Assumes params.head_dim is a multiple of 4, validated on host.
 */
[[kernel]] void paged_attn_decode_kernel(
    device      const half* queries_in              [[buffer(0)]],
    device      const half* k_cache_pool_in         [[buffer(1)]],
    device      const half* v_cache_pool_in         [[buffer(2)]],
    device      const uint* page_table_in           [[buffer(3)]],
    device      const int*  sequence_lengths_in     [[buffer(4)]],
    device      const int*  query_to_seq_map_in     [[buffer(5)]],
    device      const int*  query_token_offset_in   [[buffer(6)]],
    constant    const PagedAttentionParams& params  [[buffer(7)]],
    device      half* output_buffer                 [[buffer(8)]],
    uint actual_simd_width              [[threads_per_simdgroup]],
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

    uint global_item_idx = tg_pos_in_grid.x;    // Identifies the query-head item
    uint local_thread_idx = local_idx_in_tg;    // Thread ID within this group
    const uint num_simd_groups = max(1u, (tg_dim.x + actual_simd_width - 1) / actual_simd_width);

    // --- 4.1/10: KV Head Mapping (constant per item) ---
    const uint q_head_for_kv_map     = (params.num_q_heads > 1)
                                     ? (global_item_idx % params.num_q_heads)
                                     : 0;
    uint target_kv_head_idx = q_head_for_kv_map;
    if (params.num_q_heads > params.num_kv_heads) { // GQA
        target_kv_head_idx = q_head_for_kv_map / (params.num_q_heads / params.num_kv_heads);
    }

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

    threadgroup half* K_tile = (threadgroup half*)current_offset;

    // Update current_offset for the next section (V_tile) using params.head_dim for K_tile's size
    current_offset += params.tile_size_T_runtime * params.head_dim * sizeof(half);
    current_offset = (current_offset + kAlignmentMask) & ~kAlignmentMask;

    // V_tile for caching V-vectors in threadgroup memory
    threadgroup half* V_tile = (threadgroup half*)current_offset;

    // Update current_offset for page-table slice after V_tile
    current_offset += params.tile_size_T_runtime * params.head_dim * sizeof(half);
    current_offset = (current_offset + kAlignmentMask) & ~kAlignmentMask;

    // Threadgroup buffer for current sequence's page-table slice
    threadgroup uint* tg_page_table_slice = (threadgroup uint*)current_offset;
    current_offset += params.max_logical_blocks_per_seq * sizeof(uint);
    current_offset = (current_offset + kAlignmentMask) & ~kAlignmentMask;

    // Final padding guard (mirrors host-side calculation)
    current_offset += 32;
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
    // Use scalar loads to avoid misalignment issues
    threadgroup float4* q_vec_f4 = reinterpret_cast<threadgroup float4*>(q_shmem);

    for (uint chunk = local_thread_idx; chunk < params.head_dim / 4; chunk += tg_dim.x) {
        // Scalar loads from global memory to avoid alignment issues
        uint base_offset_global_q = chunk * 4;
        half qh0 = q_vector_item_ptr[base_offset_global_q + 0];
        half qh1 = q_vector_item_ptr[base_offset_global_q + 1];
        half qh2 = q_vector_item_ptr[base_offset_global_q + 2];
        half qh3 = q_vector_item_ptr[base_offset_global_q + 3];

        float4 q_float_chunk = float4(qh0, qh1, qh2, qh3) * params.inv_sqrt_head_dim;
        q_vec_f4[chunk] = q_float_chunk;
    }
    // All threads read q_shmem in later steps, ensure writes complete across the group

    // --- 7/10: History & Sequence Length Setup ---
    uint token_idx_for_sideband_lookup;
    if (params.num_q_heads > 1) {
        token_idx_for_sideband_lookup = global_item_idx / params.num_q_heads;
    } else {
        token_idx_for_sideband_lookup = global_item_idx;
    }

    uint item_seq_idx_in_batch = (uint)query_to_seq_map_in[token_idx_for_sideband_lookup];
    // Prefetch page-table slice for this sequence into threadgroup memory
    for (uint blk = local_thread_idx; blk < params.max_logical_blocks_per_seq; blk += tg_dim.x) {
        uint flat_idx = item_seq_idx_in_batch * params.max_logical_blocks_per_seq + blk;
        tg_page_table_slice[blk] = page_table_in[flat_idx];
    }

    int item_signed_query_token_offset = query_token_offset_in[token_idx_for_sideband_lookup];

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

    // --- 9/10: Setup Output Accumulator ---
    float acc_tile_local_fp32[kMaxHeadDimMetal]; // FP32 accumulator for numerical stability
    for (uint i_acc = 0; i_acc < params.head_dim; ++i_acc) {
        acc_tile_local_fp32[i_acc] = 0.0f; // Initialize FP32 accumulator
    }

    // --- 10/10: Main Attention Computation (Fused Pass) ---
    for (uint hist_tile_start = 0; hist_tile_start < item_effective_history_length; hist_tile_start += params.tile_size_T_runtime) {
        uint current_hist_tile_actual_len = min(params.tile_size_T_runtime, item_effective_history_length - hist_tile_start);
        // --- Load K-vectors into K_tile ---
        const uint simd_size_const = actual_simd_width; // Use the new argument
        const uint threads_pg_const = tg_dim.x;          // Total threads launched for this group
        const uint num_sg_const = max(1u, (threads_pg_const + simd_size_const - 1) / simd_size_const);
        const uint rows_in_tile_const = current_hist_tile_actual_len;
        const uint chunks_per_row_const = params.head_dim / 4;

        // Each SIMD group cooperatively loads one or more rows assigned to it
        for (uint row_idx_in_tile = simd_group_id; // SIMD group 'simd_group_id' starts with this row
             row_idx_in_tile < rows_in_tile_const;
             row_idx_in_tile += num_sg_const) { // Strides by number of SIMD groups

            // 1A. Get global pointer to the K-vector for this row_idx_in_tile
            device const half* k_vector_global_ptr = fetch_kv_pointer(
                /*is_k_vector=*/true,
                /*absolute_hist_pos=*/hist_tile_start + row_idx_in_tile,
                /*kv_head_idx=*/target_kv_head_idx,
                k_cache_pool_in,
                v_cache_pool_in, // Passed but not used by fetch_kv_pointer for K
                tg_page_table_slice,
                params
            );

            // 1B. Destination row base pointer in K_tile (threadgroup memory)
            threadgroup half* k_tile_row_base_ptr = K_tile + (row_idx_in_tile * params.head_dim);
            threadgroup half4* dst_row_h4_ptr = reinterpret_cast<threadgroup half4*>(k_tile_row_base_ptr);

            // 1C. Cooperative lane-striped copy (or zero-fill) by this SIMD group for this row
            if (k_vector_global_ptr != nullptr) {
                // Lanes within this SIMD group load chunks of the K-vector for 'row_idx_in_tile'
                for (uint chunk_idx_in_row = simd_lane_id; // Lane 'simd_lane_id' starts with this chunk
                     chunk_idx_in_row < chunks_per_row_const;
                     chunk_idx_in_row += simd_size_const) {

                    // Vectorized load - since head_dim is validated to be multiple of 4
                    device const half4* k_vec_h4_ptr = reinterpret_cast<device const half4*>(k_vector_global_ptr);
                    half4 h4_val_from_global = k_vec_h4_ptr[chunk_idx_in_row];

                    dst_row_h4_ptr[chunk_idx_in_row] = h4_val_from_global;  // Store to K_tile
                }
            }
        } // end for row_idx_in_tile

        // Each SIMD group cooperatively loads one or more rows assigned to it
        for (uint row_idx_in_tile = simd_group_id; // SIMD group 'simd_group_id' starts with this row
             row_idx_in_tile < rows_in_tile_const;
             row_idx_in_tile += num_sg_const) { // Strides by number of SIMD groups

            // 1A. Get global pointer to the V-vector for this row_idx_in_tile
            device const half* v_vector_global_ptr = fetch_kv_pointer(
                /*is_k_vector=*/false, // Now fetching V
                /*absolute_hist_pos=*/hist_tile_start + row_idx_in_tile,
                /*kv_head_idx=*/target_kv_head_idx,
                k_cache_pool_in, // Passed but not used by fetch_kv_pointer for V
                v_cache_pool_in,
                tg_page_table_slice,
                params
            );

            // 1B. Destination row base pointer in V_tile (threadgroup memory)
            threadgroup half* v_tile_row_base_ptr = V_tile + (row_idx_in_tile * params.head_dim);
            threadgroup half4* dst_row_h4_ptr = reinterpret_cast<threadgroup half4*>(v_tile_row_base_ptr);

            // 1C. Cooperative lane-striped copy (or zero-fill) by this SIMD group for this row
            if (v_vector_global_ptr != nullptr) {
                // Lanes within this SIMD group load chunks of the V-vector for 'row_idx_in_tile'
                for (uint chunk_idx_in_row = simd_lane_id;
                     chunk_idx_in_row < chunks_per_row_const;
                     chunk_idx_in_row += simd_size_const) {

                    // Vectorized load - since head_dim is validated to be multiple of 4
                    device const half4* v_vec_h4_ptr = reinterpret_cast<device const half4*>(v_vector_global_ptr);
                    half4 h4_val_from_global = v_vec_h4_ptr[chunk_idx_in_row];

                    dst_row_h4_ptr[chunk_idx_in_row] = h4_val_from_global;  // Store to V_tile
                }
            }
        } // end for row_idx_in_tile

        // --- 10.1.1/10: History Tile - Score Calculation (no stashing in fused path) ---
        float thread_score_val = -INFINITY; // Default to a state that would lead to zero contribution

        if (local_thread_idx < current_hist_tile_actual_len) {
            threadgroup const half* k_vector_from_tile_h = K_tile + (local_thread_idx * params.head_dim);
            thread_score_val = dot_product_qk(q_shmem, k_vector_from_tile_h, params);
        }

        // --- 10.1.2/10: History Tile - Local Max (m_local_tile) Reduction ---
        float current_thread_score_for_max_reduction = thread_score_val;

        tg_partial_reduce_scratch[local_thread_idx] = current_thread_score_for_max_reduction;
        // No barrier needed: each thread only reads its own index for simd_max

        float simd_max_m_tile_val = simd_max(tg_partial_reduce_scratch[local_thread_idx]);
        if (simd_lane_id == 0) { tg_simd_reduce_scratch[simd_group_id] = simd_max_m_tile_val; }
        // Thread 0 later reads all per-simd-group maxes from tg_simd_reduce_scratch
        // All simdgroup partial sums must be visible before thread 0 reduces
        threadgroup_barrier(mem_flags::mem_threadgroup); // Ensure G_simd_reduced_maxes written

        float m_local_tile_val = -INFINITY;
        if (local_thread_idx == 0) {
            if (current_hist_tile_actual_len == 0) {
                // Empty tile: contribute nothing
                m_local_tile_val = -INFINITY;
            } else {
                // Start with first SIMD‑group max, then fold the rest
                m_local_tile_val = tg_simd_reduce_scratch[0];
                for (uint sg_idx = 1; sg_idx < num_simd_groups; ++sg_idx) {
                    m_local_tile_val = max(m_local_tile_val, tg_simd_reduce_scratch[sg_idx]);
                }
            }
            tg_simd_reduce_scratch[0] = m_local_tile_val;
        }
        // Broadcast m_local_tile_val from thread 0 to the rest of the group
        // Share d_local_tile_total_val with all threads
        threadgroup_barrier(mem_flags::mem_threadgroup);
        m_local_tile_val = tg_simd_reduce_scratch[0];

        // --- 10.1.3/10: History Tile - Compute Exponentiated Values (no Score Tile) ---
        float thread_exp_val = 0.0f;
        if (local_thread_idx < current_hist_tile_actual_len &&
            m_local_tile_val != -INFINITY &&
            thread_score_val != -INFINITY) {
            thread_exp_val = precise::exp(max(thread_score_val - m_local_tile_val,
                                        params.log_exp_min_clamp));
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
            tg_simd_exp_sums_scratch[0] = d_local_tile_total_val;
        }
        // Share d_local_tile_total_val with all threads
        threadgroup_barrier(mem_flags::mem_threadgroup);
        d_local_tile_total_val = tg_simd_exp_sums_scratch[0];

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
        // All threads use updated global stats and scale factor for the next iteration
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float m_global_current_iter_atomic = (*tg_global_stats).x;
        float scale_for_acc_iter_atomic = tg_simd_reduce_scratch[0]; // This is scale_f from update_softmax_stats_kahan

        if (scale_for_acc_iter_atomic != 1.0f) {
            // Rescale previously accumulated V-values.
            // If m_global increased due to the current tile's m_local_tile,
            // previous contributions to acc_tile_local_fp32 were effectively based on an older, smaller m_global.
            // This scale_for_acc_iter_atomic adjusts them to the new m_global for consistent normalization.
            for (uint d = 0; d < params.head_dim; d += 4) {
                float4 acc_chunk = float4(acc_tile_local_fp32[d],
                                        (d + 1 < params.head_dim) ? acc_tile_local_fp32[d+1] : 0.0f,
                                        (d + 2 < params.head_dim) ? acc_tile_local_fp32[d+2] : 0.0f,
                                        (d + 3 < params.head_dim) ? acc_tile_local_fp32[d+3] : 0.0f);
                acc_chunk *= scale_for_acc_iter_atomic;
                acc_tile_local_fp32[d] = acc_chunk.x;
                if (d + 1 < params.head_dim) acc_tile_local_fp32[d+1] = acc_chunk.y;
                if (d + 2 < params.head_dim) acc_tile_local_fp32[d+2] = acc_chunk.z;
                if (d + 3 < params.head_dim) acc_tile_local_fp32[d+3] = acc_chunk.w;
            }
        }

        // --- 10.1.6/10: History Tile - Weighted V Accumulation (Fused Path) ---
        if (local_thread_idx < current_hist_tile_actual_len) {

            threadgroup const half* v_vector_from_tile_h = V_tile + (local_thread_idx * params.head_dim);

            float weight_term = thread_exp_val;
            float exp_term = precise::exp(max(m_local_tile_val - m_global_current_iter_atomic, params.log_exp_min_clamp));
            float final_p_attn_weight_numerator = weight_term * exp_term;

            for (uint d_idx = 0; d_idx < params.head_dim; d_idx += 4) {
                // Explicitly construct float4 from half values in V_tile
                float4 v_chunk_fp32 = float4(
                    (d_idx + 0 < params.head_dim) ? float(v_vector_from_tile_h[d_idx + 0]) : 0.0f,
                    (d_idx + 1 < params.head_dim) ? float(v_vector_from_tile_h[d_idx + 1]) : 0.0f,
                    (d_idx + 2 < params.head_dim) ? float(v_vector_from_tile_h[d_idx + 2]) : 0.0f,
                    (d_idx + 3 < params.head_dim) ? float(v_vector_from_tile_h[d_idx + 3]) : 0.0f
                );

                v_chunk_fp32 *= final_p_attn_weight_numerator; // Multiply as floats

                // Accumulate into FP32 accumulator
                acc_tile_local_fp32[d_idx]     += v_chunk_fp32.x;
                if (d_idx + 1 < params.head_dim) acc_tile_local_fp32[d_idx + 1] += v_chunk_fp32.y;
                if (d_idx + 2 < params.head_dim) acc_tile_local_fp32[d_idx + 2] += v_chunk_fp32.z;
                if (d_idx + 3 < params.head_dim) acc_tile_local_fp32[d_idx + 3] += v_chunk_fp32.w;
            }
        }
    } // End history tiling loop

    // --- 10.2/10: Final Normalization & Output Write (Fused Path) ---
    // Use the threadgroup shared final s_global value from g_global_stats_ptr
    float s_global_final = (*tg_global_stats).y;
    float inv_s_global = (s_global_final > kEpsilonForZeroGuard) ? fast::divide(1.0f, s_global_final) : 0.0f;

    // Normalize the full acc_tile_local_fp32 using float4 chunks
    for (uint i = 0; i < params.head_dim; i += 4) {
        thread float4* chunk_ptr = reinterpret_cast<thread float4*>(acc_tile_local_fp32 + i);
        float4 chunk = *chunk_ptr;
        chunk *= inv_s_global;
        *chunk_ptr = chunk;
    }

    // Reduce the now-normalized acc_tile_local_fp32 across the threadgroup and write to output_buffer
    for (uint i = 0; i < params.head_dim; i += 4) {
        float4 chunk_to_write = float4(0.0f);
        if (i < params.head_dim)     chunk_to_write.x = acc_tile_local_fp32[i+0];
        if (i+1 < params.head_dim) chunk_to_write.y = acc_tile_local_fp32[i+1];
        if (i+2 < params.head_dim) chunk_to_write.z = acc_tile_local_fp32[i+2];
        if (i+3 < params.head_dim) chunk_to_write.w = acc_tile_local_fp32[i+3];

        float4 reduced_simd_group_final_chunk;
        reduced_simd_group_final_chunk.x = simd_sum(chunk_to_write.x);
        reduced_simd_group_final_chunk.y = simd_sum(chunk_to_write.y);
        reduced_simd_group_final_chunk.z = simd_sum(chunk_to_write.z);
        reduced_simd_group_final_chunk.w = simd_sum(chunk_to_write.w);

        if (simd_lane_id == 0) {
            tg_simd_v_chunk_sums[simd_group_id] = reduced_simd_group_final_chunk;
        }
        // Wait for all simd groups to produce their partial sums before thread 0 combines them
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
        // Reuse tg_simd_v_chunk_sums for the next chunk, ensure previous values consumed
        threadgroup_barrier(mem_flags::mem_threadgroup); // Sync before next chunk
    }


} // End of kernel
