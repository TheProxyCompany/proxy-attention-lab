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
#include <metal_atomic>
#include "paged_attention.h.metal"

using namespace metal;

/**
 * paged_attn_prefill_kernel
 * Pass 1 of the new page-centric prefill architecture.
 * --------------------------
 *
 */
[[kernel]] void paged_attn_prefill_kernel(
    device      const half*     queries_in                      [[buffer(0)]],
    device      const half*     k_cache_pool_in                 [[buffer(1)]],
    device      const half*     v_cache_pool_in                 [[buffer(2)]],
    device      const uint*     page_table_in                   [[buffer(3)]],
    device      const int*      sequence_lengths_in             [[buffer(4)]],
    device      const int*      query_to_seq_map_in             [[buffer(5)]],
    device      const int*      query_token_offset_in           [[buffer(6)]],
    constant    const           PagedAttentionParams& params    [[buffer(7)]],      // Parameters
    device      const uint2*    active_work_item_pairs          [[buffer(8)]],      // Active (batch_item, logical_page) pairs
    device      const uint*     query_starts_for_batch_item_arr [[buffer(9)]],      // Query starts for each batch item
    device      float*          m_locals_pass1_out              [[buffer(10)]],     // Local max scores
    device      float*          s_locals_pass1_out              [[buffer(11)]],     // Local sum-exponentials
    device      half*           o_partials_pass1_out            [[buffer(12)]],     // Unnormalized partial V-accumulations
    uint        actual_simd_width                               [[threads_per_simdgroup]],
    threadgroup float* tg_mem                                   [[threadgroup(0)]],
    uint3       tg_pos_in_grid                                  [[threadgroup_position_in_grid]],
    uint3       tg_dim                                          [[threads_per_threadgroup]],
    uint        local_idx_in_tg                                 [[thread_index_in_threadgroup]],
    uint        simd_lane_id                                    [[thread_index_in_simdgroup]],
    uint        simd_group_id                                   [[simdgroup_index_in_threadgroup]]
)
{
    // A. TGMem Carving for Q_shmem_block, K_tile, and V_tile
    threadgroup uchar* tg_mem_base_byte_ptr = (threadgroup uchar*)tg_mem;
    uintptr_t current_offset = 0;

    // D_s (params.tokens_per_page) is the depth for Q-block, K-tile, and V-tile
    const uint D_s = params.tokens_per_page;
    const uint N_q_per_kv = (params.num_q_heads + params.num_kv_heads - 1) / params.num_kv_heads; // GQA Factor

    // Corrected Q_shmem_base allocation for the Q-block
    // Holds D_s Q-vectors, each for N_q_per_kv heads, stored as float
    current_offset = (current_offset + kAlignmentMask) & ~kAlignmentMask;
    threadgroup float* Q_shmem_base = (threadgroup float*)(tg_mem_base_byte_ptr + current_offset);
    // Each of the N_q_per_kv SIMD groups will get a slice of this Q_shmem_base
    // to store D_s Q-vectors. Total size is N_q_per_kv * D_s * params.head_dim * sizeof(float).
    current_offset += N_q_per_kv * D_s * params.head_dim * sizeof(float); // FIX 1: Corrected size for Q_shmem_block

    // K_tile allocation (depth D_s)
    current_offset = (current_offset + kAlignmentMask) & ~kAlignmentMask;
    threadgroup half* K_tile = (threadgroup half*)(tg_mem_base_byte_ptr + current_offset);
    current_offset += D_s * params.head_dim * sizeof(half);

    // V_tile allocation (depth D_s)
    current_offset = (current_offset + kAlignmentMask) & ~kAlignmentMask;
    threadgroup half* V_tile = (threadgroup half*)(tg_mem_base_byte_ptr + current_offset);
    current_offset += D_s * params.head_dim * sizeof(half);

    // V_Sum_Accumulators_Area allocation (depth N_q_per_kv)
    const uint total_simd_groups_in_tg_metal = tg_dim.x / actual_simd_width;
    current_offset = (current_offset + kAlignmentMask) & ~kAlignmentMask;
    threadgroup float* V_Sum_Accumulators_Area = (threadgroup float*)(tg_mem_base_byte_ptr + current_offset);
    current_offset += total_simd_groups_in_tg_metal * params.head_dim * sizeof(float);

    // B. Role Identification
    uint flat_work_item_idx = tg_pos_in_grid.x;
    if (flat_work_item_idx >= params.num_active_batch_logical_pages) {
        return;
    }
    uint2 work_item_pair = active_work_item_pairs[flat_work_item_idx];
    uint assigned_batch_item_idx = work_item_pair.x;
    uint assigned_logical_page_idx_in_sequence = work_item_pair.y;
    uint assigned_global_kv_head_idx = tg_pos_in_grid.y;

    // C. Physical Page ID Lookup
    uint page_table_flat_idx = assigned_batch_item_idx * params.max_logical_blocks_per_seq + assigned_logical_page_idx_in_sequence;
    uint physical_page_id = page_table_in[page_table_flat_idx];
    if (physical_page_id >= params.num_physical_pages_in_pool) {
        return;
    }

    // D. Cooperative K/V Loading into TGMem
    const uint threads_per_tg = tg_dim.x;
    const uint chunks_per_row_kv = params.head_dim / 4; // For K/V (half4)

    // Pre-compute page base offset for K/V cache pool
    // Note: params.tokens_per_page is D_s here.
    const ulong page_base_offset_global_kv = (ulong)physical_page_id *
                                            (ulong)params.tokens_per_page * // D_s
                                            (ulong)params.num_kv_heads *
                                            (ulong)params.head_dim;
    // Load K-tile
    for (uint token_idx_on_page = 0; token_idx_on_page < D_s; ++token_idx_on_page) {
        ulong k_global_offset = page_base_offset_global_kv +
                                (ulong)token_idx_on_page * (ulong)params.num_kv_heads * (ulong)params.head_dim +
                                (ulong)assigned_global_kv_head_idx * (ulong)params.head_dim;
        device const half* k_src_ptr = k_cache_pool_in + k_global_offset;
        threadgroup half* k_dst_ptr = K_tile + token_idx_on_page * params.head_dim;
        threadgroup half4* dst_k_vector_h4_ptr = (threadgroup half4*)(k_dst_ptr);
        device const half4* src_k_vector_h4_ptr = (device const half4*)(k_src_ptr);
        for (uint chunk_idx = local_idx_in_tg; chunk_idx < chunks_per_row_kv; chunk_idx += threads_per_tg) {
            dst_k_vector_h4_ptr[chunk_idx] = src_k_vector_h4_ptr[chunk_idx];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Load V-tile
    for (uint token_idx_on_page = 0; token_idx_on_page < D_s; ++token_idx_on_page) {
        ulong v_global_offset = page_base_offset_global_kv + // Same base as K
                                (ulong)token_idx_on_page * (ulong)params.num_kv_heads * (ulong)params.head_dim +
                                (ulong)assigned_global_kv_head_idx * (ulong)params.head_dim;
        device const half* v_src_ptr = v_cache_pool_in + v_global_offset;
        threadgroup half* v_dst_ptr = V_tile + token_idx_on_page * params.head_dim;
        threadgroup half4* dst_v_vector_h4_ptr = (threadgroup half4*)(v_dst_ptr);
        device const half4* src_v_vector_h4_ptr = (device const half4*)(v_src_ptr);
        for (uint chunk_idx = local_idx_in_tg; chunk_idx < chunks_per_row_kv; chunk_idx += threads_per_tg) {
            dst_v_vector_h4_ptr[chunk_idx] = src_v_vector_h4_ptr[chunk_idx];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup); // K_tile and V_tile (depth D_s) are now loaded.

    // E. Query Block Iteration and Loading
    uint seq_len_for_this_batch_item = (uint)sequence_lengths_in[assigned_batch_item_idx];
    // Iterate over the Q-BLOCKs for this batch item (process D_s tokens (params.tokens_per_page) at once)
    for (uint q_block_start_local_idx = 0;
        q_block_start_local_idx < seq_len_for_this_batch_item;
        q_block_start_local_idx += D_s /* params.tokens_per_page is D_s */) {

        uint num_queries_in_this_block = min(D_s, seq_len_for_this_batch_item - q_block_start_local_idx);

        // Calculate the number of SIMD groups per GQA stream
        const uint total_simd_groups_in_tg = tg_dim.x / actual_simd_width;
        const uint SIMD_GROUPS_PER_GQA_GROUP_FACTOR_simd_groups_per_gqa_stream = total_simd_groups_in_tg / N_q_per_kv;

        uint gqa_stream_idx_for_this_simd_group = simd_group_id / SIMD_GROUPS_PER_GQA_GROUP_FACTOR_simd_groups_per_gqa_stream;
        uint sub_simd_group_idx_within_stream = simd_group_id % SIMD_GROUPS_PER_GQA_GROUP_FACTOR_simd_groups_per_gqa_stream;

        if (gqa_stream_idx_for_this_simd_group < N_q_per_kv) { // Check if this SIMD group is part of an active GQA stream

            uint target_q_head_local_offset_in_gqa_group = gqa_stream_idx_for_this_simd_group;
            uint target_global_q_head_idx = (assigned_global_kv_head_idx * N_q_per_kv) + target_q_head_local_offset_in_gqa_group;

            // Base pointer in Q_shmem_base for this GQA stream's entire block of D_s Q-vectors
            threadgroup float* q_block_shmem_for_gqa_stream = Q_shmem_base +
                                    (gqa_stream_idx_for_this_simd_group * D_s * params.head_dim);

            // Iterate through the Q-vectors that *this specific SIMD group* (within its GQA stream's assigned SIMD groups) is responsible for loading.
            for (uint q_idx_in_block_for_this_sg = sub_simd_group_idx_within_stream;
                q_idx_in_block_for_this_sg < num_queries_in_this_block;
                q_idx_in_block_for_this_sg += SIMD_GROUPS_PER_GQA_GROUP_FACTOR_simd_groups_per_gqa_stream) {

                uint current_query_local_idx = q_block_start_local_idx + q_idx_in_block_for_this_sg;
                uint master_query_idx = query_starts_for_batch_item_arr[assigned_batch_item_idx] + current_query_local_idx;

                // Global source pointer for this specific Q-vector
                // Assuming queries_in is physically [NumQHeads, TotalQueries, HeadDim]
                device const half* q_head_slice_ptr = queries_in +
                    (target_global_q_head_idx * params.query_token_count_total * params.head_dim);
                device const half* q_src_global_ptr = q_head_slice_ptr +
                    (master_query_idx * params.head_dim);

                // Destination in Q_shmem for this specific Q-vector
                threadgroup float* q_dest_specific_q_in_shmem = q_block_shmem_for_gqa_stream +
                                                                (q_idx_in_block_for_this_sg * params.head_dim);

                // The actual_simd_width lanes of THIS SIMD group load this ONE Q-vector
                uint chunks_per_q_vector = params.head_dim / 4; // Qs are float4 in TGMem from half4 global
                for (uint c_idx = simd_lane_id; c_idx < chunks_per_q_vector; c_idx += actual_simd_width) {
                    ((threadgroup float4*)q_dest_specific_q_in_shmem)[c_idx] =
                        float4(((device const half4*)q_src_global_ptr)[c_idx]) * params.inv_sqrt_head_dim;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup); // Entire Q-BLOCK is now in Q_shmem_base.

        // --- START: Zero V_Sum_Accumulators_Area ONCE PER Q_BLOCK ---
        uint total_floats_in_v_acc_area = total_simd_groups_in_tg_metal * params.head_dim;
        uint num_float4s_to_zero = total_floats_in_v_acc_area / 4;
        uint remainder_floats = total_floats_in_v_acc_area % 4;

        // Vectorized zeroing using float4
        threadgroup float4* v_acc_area_f4_ptr = (threadgroup float4*)V_Sum_Accumulators_Area;
        for (uint i = local_idx_in_tg; i < num_float4s_to_zero; i += tg_dim.x) {
            v_acc_area_f4_ptr[i] = float4(0.0f);
        }

        // Handle remainder floats
        if (remainder_floats > 0 && local_idx_in_tg < remainder_floats) {
            V_Sum_Accumulators_Area[num_float4s_to_zero * 4 + local_idx_in_tg] = 0.0f;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
        // --- END: Zero V_Sum_Accumulators_Area ---

        // F. Compute QK^T
        if (gqa_stream_idx_for_this_simd_group >= N_q_per_kv) {
            // This SIMD group is outside of the active GQA stream
            continue;
        }

        threadgroup float* q_block_shmem_for_this_gqa_stream_for_compute = Q_shmem_base +
                                    (gqa_stream_idx_for_this_simd_group * D_s * params.head_dim);

        // Proceed with "Row Strip" compute for this SIMD group.
        // This loop iterates over queries within the current block assigned to this SIMD group.
        // It forms the core of the QK^T computation and subsequent V-accumulation.
        // Profiling has indicated this section as a significant contributor to latency,
        // particularly as the number of queries (sequence length) increases.
        for (uint q_idx_in_block_for_this_sg = sub_simd_group_idx_within_stream;
              q_idx_in_block_for_this_sg < num_queries_in_this_block; // num_queries_in_this_block is D_s
              q_idx_in_block_for_this_sg += SIMD_GROUPS_PER_GQA_GROUP_FACTOR_simd_groups_per_gqa_stream
        ) {
            // F.1. Per-Query Setup
            // F.1.a: Calculate the local index of this query within the original full sequence
            uint current_query_local_idx_in_sequence = q_block_start_local_idx + q_idx_in_block_for_this_sg;
            // F.1.b: Calculate the master (global) index for this query across the entire batch.
            uint master_query_idx = query_starts_for_batch_item_arr[assigned_batch_item_idx] + current_query_local_idx_in_sequence;
            // F.1.c: Get the logical position (offset) of this query token within its sequence (for causal masking).
            uint current_q_logical_pos = (uint)query_token_offset_in[master_query_idx];
            // F.1.d: Determine the target global Q head index this SIMD group is working for.
            uint target_global_q_head_idx = (assigned_global_kv_head_idx * N_q_per_kv) + gqa_stream_idx_for_this_simd_group;

            // F.2.a: Get pointer to the current Q-vector in Q_shmem_block.
            threadgroup const float* q_vec_ptr = q_block_shmem_for_this_gqa_stream_for_compute +
                                                 (q_idx_in_block_for_this_sg * params.head_dim);
            // F.2.b: Get pointer to this SIMD group's dedicated V-sum accumulator.
            threadgroup float* v_sum_accumulator_ptr = V_Sum_Accumulators_Area +
                                                       (simd_group_id * params.head_dim);
            // F.2.c: Initialize this SIMD group's per-Q softmax statistics (will be held in registers)
            float page_max_score = -INFINITY;
            float page_sum_exp_norm_by_page_max = 0.0f;
            float kahan_c_for_sum_exp = 0.0f; // Kahan compensation term for page_sum_exp

            // F.3. Loop through K/V vectors in the loaded K_tile / V_tile
            for (uint k_idx_in_tile = 0; k_idx_in_tile < D_s; ++k_idx_in_tile) {
                // F.3.a. Per-K/V vector setup
                threadgroup const half* k_vec_hist_ptr = K_tile + (k_idx_in_tile * params.head_dim);
                // Determine the logical position of this history token in its original sequence.
                uint history_token_logical_pos = (assigned_logical_page_idx_in_sequence * D_s) + k_idx_in_tile;

                // F.3.b. Causal Masking / Length Check
                if (history_token_logical_pos >= current_q_logical_pos ||
                    history_token_logical_pos >= seq_len_for_this_batch_item) {
                    continue; // Skip this history K/V pair if it's non-causal or out of bounds
                }

                // F.3.c. QK^T dot product
                float per_lane_partial_score = 0.0f;
                uint num_f4_chunks_total = params.head_dim / 4;

                for (uint f4_chunk_idx = simd_lane_id; f4_chunk_idx < num_f4_chunks_total; f4_chunk_idx += actual_simd_width) {
                    uint d_offset = f4_chunk_idx * 4;
                    float4 qv = *((threadgroup const float4*)(q_vec_ptr + d_offset));
                    float4 kv = float4(*((threadgroup const half4*)(k_vec_hist_ptr + d_offset)));
                    per_lane_partial_score += dot(qv, kv);
                }
                float score = simd_sum(per_lane_partial_score);
                // Ensure all lanes have the same value
                score = simd_broadcast_first(score);
                // float score = 1.0f; // comment to skip main compute

                // F.3.d. Online Softmax Update
                float old_page_max_score_val = page_max_score;
                page_max_score = max(page_max_score, score);

                // Term for the current score, normalized by the new page_max_score.
                float current_score_exp_contribution;

                // If page_max_score changed, we need to rescale the existing sum and its Kahan compensator.
                if (page_max_score > old_page_max_score_val && old_page_max_score_val != -INFINITY) {
                    // Clamp argument to prevent underflow
                    float rescale_exp_arg = max(old_page_max_score_val - page_max_score, params.log_exp_min_clamp);
                    float actual_scale_factor = precise::exp(rescale_exp_arg);

                    // check if we need to rescale
                    if (fabs(actual_scale_factor - 1.0f) > 1e-6f) {
                        page_sum_exp_norm_by_page_max *= actual_scale_factor;
                        kahan_c_for_sum_exp *= actual_scale_factor; // Rescale Kahan compensation term

                        // Rescale the existing sum and its Kahan compensator.
                        for (uint h_rescale_idx = simd_lane_id;
                            h_rescale_idx < params.head_dim / 4;
                            h_rescale_idx += actual_simd_width)
                        {
                            uint d = h_rescale_idx * 4;
                            *((threadgroup float4*)(v_sum_accumulator_ptr + d)) *= actual_scale_factor;
                        }
                    }
                }

                // Calculate the exponential of (current score - new_page_max_score)
                float current_term_exp_arg = max(score - page_max_score, params.log_exp_min_clamp);
                current_score_exp_contribution = precise::exp(current_term_exp_arg);

                // Kahan summation to add current_score_exp_contribution to page_sum_exp_norm_by_page_max
                float y_kahan = current_score_exp_contribution - kahan_c_for_sum_exp;
                float t_kahan = page_sum_exp_norm_by_page_max + y_kahan;
                kahan_c_for_sum_exp = (t_kahan - page_sum_exp_norm_by_page_max) - y_kahan;
                page_sum_exp_norm_by_page_max = t_kahan;
                // --- END F.3.d: Online Softmax Update ---

                // F.3.e. V-Aggregation
                threadgroup const half* v_vec_hist_ptr = V_tile + (k_idx_in_tile * params.head_dim);
                float weight_exp_arg = (page_max_score == -INFINITY && score == -INFINITY) ?
                                       params.log_exp_min_clamp :
                                       max(score - page_max_score, params.log_exp_min_clamp);

                if (current_score_exp_contribution < kEpsilonForZeroGuard) {
                    // Skip this V-vector if the current score's exp contribution is effectively zero.
                    continue;
                }

                // Each lane (simd_lane_id) processes different chunks of the head_dim.
                for (uint h_chunk_idx = simd_lane_id;
                      h_chunk_idx < (params.head_dim / 4);
                      h_chunk_idx += actual_simd_width) {
                    // Calculate the actual memory offset for this float4 chunk
                    uint h_dim_offset = h_chunk_idx * 4;

                    // Load V_vector chunk (half4) from V_tile and convert to float4
                    float4 v_chunk_f = float4( *((threadgroup const half4*)(v_vec_hist_ptr + h_dim_offset)) );

                    // Load current accumulator value
                    float4 current_acc = *((threadgroup float4*)(v_sum_accumulator_ptr + h_dim_offset));

                    // Perform FMA and store back
                    float4 updated_acc;
                    updated_acc.x = fma(current_score_exp_contribution, v_chunk_f.x, current_acc.x);
                    updated_acc.y = fma(current_score_exp_contribution, v_chunk_f.y, current_acc.y);
                    updated_acc.z = fma(current_score_exp_contribution, v_chunk_f.z, current_acc.z);
                    updated_acc.w = fma(current_score_exp_contribution, v_chunk_f.w, current_acc.w);

                    *((threadgroup float4*)(v_sum_accumulator_ptr + h_dim_offset)) = updated_acc;
                }
                // --- END F.3.e: V-Aggregation ---
            } // end of k_idx_in_tile loop

            // F.4. Write results for this (master_query_idx, target_global_q_head_idx, flat_work_item_idx)
             if (simd_lane_id == 0) { // Only lane 0 writes the scalar M and S values
                ulong ms_base_offset = (ulong)master_query_idx * params.num_q_heads * params.num_active_batch_logical_pages +
                                       (ulong)target_global_q_head_idx * params.num_active_batch_logical_pages;
                ulong ms_flat_output_idx = ms_base_offset + flat_work_item_idx;

                m_locals_pass1_out[ms_flat_output_idx] = page_max_score;
                s_locals_pass1_out[ms_flat_output_idx] = page_sum_exp_norm_by_page_max;
            }

            ulong o_base_offset = (ulong)master_query_idx * params.num_q_heads * params.num_active_batch_logical_pages * params.head_dim +
                                    (ulong)target_global_q_head_idx * params.num_active_batch_logical_pages * params.head_dim +
                                    (ulong)flat_work_item_idx * params.head_dim;

            device half* o_dest_ptr = o_partials_pass1_out + o_base_offset;

            // Each lane handles a float4 chunk
            for (uint h_chunk_idx = simd_lane_id;
                h_chunk_idx < (params.head_dim / 4);
                h_chunk_idx += actual_simd_width) {

                uint h_dim_offset = h_chunk_idx * 4;
                float4 val_f4 = *((threadgroup float4*)(v_sum_accumulator_ptr + h_dim_offset));
                *((device half4*)(o_dest_ptr + h_dim_offset)) = half4(val_f4);
            }
            // all lanes finished their chunk
        } // end of k_idx_in_tile loop
    } // end of q_block_start_local_idx loop
} // End of kernel

// =================================================================================================
// KERNEL DEVELOPMENT NOTES & CURRENT STATUS (As of 2025-05-29, after F.4 implementation)
//
// I. KERNEL OBJECTIVE (Pass 1 of Two-Pass Page-Centric Prefill):
//    This kernel is responsible for Pass 1 of a two-pass prefill strategy.
//    - Dispatch: One ThreadGroup (TG) per (BatchItem, Logical K/V Page, KVHead).
//    - Input: Global Q, K-cache, V-cache, page tables, sequence metadata.
//    - Tiling:
//        - Symmetric QKV Tiling: A dynamically calculated depth 'D_s' (params.tokens_per_page
//          for prefill) is used for K_tile, V_tile, and Q-blocks within TGMem.
//        - K_tile & V_tile: Depth D_s, loaded once per TG for its assigned K/V page.
//        - Q-block: Depth D_s Q-vectors, loaded iteratively to cover all queries for the
//          assigned BatchItem. Qs are loaded into Q_shmem_base.
//    - Output: For each (Query Token, Query Head) processed against the TG's K/V page,
//      this kernel writes out intermediate results to global buffers:
//        - m_locals_pass1_out: The local maximum score (page_max_score).
//        - s_locals_pass1_out: The local sum of exponentials normalized by page_max_score
//                              (page_sum_exp_norm_by_page_max).
//        - o_partials_pass1_out: The V-vectors weighted by exp(score - page_max_score) and
//                                accumulated (unnormalized partial V-sum for this page).
//
// II. CURRENT IMPLEMENTATION STATE:
//    1. TGMem Carving (Section A):
//        - Correctly allocates space for Q_shmem_block (N_q_per_kv * D_s * HeadDim floats),
//          K_tile (D_s * HeadDim halves), V_tile (D_s * HeadDim halves), and
//          V_Sum_Accumulators_Area (total_simd_groups_in_tg * HeadDim floats).
//        - D_s calculation in C++ correctly accounts for all these TGMem regions.
//
//    2. Role Identification & K/V Loading (Sections B, C, D):
//        - TG correctly identifies its assigned BatchItem, LogicalPage, and KVHead.
//        - K_tile and V_tile loading from global K/V cache pools is implemented and
//          cooperatively performed by TG threads.
//
//    3. Q-Block Iteration & Loading (Section E):
//        - Iterates through Q-blocks of depth D_s to cover all queries for the BatchItem.
//        - Q-block loading into Q_shmem_base is parallelized by K_FACTOR SIMD groups
//          per GQA stream (SIMD_GROUPS_PER_GQA_GROUP_FACTOR_simd_groups_per_gqa_stream).
//        - Qs are scaled by inv_sqrt_head_dim upon loading into Q_shmem_base.
//
//    4. V_Sum_Accumulators_Area Zeroing:
//        - Correctly zeroed once per Q-block iteration by all TG threads, covering the
//          full area required for one accumulator per SIMD group.
//
//    5. Compute Phase (Section F - QK^T, Softmax, V-Agg, Writes):
//        - "Row Strip" Parallelism: The D_s Q-vectors within a Q-block (for a given GQA stream)
//          are distributed among the K_FACTOR SIMD groups assigned to that stream. Each
//          SIMD group processes its subset of Q-vectors.
//        - Per-Q Processing: For each assigned Q-vector, the SIMD group:
//            - F.3.a (K/V Setup): Sets up pointers to K_tile, V_tile, and calculates
//              history_token_logical_pos for the inner K-loop.
//            - F.3.b (Masking): Implements causal and sequence length masking for K/V pairs.
//            - F.3.c (QK^T): Computes dot product score using dot_product_qk helper.
//            - F.3.d (Online Softmax): Updates per-Q page_max_score and
//              page_sum_exp_norm_by_page_max (using Kahan summation) for each valid score.
//            - F.3.e (V-Aggregation): Calculates attention_weight_numerator and accumulates
//              weighted V-vectors into its dedicated slice of V_Sum_Accumulators_Area.
//              The HeadDim-wide accumulation is vectorized across SIMD lanes.
//        - F.4 (Write Outputs): After processing a Q-vector against all K/V pairs in the
//          tile, the SIMD group writes its final page_max_score, page_sum_exp_norm_by_page_max,
//          and the contents of its V-sum accumulator to the global intermediate buffers
//          (m_locals_pass1_out, s_locals_pass1_out, o_partials_pass1_out).
//          The write to o_partials_pass1_out is vectorized across SIMD lanes.
//
// III. PERFORMANCE CHARACTERISTICS & OBSERVATIONS (as of 2025-05-29):
//
//    A. Load-Only Phase (Compute sections F.3.c-e were stubs/minimal ops):
//        - Initial K/V tile loading (D_s depth): Scales very well, O(S^~0.3-0.4) latency.
//        - Q-Block Loading (K_FACTOR=6):
//            - Benchmark (2025-05-29, before V_Sum_Accumulators_Area sizing fix):
//              "cpp_pal_paged_attention_prefill": { "4096.0": 13.8087 } ms. Slope ~S^1.0.
//            - Benchmark (2025-05-29, after C++ V_Sum fix impacting D_s):
//              "cpp_pal_paged_attention_prefill": { "4096.0": 12.1743 } ms. Slope ~S^1.0.
//            - Benchmark (2025-05-29, after Metal V_Sum TGMem fix & loop structure for F):
//              "cpp_pal_paged_attention_prefill": { "4096.0": 12.3909 } ms. Slope ~S^1.0.
//        - The load phase provides significant headroom compared to full MLX SDPA compute.
//
//    B. With Compute Phase (F.3.a-e) Implemented (but F.4 writes not yet added):
//        - Benchmark (2025-05-29):
//          "cpp_pal_paged_attention_prefill": { "4096.0": 722.8587 } ms.
//        - Observation: Significant latency increase, GPU utilization at 100%, high power draw.
//          This indicates the compute operations are "live" and substantial.
//          The scaling slope appeared to be ~S^1.70 on a log-log plot.
//
//    C. With Full Pass 1 Compute (F.3.a-e + F.4 Writes) Implemented:
//        - Benchmark (2025-05-29, current):
//          "cpp_pal_paged_attention_prefill": { "4096.0": 829.6074 } ms.
//        - MLX SDPA for reference (same run): { "4096.0": 29.0186 } ms.
//        - Observation: Further latency increase due to global writes.
//          The PAL Pass 1 is ~28.6x slower than MLX SDPA at 4096 tokens.
//          The scaling slope for PAL Pass 1 remains poor (~S^1.70), while MLX SDPA is ~S^1.16.
//
// IV. CURRENT THINKING & NEXT STEPS:
//    1. High Absolute Latency: The current compute path (iterating D_s K-vectors, each involving
//       a dot product over HeadDim, online softmax ops, and a V-aggregation over HeadDim)
//       is computationally intensive. This contributes to the high "y-intercept".
//    2. S^1.70 Scaling Concern: The primary hypothesis for this non-linear scaling is the
//       total global memory read volume for Query vectors (`queries_in`). Each of the ~S/D_s
//       threadgroups (for a sequence of length S) currently reads all ~S query vectors
//       for that sequence over its Q-block iterations. This results in an S^2/D_s total
//       Q-read volume from global memory, likely becoming the bottleneck at larger S.
//    3. Path Forward:
//        - This Pass 1 kernel is now considered *functionally complete*.
//        - Next immediate step: Implement Pass 2 (`paged_attn_prefill_pass2.metal`) to consume
//          the intermediate outputs from Pass 1 and produce the final normalized attention.
//        - Correctness Verification: Once Pass 1 + Pass 2 are complete, rigorously verify
//          numerical correctness against a Python reference model of the two-pass algorithm.
//        - Performance Optimization (Post-Correctness):
//            - Re-evaluate Pass 1 performance.
//            - If S^1.70 scaling persists and is the primary limiter, architectural changes
//              to Pass 1's Q-vector access/sharing strategy will be needed.
//            - Micro-optimizations within Pass 1 compute (e.g., K-micro-tiling for the
//              1Q vs D_s Ks interaction, register tuning) can be explored.
//            - Optimize Pass 2 for efficiency.
//
// V. OPEN QUESTIONS / FUTURE OPTIMIZATIONS CONSIDERED (Post-Functional Pass 1 & 2):
//    - Advanced intra-SIMD group compute for the (1 Q vs D_s Ks) part (K-micro-tiling).
//    - Asynchronous Q-block loading (double buffering Q_shmem_block) if Q-load still shows
//      up as a bottleneck after compute optimization.
//    - Reducing global memory traffic for Q-vectors in Pass 1 for very long sequences.
//
// This iterative approach (implement, benchmark, verify correctness, then optimize) is key.
// The current high latency and S^1.70 scaling are understood in context of the work being done
// and the current data flow for Qs.
// =================================================================================================

//
// VI. ADDENDUM (2025-05-29): Impact of QK^T Compute on Performance
//    - Experiment: To isolate the cost of the F.3.c `dot_product_qk` calls, an experiment
//      was run where the actual dot product was commented out and `score` was set to a
//      constant `1.0f`
//      All other logic (online softmax F.3.d, V-aggregation F.3.e, writes F.4) remained.
//
//    - Benchmark Results (`score = 1.0f` vs. full `dot_product_qk`):
//        - Full `dot_product_qk` (4096 tokens): ~830 ms, Log-log slope ~1.70
//        - `score = 1.0f` (4096 tokens): ~87 ms, Log-log slope ~1.26
//
//    - Key Insights from this Experiment:
//        1. Absolute Latency: The `dot_product_qk` computation (iterated D_s times per Q
//           processed by a SIMD group) is a very significant contributor to the kernel's
//           absolute latency ("y-intercept" of the performance curve).
//        2. Scaling Slope: The heavy compute of `dot_product_qk` also negatively impacted
//           the overall latency scaling with sequence length, pushing it from ~S^1.26
//           (when compute is minimal) towards ~S^1.70. This suggests that while the
//           underlying S^2 global Q-read traffic is likely responsible for the > S^1.0
//           scaling, the intense per-QK compute was exacerbating this, possibly by
//           saturating other hardware resources more quickly.
//        3. Path for Optimization: Optimizing the QK^T score computation (i.e., the
//           interaction of one Q-vector with the D_s K-vectors in K_tile by its
//           assigned SIMD group) is a critical path for improving Pass 1 performance.
//           Techniques like K-micro-tiling will be explored after end-to-end correctness
//           with Pass 2 is established.
//
//    - Current State for Correctness Testing:
//        - Pass 1 is now using the full `dot_product_qk` for score computation.
//        - Pass 2 is fully implemented.
//        - End-to-end numerical correctness testing against a Python reference is underway.
//        - Current performance (2025-05-29) with full P1 compute + full P2:
//          "cpp_pal_paged_attention_prefill": { "4096.0": 1001.0590 } ms. Slope ~S^1.64.
//          This highlights that while the online softmax logic is now more correct,
//          the QK^T compute cost and underlying S^2 Q-read traffic remain major
//          performance factors to address after correctness is fully validated.
// =================================================================================================
