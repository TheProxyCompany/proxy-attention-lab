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
 * Pass 1 of the page-centric prefill architecture.
 */
[[kernel]] void paged_attn_prefill_pass1_kernel(
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
                if (history_token_logical_pos > current_q_logical_pos ||
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


/**
 * paged_attn_prefill_kernel
 * Pass 2 of the page-centric prefill architecture.
 */
[[kernel]] void paged_attn_prefill_pass2_kernel(
    // Pass 1 output buffers
    device      const float* m_pass1_results        [[buffer(0)]],  // Local max scores per page
    device      const float* s_pass1_results        [[buffer(1)]],  // Local sum-exponentials per page
    device      const half*  o_pass1_results        [[buffer(2)]],  // Unnormalized partial V-accumulations
    // Active work items buffer
    device      const uint2* active_work_item_pairs [[buffer(3)]],  // Active (batch_item, logical_page) pairs
    // Parameters
    constant    const PagedAttentionParams& params  [[buffer(4)]],
    // Final output buffer
    device      half* final_output_buffer           [[buffer(5)]],
    // Thread/grid identifiers
    uint actual_simd_width                          [[threads_per_simdgroup]],
    threadgroup float* tg_mem                       [[threadgroup(0)]],
    uint3       tg_pos_in_grid                      [[threadgroup_position_in_grid]],
    uint3       tg_dim                              [[threads_per_threadgroup]],
    uint        local_idx_in_tg                     [[thread_index_in_threadgroup]]
) {
    // --- TGMem Carving for Pass 2 ---
    threadgroup uchar* tg_mem_base_byte_ptr = (threadgroup uchar*)tg_mem;
    uintptr_t current_tg_offset = 0;

    current_tg_offset = (current_tg_offset + kAlignmentMask) & ~kAlignmentMask;
    threadgroup float* O_final_accumulators_base_tg = (threadgroup float*)(tg_mem_base_byte_ptr + current_tg_offset);
    // Each thread gets its own slice of this O_final_accumulators_base_tg
    threadgroup float* O_final_for_this_thread_tg = O_final_accumulators_base_tg + (local_idx_in_tg * params.head_dim);

    // --- 1. Determine the (Query Token Block, Q-Head Block) this TG is responsible for ---
    uint query_token_block_offset = tg_pos_in_grid.x * params.pass2_token_block_size;
    uint q_head_block_offset = tg_pos_in_grid.y * params.pass2_qhead_block_size;

    // --- 2. Each thread iterates over the output items assigned to it within the TG's block ---
    // N_outputs_per_TG is the number of (QueryToken, QHead) pairs this TG processes.
    uint num_outputs_in_tg_block = params.pass2_token_block_size * params.pass2_qhead_block_size;
    uint threads_in_tg_p2 = tg_dim.x; // Total threads in this Pass 2 TG

    for (uint item_idx_processed_by_thread = local_idx_in_tg;
          item_idx_processed_by_thread < num_outputs_in_tg_block;
          item_idx_processed_by_thread += threads_in_tg_p2) {

        // Map the flat item_idx_processed_by_thread to 2D local indices within the block
        uint local_token_idx_in_block = item_idx_processed_by_thread % params.pass2_token_block_size;
        uint local_q_head_idx_in_block = item_idx_processed_by_thread / params.pass2_token_block_size; // Integer division

        // Calculate the absolute master_query_idx and target_global_q_head_idx for this item
        uint master_query_idx = query_token_block_offset + local_token_idx_in_block;
        uint target_global_q_head_idx = q_head_block_offset + local_q_head_idx_in_block;

        // Boundary checks for the specific item this thread is processing
        if (master_query_idx >= params.query_token_count_total || target_global_q_head_idx >= params.num_q_heads) {
            continue;
        }

        // --- 3. Initialize and find M_global for this (master_query_idx, target_global_q_head_idx) item ---
        float M_global_for_this_item = -INFINITY;

        // Loop over all pages that Pass 1 processed.
        // params.num_active_batch_logical_pages is the size of the third dimension of m_pass1_results.
        for (uint page_idx = 0; page_idx < params.num_active_batch_logical_pages; ++page_idx) {
            // Calculate flat index into m_pass1_results.
            // Layout of m_pass1_results: [TotalQueryTokens, NumQHeads, NumActivePages]
            ulong m_s_stride_qhead = params.num_active_batch_logical_pages;
            ulong m_s_stride_query = params.num_q_heads * params.num_active_batch_logical_pages;

            ulong flat_idx_m_pass1 = (ulong)master_query_idx * m_s_stride_query +
                                     (ulong)target_global_q_head_idx * m_s_stride_qhead +
                                     page_idx;

            float current_page_m_local = m_pass1_results[flat_idx_m_pass1];
            M_global_for_this_item = max(M_global_for_this_item, current_page_m_local);
        }

        // --- Step 4: Calculate S_global_for_this_item ---
        float S_global_for_this_item = 0.0f;
        float kahan_c_for_S_global = 0.0f; // Kahan compensation term for S_global

        // fuse with above for loop eventually.
        for (uint page_idx = 0; page_idx < params.num_active_batch_logical_pages; ++page_idx) {
                        // Layout: [TotalQueryTokens, NumQHeads, NumActivePages]
            ulong m_s_stride_qhead = params.num_active_batch_logical_pages;
            ulong m_s_stride_query = params.num_q_heads * params.num_active_batch_logical_pages;

            ulong flat_idx_ms_pass1 = (ulong)master_query_idx * m_s_stride_query +
                                      (ulong)target_global_q_head_idx * m_s_stride_qhead +
                                      page_idx;

            float m_local_p = m_pass1_results[flat_idx_ms_pass1];
            float s_local_p = s_pass1_results[flat_idx_ms_pass1];

            // Calculate exp(m_local_p - M_global_for_this_item)
            // Clamp the argument to exp.
            float rescale_factor_exp_arg = (M_global_for_this_item == -INFINITY && m_local_p == -INFINITY) ?
                                           params.log_exp_min_clamp :
                                           max(m_local_p - M_global_for_this_item, params.log_exp_min_clamp);
            float rescale_factor = precise::exp(rescale_factor_exp_arg);

            // Contribution of this page to S_global
            float s_page_contribution = s_local_p * rescale_factor;

            // Kahan summation for S_global_for_this_item
            float y_kahan = s_page_contribution - kahan_c_for_S_global;
            float t_kahan = S_global_for_this_item + y_kahan;
            kahan_c_for_S_global = (t_kahan - S_global_for_this_item) - y_kahan;
            S_global_for_this_item = t_kahan;
        }

        // --- BEGIN Step 5: O_partial Aggregation & Normalization (using TGMem for O_final) ---

        // Initialize this thread's dedicated O_final accumulator in TGMem to zeros.
        // Each thread zeros its own HeadDim slice.
        for (uint h_idx_init = 0; h_idx_init < params.head_dim; h_idx_init += 4) {
             *((threadgroup float4*)(O_final_for_this_thread_tg + h_idx_init)) = float4(0.0f);
        }

        // Loop over all pages again for this item to aggregate o_partials
        for (uint page_idx = 0; page_idx < params.num_active_batch_logical_pages; ++page_idx) {
            // Calculate flat index for m_pass1_results
            ulong m_s_stride_qhead = params.num_active_batch_logical_pages;
            ulong m_s_stride_query = params.num_q_heads * params.num_active_batch_logical_pages;
            ulong flat_idx_m_pass1 = (ulong)master_query_idx * m_s_stride_query +
                                     (ulong)target_global_q_head_idx * m_s_stride_qhead +
                                     page_idx;
            float m_local_p = m_pass1_results[flat_idx_m_pass1];

            // Calculate rescale_factor: exp(m_local_p - M_global_for_this_item)
            float rescale_factor_exp_arg = (M_global_for_this_item == -INFINITY && m_local_p == -INFINITY) ?
                                           params.log_exp_min_clamp :
                                           max(m_local_p - M_global_for_this_item, params.log_exp_min_clamp);
            float rescale_factor = precise::exp(rescale_factor_exp_arg);

            // If rescale_factor is effectively zero, this page's o_partial won't contribute,
            // so we can skip the expensive HeadDim loop.
            if (rescale_factor < kEpsilonForZeroGuard) {
                continue;
            }

            // Calculate base offset for o_partials_pass1_out for this item and page
            // Layout: [TotalQueries, NumQHeads, NumActivePages, HeadDim]
            ulong o_stride_page = params.head_dim;
            ulong o_stride_qhead = params.num_active_batch_logical_pages * params.head_dim;
            ulong o_stride_query = params.num_q_heads * params.num_active_batch_logical_pages * params.head_dim;

            ulong base_offset_o_partial = (ulong)master_query_idx * o_stride_query +
                                          (ulong)target_global_q_head_idx * o_stride_qhead +
                                          (ulong)page_idx * o_stride_page;

            device const half* o_partial_p_ptr = o_pass1_results + base_offset_o_partial;

            // Aggregate the HeadDim components into this thread's TGMem O accumulator
            // This inner loop is done by THIS thread, operating on its O_final_for_this_thread_tg.
            for (uint h_idx = 0; h_idx < params.head_dim; h_idx += 4) {
                float4 o_partial_chunk_f = float4( *((device const half4*)(o_partial_p_ptr + h_idx)) );

                // Accumulate into this thread's TGMem accumulator
                *((threadgroup float4*)(O_final_for_this_thread_tg + h_idx)) += o_partial_chunk_f * rescale_factor;
            }
        }

        // Normalize the final O_vector (in TGMem) by S_global_for_this_item
        float inv_S_global = (S_global_for_this_item > kSmallDenominatorThreshold) ?
                              (1.0f / S_global_for_this_item) : 0.0f;

        for (uint h_idx = 0; h_idx < params.head_dim; h_idx += 4) {
            *((threadgroup float4*)(O_final_for_this_thread_tg + h_idx)) *= inv_S_global;
        }
        // --- END Step 5: O_partial Aggregation & Normalization ---

        // --- BEGIN Step 6: Write O_final_for_this_thread_tg to final_output_buffer ---
        // Calculate the flat index for this item in the 1D view of [TotalItems, HeadDim].
        ulong output_item_flat_idx = (ulong)master_query_idx * params.num_q_heads + target_global_q_head_idx;
        ulong base_offset_final_output = output_item_flat_idx * params.head_dim;

        device half* final_out_ptr_for_item = final_output_buffer + base_offset_final_output;

        // This thread writes its HeadDim float values (from O_final_for_this_thread_tg)
        // to global memory, converting to half.
        // This is a HeadDim-wide operation.
        for (uint h_idx = 0; h_idx < params.head_dim; h_idx += 4) { // Process in float4 chunks
            // Read the float4 chunk from this thread's TGMem accumulator
            float4 val_f4_to_write = *((threadgroup float4*)(O_final_for_this_thread_tg + h_idx));

            // Convert to half4 and write to global memory
            *((device half4*)(final_out_ptr_for_item + h_idx)) = half4(val_f4_to_write);
        }
        // --- END Step 6: Write O_final_for_this_thread_tg to final_output_buffer ---
    } // End loop over items assigned to this thread

} // End of kernel

[[kernel]] void get_device_info() {
    // used for fetching a metal compute pipeline state
    // for the current device to get the max threads per group
    // and simd group size
}
