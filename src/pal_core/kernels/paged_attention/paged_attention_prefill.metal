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


[[kernel]] void get_device_info() {
    // used for fetching a metal compute pipeline state
    // for the current device to get the max threads per group
    // and simd group size
}


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
    const uint chunked_head_dim_size = params.head_dim / 4; // For K/V (half4)

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
        for (uint chunk_idx = local_idx_in_tg; chunk_idx < chunked_head_dim_size; chunk_idx += threads_per_tg) {
            dst_k_vector_h4_ptr[chunk_idx] = src_k_vector_h4_ptr[chunk_idx];
        }
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);

    // Load V-tile
    for (uint token_idx_on_page = 0; token_idx_on_page < D_s; ++token_idx_on_page) {
        ulong v_global_offset = page_base_offset_global_kv + // Same base as K
                                (ulong)token_idx_on_page * (ulong)params.num_kv_heads * (ulong)params.head_dim +
                                (ulong)assigned_global_kv_head_idx * (ulong)params.head_dim;
        device const half* v_src_ptr = v_cache_pool_in + v_global_offset;
        threadgroup half* v_dst_ptr = V_tile + token_idx_on_page * params.head_dim;
        threadgroup half4* dst_v_vector_h4_ptr = (threadgroup half4*)(v_dst_ptr);
        device const half4* src_v_vector_h4_ptr = (device const half4*)(v_src_ptr);
        for (uint chunk_idx = local_idx_in_tg; chunk_idx < chunked_head_dim_size; chunk_idx += threads_per_tg) {
            dst_v_vector_h4_ptr[chunk_idx] = src_v_vector_h4_ptr[chunk_idx];
        }
    }
    simdgroup_barrier(mem_flags::mem_threadgroup); // K_tile and V_tile (depth D_s) are now loaded.

    // E. Query Block Iteration and Loading
    uint seq_len_for_this_batch_item = (uint)sequence_lengths_in[assigned_batch_item_idx];
    // Iterate over the Q-BLOCKs for this batch item (process D_s tokens (params.tokens_per_page) at once)
    for (uint q_block_start_local_idx = 0;
        q_block_start_local_idx < seq_len_for_this_batch_item;
        q_block_start_local_idx += D_s /* params.tokens_per_page is D_s */) {

        uint num_queries_in_this_block = min(D_s, seq_len_for_this_batch_item - q_block_start_local_idx);

        // Calculate the number of SIMD groups per GQA stream
        const uint simd_groups_per_gqa_stream = total_simd_groups_in_tg_metal / N_q_per_kv;

        uint gqa_stream_idx_for_this_simd_group = simd_group_id / simd_groups_per_gqa_stream;
        uint sub_simd_group_idx_within_stream = simd_group_id % simd_groups_per_gqa_stream;

        if (gqa_stream_idx_for_this_simd_group < N_q_per_kv) { // Check if this SIMD group is part of an active GQA stream

            uint target_q_head_local_offset_in_gqa_group = gqa_stream_idx_for_this_simd_group;
            uint target_global_q_head_idx = (assigned_global_kv_head_idx * N_q_per_kv) + target_q_head_local_offset_in_gqa_group;

            // Check if the target query head exists (important for MQA where num_q_heads < num_kv_heads)
            if (target_global_q_head_idx >= params.num_q_heads) {
                continue; // Skip this SIMD group as it maps to a non-existent query head
            }

            // Base pointer in Q_shmem_base for this GQA stream's entire block of D_s Q-vectors
            threadgroup float* q_block_shmem_for_gqa_stream = Q_shmem_base +
                                    (gqa_stream_idx_for_this_simd_group * D_s * params.head_dim);

            // Iterate through the Q-vectors that *this specific SIMD group* (within its GQA stream's assigned SIMD groups) is responsible for loading.
            for (uint q_idx_in_block_for_this_sg = sub_simd_group_idx_within_stream;
                q_idx_in_block_for_this_sg < num_queries_in_this_block;
                q_idx_in_block_for_this_sg += simd_groups_per_gqa_stream) {

                uint current_query_local_idx = q_block_start_local_idx + q_idx_in_block_for_this_sg;
                uint master_query_idx = query_starts_for_batch_item_arr[assigned_batch_item_idx] + current_query_local_idx;

                // Global source pointer for this specific Q-vector
                // Queries are in format [TotalQueries, NumQHeads, HeadDim]
                device const half* q_src_global_ptr = queries_in +
                    (master_query_idx * params.num_q_heads * params.head_dim) +
                    (target_global_q_head_idx * params.head_dim);

                // Destination in Q_shmem for this specific Q-vector
                threadgroup float* q_dest_specific_q_in_shmem = q_block_shmem_for_gqa_stream +
                                                                (q_idx_in_block_for_this_sg * params.head_dim);

                // The actual_simd_width lanes of THIS SIMD group load this ONE Q-vector
                for (uint c_idx = simd_lane_id; c_idx < chunked_head_dim_size; c_idx += actual_simd_width) {
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
        // --- END: Zero V_Sum_Accumulators_Area ---

        // F. Compute QK^T
        // Skip computation for inactive SIMD groups
        if (gqa_stream_idx_for_this_simd_group >= N_q_per_kv) {
            continue; // Skip to next Q-block iteration
        }

        threadgroup float* q_block_shmem_for_this_gqa_stream_for_compute = Q_shmem_base +
                                    (gqa_stream_idx_for_this_simd_group * D_s * params.head_dim);

        // Proceed with "Row Strip" compute for this SIMD group.
        // It forms the core of the QK^T computation and subsequent V-accumulation.
        for (uint q_idx_in_block_for_this_sg = sub_simd_group_idx_within_stream;
              q_idx_in_block_for_this_sg < num_queries_in_this_block;
              q_idx_in_block_for_this_sg += simd_groups_per_gqa_stream
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

            // Check if the target query head exists (important for MQA where num_q_heads < num_kv_heads)
            if (target_global_q_head_idx >= params.num_q_heads) {
                continue; // Skip this query as it maps to a non-existent query head
            }

            // F.2.a: Get pointer to the current Q-vector in Q_shmem_block.
            threadgroup const float* q_vec_ptr = q_block_shmem_for_this_gqa_stream_for_compute +
                                                 (q_idx_in_block_for_this_sg * params.head_dim);
            // F.2.b: Get pointer to this SIMD group's dedicated V-sum accumulator.
            threadgroup float* v_sum_accumulator_ptr = V_Sum_Accumulators_Area +
                                                       (simd_group_id * params.head_dim);

            // Zero the V accumulator for this specific query
            threadgroup float4* v_acc_f4_ptr = (threadgroup float4*)v_sum_accumulator_ptr;
            for (uint f4_idx = simd_lane_id; f4_idx < chunked_head_dim_size; f4_idx += actual_simd_width) {
                v_acc_f4_ptr[f4_idx] = float4(0.0f);
            }

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
                for (uint f4_chunk_idx = simd_lane_id; f4_chunk_idx < chunked_head_dim_size; f4_chunk_idx += actual_simd_width) {
                    uint d_offset = f4_chunk_idx * 4;
                    float4 qv = *((threadgroup const float4*)(q_vec_ptr + d_offset));
                    float4 kv = float4(*((threadgroup const half4*)(k_vec_hist_ptr + d_offset)));
                    per_lane_partial_score += dot(qv, kv);
                }
                float score = simd_sum(per_lane_partial_score);
                score = simd_broadcast_first(score);

                // F.3.d. Online Softmax Update
                float old_page_max_score_val = page_max_score;
                page_max_score = max(page_max_score, score);

                // Term for the current score, normalized by the new page_max_score.
                float current_score_exp_contribution;

                // If page_max_score changed, we need to rescale the existing sum and its Kahan compensator.
                if (page_max_score > old_page_max_score_val && old_page_max_score_val != -INFINITY) {
                    float rescale_exp_arg = max(old_page_max_score_val - page_max_score, params.log_exp_min_clamp);
                    float actual_scale_factor = fast::exp(rescale_exp_arg);

                    page_sum_exp_norm_by_page_max *= actual_scale_factor;
                    kahan_c_for_sum_exp *= actual_scale_factor; // Rescale Kahan compensation term

                    // Rescale the existing sum and its Kahan compensator.
                    for (uint h_rescale_idx = simd_lane_id;
                        h_rescale_idx < chunked_head_dim_size;
                        h_rescale_idx += actual_simd_width)
                    {
                        uint d = h_rescale_idx * 4;
                        *((threadgroup float4*)(v_sum_accumulator_ptr + d)) *= actual_scale_factor;
                    }
                }

                // Calculate the exponential of (current score - new_page_max_score)
                float current_term_exp_arg = max(score - page_max_score, params.log_exp_min_clamp);
                current_score_exp_contribution = fast::exp(current_term_exp_arg);

                // Kahan summation to add current_score_exp_contribution to page_sum_exp_norm_by_page_max
                float y_kahan = current_score_exp_contribution - kahan_c_for_sum_exp;
                float t_kahan = page_sum_exp_norm_by_page_max + y_kahan;
                kahan_c_for_sum_exp = (t_kahan - page_sum_exp_norm_by_page_max) - y_kahan;
                page_sum_exp_norm_by_page_max = t_kahan;
                // --- END F.3.d: Online Softmax Update ---

                // F.3.e. V-Aggregation
                if (current_score_exp_contribution < kEpsilonForZeroGuard) {
                    // Skip this V-vector if the current score's exp contribution is effectively zero.
                    continue;
                }

                threadgroup const half* v_vec_hist_ptr = V_tile + (k_idx_in_tile * params.head_dim);
                // Each lane (simd_lane_id) processes different chunks of the head_dim.
                for (uint h_chunk_idx = simd_lane_id;
                      h_chunk_idx < chunked_head_dim_size;
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
                h_chunk_idx < chunked_head_dim_size;
                h_chunk_idx += actual_simd_width) {

                uint h_dim_offset = h_chunk_idx * 4;
                float4 val_f4 = *((threadgroup float4*)(v_sum_accumulator_ptr + h_dim_offset));
                *((device half4*)(o_dest_ptr + h_dim_offset)) = half4(val_f4);
            }
            // all lanes finished their chunk
        } // end of query processing loop
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
    // Number of SIMD groups in this TG
    const uint NumSIMDgroups_Pass2 = tg_dim.x / actual_simd_width;
    const uint simd_group_id = local_idx_in_tg / actual_simd_width;
    const uint simd_lane_id = local_idx_in_tg % actual_simd_width;

    // Base for TGMem carving
    threadgroup uchar* tg_mem_base_byte_ptr = (threadgroup uchar*)tg_mem;
    uintptr_t current_tg_offset = 0;

     // Helper for alignment
    auto align_offset = [] (uintptr_t offset, size_t alignment) {
        return (offset + alignment - 1) & ~(alignment - 1);
    };
    const size_t float_alignment = alignof(float);
    const size_t float4_alignment = alignof(float4);

    // 1. M_item_shared_scalar (single float)
    current_tg_offset = align_offset(current_tg_offset, float_alignment);
    threadgroup float* M_item_shared_scalar = (threadgroup float*)(tg_mem_base_byte_ptr + current_tg_offset);
    current_tg_offset += sizeof(float);

    // 2. S_item_shared_scalar (single float)
    current_tg_offset = align_offset(current_tg_offset, float_alignment);
    threadgroup float* S_item_shared_scalar = (threadgroup float*)(tg_mem_base_byte_ptr + current_tg_offset);
    current_tg_offset += sizeof(float);

    // 3. S_item_kahan_c_shared (single float)
    current_tg_offset = align_offset(current_tg_offset, float_alignment);
    threadgroup float* S_item_kahan_c_shared = (threadgroup float*)(tg_mem_base_byte_ptr + current_tg_offset);
    current_tg_offset += sizeof(float);

    // 4. O_item_shared_accumulator[params.head_dim] (float array)
    //    Align to float4_alignment because it's often accessed as float4
    current_tg_offset = align_offset(current_tg_offset, float4_alignment);
    threadgroup float* O_item_shared_accumulator = (threadgroup float*)(tg_mem_base_byte_ptr + current_tg_offset);
    current_tg_offset += params.head_dim * sizeof(float);

    // 5. simdgroup_m_scratch[NumSIMDgroups_Pass2] (float array)
    current_tg_offset = align_offset(current_tg_offset, float_alignment); // Align base of this array
    threadgroup float* simdgroup_m_scratch = (threadgroup float*)(tg_mem_base_byte_ptr + current_tg_offset);
    current_tg_offset += NumSIMDgroups_Pass2 * sizeof(float);

    // 6. simdgroup_s_scratch[NumSIMDgroups_Pass2] (float array)
    current_tg_offset = align_offset(current_tg_offset, float_alignment); // Align base
    threadgroup float* simdgroup_s_scratch = (threadgroup float*)(tg_mem_base_byte_ptr + current_tg_offset);
    current_tg_offset += NumSIMDgroups_Pass2 * sizeof(float);

    // 7. simdgroup_o_partials[NumSIMDgroups_Pass2][params.head_dim] (float 2D array)
    //    Align base to float4_alignment as it will be accessed in chunks
    current_tg_offset = align_offset(current_tg_offset, float4_alignment);
    threadgroup float* simdgroup_o_partials = (threadgroup float*)(tg_mem_base_byte_ptr + current_tg_offset);
    // --- End of TGMem Carving ---

    // --- Outer Item Loop (Serial within TG) ---
    const uint items_in_token_dim = params.pass2_token_block_size;
    const uint items_in_qhead_dim = params.pass2_qhead_block_size;

    const uint items_in_flat_dim = items_in_token_dim * items_in_qhead_dim;

    // A single loop that iterates through all items in the TG's 2D block.
    // Each iteration of this loop processes ONE (QueryToken, QHead) item cooperatively by the whole TG.
    for (uint item_flat_idx_in_block = 0; item_flat_idx_in_block < items_in_flat_dim; ++item_flat_idx_in_block) {
        // Map the flat item index back to 2D local indices within this TG's block
        uint local_token_idx_in_block = item_flat_idx_in_block % items_in_token_dim;
        uint local_q_head_idx_in_block = item_flat_idx_in_block / items_in_token_dim;

        // Calculate the absolute global master_query_idx and target_global_q_head_idx for the current item
        uint current_master_query_idx = (tg_pos_in_grid.x * items_in_token_dim) + local_token_idx_in_block;
        uint current_target_q_head_idx = (tg_pos_in_grid.y * items_in_qhead_dim) + local_q_head_idx_in_block;

        // Boundary Check: Ensure this item is within the actual data dimensions
        if (current_master_query_idx >= params.query_token_count_total ||
            current_target_q_head_idx >= params.num_q_heads) {
            continue;
        }

        // Single thread initializes the scalar shared values
        if (local_idx_in_tg == 0) {
            *M_item_shared_scalar = -INFINITY;
            *S_item_shared_scalar = 0.0f;
            *S_item_kahan_c_shared = 0.0f;
        }

        // All threads in the TG cooperatively zero out the shared array accumulators.
        // Each thread zeros a slice of the array.

        // Zero O_item_shared_accumulator[params.head_dim]
        for (uint i = local_idx_in_tg; i < params.head_dim; i += tg_dim.x) {
            O_item_shared_accumulator[i] = 0.0f;
        }

        // Zero simdgroup_m_scratch[NumSIMDgroups_Pass2]
        for (uint i = local_idx_in_tg; i < NumSIMDgroups_Pass2; i += tg_dim.x) {
            simdgroup_m_scratch[i] = -INFINITY; // Initialize max accumulators to -INF
        }

        // Zero simdgroup_s_scratch[NumSIMDgroups_Pass2]
        for (uint i = local_idx_in_tg; i < NumSIMDgroups_Pass2; i += tg_dim.x) {
            simdgroup_s_scratch[i] = 0.0f;
        }

        // Zero simdgroup_o_partials[NumSIMDgroups_Pass2][params.head_dim]
        // Total elements = NumSIMDgroups_Pass2 * params.head_dim
        for (uint i = local_idx_in_tg; i < (NumSIMDgroups_Pass2 * params.head_dim); i += tg_dim.x) {
            simdgroup_o_partials[i] = 0.0f;
        }

        // Barrier to ensure all initializations are complete
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // --- Phase A: Compute M_global ---
        float m_thread_private_max = -INFINITY;
        for (uint page_idx = local_idx_in_tg; page_idx < params.num_active_batch_logical_pages; page_idx += tg_dim.x) {
            ulong m_s_stride_qhead_dim = params.num_active_batch_logical_pages;
            ulong m_s_stride_query_dim = params.num_q_heads * m_s_stride_qhead_dim;
            ulong flat_idx_m_value = (ulong)current_master_query_idx * m_s_stride_query_dim +
                                     (ulong)current_target_q_head_idx * m_s_stride_qhead_dim +
                                     page_idx;

            float m_value_from_page = m_pass1_results[flat_idx_m_value];
            m_thread_private_max = max(m_thread_private_max, m_value_from_page);
        }

        float m_simdgroup_max = simd_max(m_thread_private_max);

        if (simd_lane_id == 0) {
            simdgroup_m_scratch[simd_group_id] = m_simdgroup_max;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (simd_group_id == 0) {
            float final_max_for_item_local_to_sg0 = -INFINITY;
            // Lanes of SIMDgroup 0 cooperatively reduce simdgroup_m_scratch
            for (uint i = simd_lane_id; i < NumSIMDgroups_Pass2; i += actual_simd_width) {
                final_max_for_item_local_to_sg0 = max(final_max_for_item_local_to_sg0, simdgroup_m_scratch[i]);
            }
            // Reduce among lanes of SIMDgroup 0
            float final_max_for_item_reduced_in_sg0 = simd_max(final_max_for_item_local_to_sg0);

            if (simd_lane_id == 0) { // Lane 0 of SIMDgroup 0 writes the final result
                *M_item_shared_scalar = final_max_for_item_reduced_in_sg0;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // --- End of Phase A ---

        // --- Phase B: Cooperative S_global & O_final Accumulation ---
        float M_final_item_val = *M_item_shared_scalar;

        // Each SIMDgroup initializes its private sum for S and Kahan compensator (in registers)
        float s_sg_private_sum = 0.0f;

        threadgroup float* my_sg_o_partial_accumulator = simdgroup_o_partials +
                                                         (simd_group_id * params.head_dim);

        // Each SIMDgroup 'simd_group_id' processes a slice of pages.
        // All lanes within that SIMDgroup will cooperate on each of these pages.
        uint pages_per_sg_base = params.num_active_batch_logical_pages / NumSIMDgroups_Pass2;
        uint pages_remainder_sg = params.num_active_batch_logical_pages % NumSIMDgroups_Pass2;
        uint start_page_for_this_sg = simd_group_id * pages_per_sg_base + min(simd_group_id, pages_remainder_sg);
        uint end_page_for_this_sg = start_page_for_this_sg + pages_per_sg_base + (simd_group_id < pages_remainder_sg ? 1 : 0);

        for (uint page_idx = start_page_for_this_sg; page_idx < end_page_for_this_sg; ++page_idx) {
            float m_local_p_this_page;
            float s_local_p_this_page;
            float rescale_factor_this_page;

            if (simd_lane_id == 0) {
                // Lane 0 calculates the flat index for m_pass1_results and s_pass1_results
                ulong m_s_stride_qhead_dim = params.num_active_batch_logical_pages;
                ulong m_s_stride_query_dim = params.num_q_heads * m_s_stride_qhead_dim;
                ulong flat_idx_ms_value = (ulong)current_master_query_idx * m_s_stride_query_dim +
                                        (ulong)current_target_q_head_idx * m_s_stride_qhead_dim +
                                        page_idx; // Use the 'page_idx' from the loop

                m_local_p_this_page = m_pass1_results[flat_idx_ms_value];
                s_local_p_this_page = s_pass1_results[flat_idx_ms_value];

                rescale_factor_this_page = precise::exp(max(m_local_p_this_page - M_final_item_val,
                                                            params.log_exp_min_clamp));
            }

            // Broadcast the values from lane 0 to all other lanes in the SIMDgroup
            s_local_p_this_page = simd_broadcast_first(s_local_p_this_page);
            rescale_factor_this_page = simd_broadcast_first(rescale_factor_this_page);

            if (simd_lane_id == 0) {
                s_sg_private_sum += s_local_p_this_page * rescale_factor_this_page;
            }

            if (rescale_factor_this_page >= kEpsilonForZeroGuard) {
                // 1. Calculate base pointer to o_partial_p for the current page in global memory
                ulong o_stride_page_dim = params.head_dim;
                ulong o_stride_qhead_dim = params.num_active_batch_logical_pages * o_stride_page_dim;
                ulong o_stride_query_dim = params.num_q_heads * o_stride_qhead_dim;
                ulong base_offset_o_global = (ulong)current_master_query_idx * o_stride_query_dim +
                                            (ulong)current_target_q_head_idx * o_stride_qhead_dim +
                                            (ulong)page_idx * o_stride_page_dim;
                device const half* o_partial_p_global_ptr = o_pass1_results + base_offset_o_global;

                // 2. Lanes of this SIMDgroup cooperatively load, scale, and accumulate
                //    Each lane 'simd_lane_id' handles a slice of params.head_dim.
                for (uint h_offset_in_head = simd_lane_id;
                    h_offset_in_head < params.head_dim;
                    h_offset_in_head += actual_simd_width) {

                    // Load one 'half' component from global o_partial_p
                    float o_val_from_page_float = (float)o_partial_p_global_ptr[h_offset_in_head];

                    // Scale and accumulate into this SIMDgroup's TGMem slice
                    // using FMA for precision and potential performance.
                    my_sg_o_partial_accumulator[h_offset_in_head] =
                        fma(o_val_from_page_float,
                            rescale_factor_this_page,
                            my_sg_o_partial_accumulator[h_offset_in_head]);
                }
            } // End if (rescale_factor_this_page >= kEpsilonForZeroGuard)
        }
        // --- After this loop, this SIMDgroup 'simd_group_id' has processed all its assigned pages. ---
        if (simd_lane_id == 0) {
            simdgroup_s_scratch[simd_group_id] = s_sg_private_sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // --- End of Phase B ---

        if ((simd_group_id == 0) && (simd_lane_id == 0)) { // Only lane 0 of SIMDgroup 0 performs the serial Kahan sum
            for (uint i = 0; i < NumSIMDgroups_Pass2; ++i) {
                float s_contrib_from_sg_scratch = simdgroup_s_scratch[i];

                // Apply Kahan summation to *S_item_shared_scalar
                float y_kahan = s_contrib_from_sg_scratch - (*S_item_kahan_c_shared);
                float t_kahan = (*S_item_shared_scalar) + y_kahan;
                *S_item_kahan_c_shared = (t_kahan - (*S_item_shared_scalar)) - y_kahan;
                *S_item_shared_scalar = t_kahan;
            }
        }

        for (uint h_target = local_idx_in_tg; // Each thread starts at its own ID as an index into HeadDim
              h_target < params.head_dim;
              h_target += tg_dim.x) { // Strides by total threads in TG

            float sum_for_this_h_component = 0.0f; // Private register accumulator for this thread for this h_target

            // This thread now sums contributions for its h_target across all SIMDgroups' partials
            for (uint sg_idx = 0; sg_idx < NumSIMDgroups_Pass2; ++sg_idx) {
                // Accessing simdgroup_o_partials[sg_idx][h_target]
                // Layout: simdgroup_o_partials is [NumSIMDgroups_Pass2 * params.head_dim]
                sum_for_this_h_component += simdgroup_o_partials[sg_idx * params.head_dim + h_target];
            }

            // Write the final sum for this h_target to the shared TGMem accumulator
            O_item_shared_accumulator[h_target] = sum_for_this_h_component;
        }
        // --- End of Phase C ---
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // --- Phase D: Normalization and Write ---
        float S_final_val = *S_item_shared_scalar;
        float inv_S_final = 0.0f;

        if (S_final_val > kSmallDenominatorThreshold) {
            inv_S_final = 1.0f / S_final_val;
        }

        ulong output_item_flat_idx = (ulong)current_master_query_idx * params.num_q_heads + current_target_q_head_idx;
        ulong base_offset_global_output = output_item_flat_idx * params.head_dim;
        device half* final_out_ptr_for_item_base = final_output_buffer + base_offset_global_output;

        const uint elements_per_vector_write = 4; // for half4
        uint num_vector_writes = params.head_dim / elements_per_vector_write;

        for (uint vec_idx = local_idx_in_tg;
              vec_idx < num_vector_writes;
              vec_idx += tg_dim.x) {

            uint h_start_idx = vec_idx * elements_per_vector_write;

            // Read 4 float components from O_item_shared_accumulator
            float4 o_chunk_float = float4(O_item_shared_accumulator[h_start_idx + 0],
                                          O_item_shared_accumulator[h_start_idx + 1],
                                          O_item_shared_accumulator[h_start_idx + 2],
                                          O_item_shared_accumulator[h_start_idx + 3]);

            // Normalize
            o_chunk_float *= inv_S_final;

            // Convert to half4
            half4 o_chunk_half = half4(o_chunk_float);

            // Write to global memory
            device half4* dest_ptr_h4 = (device half4*)(final_out_ptr_for_item_base + h_start_idx);
            *dest_ptr_h4 = o_chunk_half;
        }

        // Barrier before the TG proceeds to the next item_flat_idx_in_block.
        // Ensures all global writes for the current item are done, and TGMem is safe for reuse.
        threadgroup_barrier(mem_flags::mem_threadgroup);
    } // end of item_flat_idx_in_block loop

} // End of kernel
