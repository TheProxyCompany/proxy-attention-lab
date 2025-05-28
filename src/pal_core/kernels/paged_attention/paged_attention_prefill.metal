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
    current_offset = (current_offset + kAlignmentMask) & ~kAlignmentMask;
    threadgroup float* V_Sum_Accumulators_Area = (threadgroup float*)(tg_mem_base_byte_ptr + current_offset);
    current_offset += N_q_per_kv * params.head_dim * sizeof(float); // Space for N_q_per_kv accumulators

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
                // uint current_q_logical_pos = (uint)query_token_offset_in[master_query_idx]; // For compute later

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
        uint total_floats_in_v_acc_area = N_q_per_kv * params.head_dim;
        uint floats_to_zero_per_thread4 = total_floats_in_v_acc_area / 4;

        // Vectorized zeroing using float4
        threadgroup float4* v_acc_area_f4_ptr = (threadgroup float4*)V_Sum_Accumulators_Area;
        for (uint i = local_idx_in_tg; i < floats_to_zero_per_thread4; i += threads_per_tg) {
            v_acc_area_f4_ptr[i] = float4(0.0f);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // --- END: Zero V_Sum_Accumulators_Area ---

        // F. Compute QK^T
        // Iterate through each query vector within the loaded Q-block
        for (uint q_idx_in_block = 0; q_idx_in_block < num_queries_in_this_block; ++q_idx_in_block) {
            if (gqa_stream_idx_for_this_simd_group >= N_q_per_kv) {
                // This SIMD group is outside of the active GQA stream
                continue;
            }

            // F.1. Per-Query Setup
            uint current_query_local_idx = q_block_start_local_idx + q_idx_in_block;
            uint master_query_idx = query_starts_for_batch_item_arr[assigned_batch_item_idx] + current_query_local_idx;
            // This is used for causal masking: a query at position P can only attend to keys at positions < P.
            uint current_q_logical_pos = (uint)query_token_offset_in[master_query_idx];

            // F.2.a Per-SIMD-Group Setup
            uint target_q_head_local_offset_in_gqa_group = gqa_stream_idx_for_this_simd_group; // Already calculated to enter this block
            uint target_global_q_head_idx = (assigned_global_kv_head_idx * N_q_per_kv) + target_q_head_local_offset_in_gqa_group;

            // Pointer to the current Q-vector in Q_shmem_base for this q_idx_in_block and this SIMD group's GQA stream
            threadgroup const float* q_vec_ptr = Q_shmem_base +
                                                    (target_q_head_local_offset_in_gqa_group * D_s * params.head_dim) +
                                                    (q_idx_in_block * params.head_dim);

            // Initialize accumulator for this Q-instance (query, q_head) processing this KV-page
            threadgroup float* v_sum_accumulator_ptr = V_Sum_Accumulators_Area +
                                           (target_q_head_local_offset_in_gqa_group * params.head_dim);

            // F.3. Loop through K/V vectors in the loaded K_tile / V_tile
            for (uint k_idx_in_tile = 0; k_idx_in_tile < D_s; ++k_idx_in_tile) {
                // F.3.a. Per-K/V vector setup
                threadgroup const half* k_vec_hist_ptr = K_tile + (k_idx_in_tile * params.head_dim);
                threadgroup const half* v_vec_hist_ptr = V_tile + (k_idx_in_tile * params.head_dim);

                // Determine the logical position of this history token in its original sequence.
                uint history_token_logical_pos = (assigned_logical_page_idx_in_sequence * D_s) + k_idx_in_tile;

                // F.3.b. Causal Masking / Length Check
                if (history_token_logical_pos >= current_q_logical_pos ||
                    history_token_logical_pos >= seq_len_for_this_batch_item) {
                    continue; // Skip this history token
                }

                // F.3.c. QK^T dot product
                float score = dot_product_qk(q_vec_ptr, k_vec_hist_ptr, params);

                // F.3.d. Online Softmax Update (will go here)
                // - (Comment: Content to come: update page_max_score, page_sum_exp)

                // F.3.e. V-Aggregation (will go here)
                // - (Comment: Content to come: v_sum_accumulator += weight * V_vector)

            } // End loop k_idx_in_tile (history within page)
            // F.4. Write results for this (master_query_idx, target_global_q_head_idx, flat_work_item_idx)

        } // End loop q_idx_in_block (queries within Q-block)

    } // end of q_block_start_local_idx loop
} // End of kernel
