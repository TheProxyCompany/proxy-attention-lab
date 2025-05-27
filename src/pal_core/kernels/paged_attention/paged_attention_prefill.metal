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
    constant    const PagedAttentionParams& params  [[buffer(7)]],      // Parameters
    device      const uint2* active_work_item_pairs [[buffer(8)]],      // Active (batch_item, logical_page) pairs
    device      float* m_locals_pass1_out           [[buffer(9)]],      // Local max scores
    device      float* s_locals_pass1_out           [[buffer(10)]],     // Local sum-exponentials
    device      half*  o_partials_pass1_out         [[buffer(11)]],     // Unnormalized partial V-accumulations
    uint actual_simd_width                          [[threads_per_simdgroup]],
    threadgroup float* tg_mem                       [[threadgroup(0)]],
    uint3       tg_pos_in_grid                      [[threadgroup_position_in_grid]],
    uint3       tg_dim                              [[threads_per_threadgroup]],
    uint        local_idx_in_tg                     [[thread_index_in_threadgroup]],
    uint        simd_lane_id                        [[thread_index_in_simdgroup]],
    uint        simd_group_id                       [[simdgroup_index_in_threadgroup]]
) {
    // A. TGMem Carving for K_tile and V_tile
    threadgroup uchar* tg_mem_base_byte_ptr = (threadgroup uchar*)tg_mem;
    uintptr_t current_offset = 0;

    // Q_shmem needs to hold N_q_per_kv Q-vectors, each of head_dim floats.
    // Example: N_q_per_kv=2, head_dim=128, sizeof(float)=4 => 2 * 128 * 4 = 1024 bytes for Q_shmem.
    const uint N_q_per_kv = (params.num_q_heads + params.num_kv_heads - 1) / params.num_kv_heads; // GQA Factor
    current_offset = (current_offset + kAlignmentMask) & ~kAlignmentMask;
    threadgroup float* Q_shmem_base = (threadgroup float*)(tg_mem_base_byte_ptr + current_offset);
    current_offset += N_q_per_kv * params.head_dim * sizeof(float);

    // K_tile allocation with alignment and padding
    current_offset = (current_offset + kAlignmentMask) & ~kAlignmentMask;
    threadgroup half* K_tile = (threadgroup half*)(tg_mem_base_byte_ptr + current_offset);
    current_offset += params.tokens_per_page * params.head_dim * sizeof(half);

    // V_tile allocation with alignment and padding
    current_offset = (current_offset + kAlignmentMask) & ~kAlignmentMask;
    threadgroup half* V_tile = (threadgroup half*)(tg_mem_base_byte_ptr + current_offset);
    current_offset += params.tokens_per_page * params.head_dim * sizeof(half);

    // B. Role Identification
    uint flat_work_item_idx = tg_pos_in_grid.x;

    // Safety check for work item bounds
    if (flat_work_item_idx >= params.num_active_batch_logical_pages) {
        return;
    }

    // Extract work item pair
    uint2 work_item_pair = active_work_item_pairs[flat_work_item_idx];
    uint assigned_batch_item_idx = work_item_pair.x;
    uint assigned_logical_page_idx_in_sequence = work_item_pair.y;
    uint assigned_global_kv_head_idx = tg_pos_in_grid.y;

    // C. Physical Page ID Lookup
    uint page_table_flat_idx = assigned_batch_item_idx * params.max_logical_blocks_per_seq + assigned_logical_page_idx_in_sequence;
    uint physical_page_id = page_table_in[page_table_flat_idx];

    // Safety check for physical page bounds
    if (physical_page_id >= params.num_physical_pages_in_pool) {
        return;
    }

    // D. Cooperative K/V Loading into TGMem
    // Constants for vectorized loading
    const uint threads_per_tg = tg_dim.x;
    const uint chunks_per_row = params.head_dim / 4; // Assuming head_dim is multiple of 4
    // Pre-compute page base offset
    const ulong page_base_offset = (ulong)physical_page_id *
                                   (ulong)params.tokens_per_page *
                                   (ulong)params.num_kv_heads *
                                   (ulong)params.head_dim;
    // Load K-tile cooperatively
    for (uint token_idx_on_page = 0; token_idx_on_page < params.tokens_per_page; token_idx_on_page++) {
        // Calculate global memory offset for K-vector
        ulong k_global_offset = page_base_offset +
                                (ulong)token_idx_on_page * (ulong)params.num_kv_heads * (ulong)params.head_dim +
                                (ulong)assigned_global_kv_head_idx * (ulong)params.head_dim;

        device const half* k_src_ptr = k_cache_pool_in + k_global_offset;
        threadgroup half* k_dst_ptr = K_tile + token_idx_on_page * params.head_dim;

        // Corrected cooperative loading for one K-vector
        threadgroup half4* dst_k_vector_h4_ptr = (threadgroup half4*)(k_dst_ptr);
        device const half4* src_k_vector_h4_ptr = (device const half4*)(k_src_ptr);

        // All threads_per_tg (tg_dim.x) participate in loading this *one* K-vector.
        // chunks_per_row is params.head_dim / 4.
        // local_idx_in_tg is the thread's index within the entire threadgroup.
        for (uint chunk_idx_in_k_vector = local_idx_in_tg;
             chunk_idx_in_k_vector < chunks_per_row;
             chunk_idx_in_k_vector += threads_per_tg) {
            dst_k_vector_h4_ptr[chunk_idx_in_k_vector] = src_k_vector_h4_ptr[chunk_idx_in_k_vector];
        }
    }

    // Synchronize after K-tile loading
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Load V-tile cooperatively
    for (uint token_idx_on_page = 0; token_idx_on_page < params.tokens_per_page; token_idx_on_page++) {
        // Calculate global memory offset for V-vector
        ulong v_global_offset = page_base_offset +
                                (ulong)token_idx_on_page * (ulong)params.num_kv_heads * (ulong)params.head_dim +
                                (ulong)assigned_global_kv_head_idx * (ulong)params.head_dim;

        device const half* v_src_ptr = v_cache_pool_in + v_global_offset;
        threadgroup half* v_dst_ptr = V_tile + token_idx_on_page * params.head_dim;

        // Corrected cooperative loading for one V-vector
        threadgroup half4* dst_v_vector_h4_ptr = (threadgroup half4*)(v_dst_ptr);
        device const half4* src_v_vector_h4_ptr = (device const half4*)(v_src_ptr);

        // All threads_per_tg (tg_dim.x) participate in loading this *one* V-vector.
        for (uint chunk_idx_in_v_vector = local_idx_in_tg;
             chunk_idx_in_v_vector < chunks_per_row;
             chunk_idx_in_v_vector += threads_per_tg) {
            dst_v_vector_h4_ptr[chunk_idx_in_v_vector] = src_v_vector_h4_ptr[chunk_idx_in_v_vector];
        }
    }

    // Synchronize after V-tile loading
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // E. Query Iteration Loop
    // Iterate through ALL query tokens provided in the input `queries_in`
    for (uint master_query_idx = 0; master_query_idx < params.query_token_count_total; ++master_query_idx) {
        // E1. Check if this query token belongs to the batch item this TG is processing
        if (query_to_seq_map_in[master_query_idx] != (int)assigned_batch_item_idx) {
            continue; // Skip this query, it's for a different batch item
        }
        // This query token belongs to our assigned batch item.
        uint current_q_logical_pos = (uint)query_token_offset_in[master_query_idx];

        // // E2. Intra-threadgroup parallelization of multiple Q heads
        // takes our scaling rate from 0.2 to 0.9 ☹️
        // if (simd_group_id < N_q_per_kv) {
        //     // Each SIMD group processes one Q head
        //     uint target_q_head_local_offset = simd_group_id;
        //     uint target_global_q_head_idx = (assigned_global_kv_head_idx * N_q_per_kv) + target_q_head_local_offset;
        //     threadgroup float* q_vec_tgmem_for_this_simd_group = Q_shmem_base + (simd_group_id * params.head_dim);

        //      // Load Q-vector for (master_query_idx, target_global_q_head_idx)
        //     device const half* q_src_ptr = queries_in + (master_query_idx * params.num_q_heads * params.head_dim) + (target_global_q_head_idx * params.head_dim);
        //     uint chunks_per_q_vector = params.head_dim / 4;
        //     for (uint c_idx = simd_lane_id; c_idx < chunks_per_q_vector; c_idx += actual_simd_width) {
        //         ((threadgroup float4*)q_vec_tgmem_for_this_simd_group)[c_idx] =
        //             float4(((device const half4*)q_src_ptr)[c_idx]) * params.inv_sqrt_head_dim;
        //     }
        //     simdgroup_barrier(mem_flags::mem_threadgroup); // Q for this SIMD group is ready

        // } // end of simd_group_id < N_q_per_kv

    } // end of query token loop
} // End of kernel
