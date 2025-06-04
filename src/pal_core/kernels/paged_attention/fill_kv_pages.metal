// fill_kv_pages.metal
// Metal shader implementation for filling KV cache pages.
//
// Copyright 2025 The Proxy Company. All Rights Reserved.
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
#include "paged_attention_types.h"

using namespace metal;

/**
 * fill_kv_pages_kernel - Scatters new K/V pairs into the global KV cache pools
 *
 * Each threadgroup processes a chunk of K/V pairs, with threads within the group
 * cooperatively copying data using vectorized operations.
 */
[[kernel]] void fill_kv_pages_kernel(
    device const    half*       new_keys [[buffer(0)]],
    device const    half*       new_values [[buffer(1)]],
    device const    uint*       page_table_data [[buffer(2)]],
    device const    int*        current_token_write_positions [[buffer(3)]],
    device const    uint*       query_to_seq_map_data [[buffer(4)]],
    constant const  FillKVPagesParams& params [[buffer(5)]],
    device          half*       global_k_pool_out [[buffer(6)]],
    device          half*       global_v_pool_out [[buffer(7)]],
    uint3           tg_id [[threadgroup_position_in_grid]], // tg_id.x is the chunk_idx
    uint3           thread_idx_in_group_3d [[thread_position_in_threadgroup]], // Use 3D for consistency
    uint3           threads_per_tg_3d [[threads_per_threadgroup]] // Use 3D for consistency
) {
    // Extract scalar values from 3D vectors
    uint thread_idx_in_group = thread_idx_in_group_3d.x;
    uint threads_per_tg_actual = threads_per_tg_3d.x;
    uint kv_pairs_per_threadgroup = params.kv_pairs_per_threadgroup;

    // Calculate starting token index for this threadgroup's chunk
    uint start_token_idx_for_chunk = tg_id.x * kv_pairs_per_threadgroup;

    // Calculate base pointers for destination in global pools
    ulong elements_per_kv_pair_in_page = (ulong)params.num_kv_heads * params.head_dim;
    ulong elements_per_full_page = (ulong)params.tokens_per_page * elements_per_kv_pair_in_page;

    // Vectorized cooperative copy using half4
    // Ensure elements_per_kv_pair_in_page is divisible by 4 for half4 alignment
    uint num_h4_to_copy = (uint)(elements_per_kv_pair_in_page / 4);

    // Outer loop: iterate through K/V pairs in this threadgroup's chunk
    for (uint i = 0; i < kv_pairs_per_threadgroup; i++) {
        uint current_token_global_idx = start_token_idx_for_chunk + i;

        // Boundary check - stop if we've processed all tokens
        if (current_token_global_idx >= params.total_new_tokens_to_write) {
            break;
        }

        // Retrieve metadata for current token
        uint seq_idx_in_batch = query_to_seq_map_data[current_token_global_idx];
        uint logical_token_pos_in_sequence = (uint)current_token_write_positions[current_token_global_idx];

        // Calculate target page and slot within page
        uint logical_block_idx = logical_token_pos_in_sequence / params.tokens_per_page;
        uint slot_in_page = logical_token_pos_in_sequence % params.tokens_per_page;

        // Get physical page ID from page table
        uint page_table_flat_idx = seq_idx_in_batch * params.page_table_max_logical_blocks + logical_block_idx;
        uint physical_page_id = page_table_data[page_table_flat_idx];

        ulong page_base_offset_in_pool = (ulong)physical_page_id * elements_per_full_page;
        ulong slot_base_offset_in_page = (ulong)slot_in_page * elements_per_kv_pair_in_page;

        ulong k_target_start_offset = page_base_offset_in_pool + slot_base_offset_in_page;
        ulong v_target_start_offset = page_base_offset_in_pool + slot_base_offset_in_page;

        // Calculate base pointer for source in new_keys/new_values
        ulong source_kv_pair_offset = (ulong)current_token_global_idx * elements_per_kv_pair_in_page;

        device half4* k_write_ptr_h4 = (device half4*)(global_k_pool_out + k_target_start_offset);
        device const half4* new_k_src_ptr_h4 = (device const half4*)(new_keys + source_kv_pair_offset);

        device half4* v_write_ptr_h4 = (device half4*)(global_v_pool_out + v_target_start_offset);
        device const half4* new_v_src_ptr_h4 = (device const half4*)(new_values + source_kv_pair_offset);

        // Each thread copies strided elements
        for (uint h4_idx = thread_idx_in_group;
             h4_idx < num_h4_to_copy;
             h4_idx += threads_per_tg_actual) {
            k_write_ptr_h4[h4_idx] = new_k_src_ptr_h4[h4_idx];
            v_write_ptr_h4[h4_idx] = new_v_src_ptr_h4[h4_idx];
        }
    }
}
