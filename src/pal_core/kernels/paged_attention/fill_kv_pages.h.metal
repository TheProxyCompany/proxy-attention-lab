// fill_kv_pages.h.metal
// Metal shader header for filling KV cache pages.
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

#pragma once

#include <metal_stdlib>
#include "paged_attention_types.h"
#include "pal_types.h.metal"

using namespace metal;

template <typename T>
[[kernel]] void fill_kv_pages_kernel(
    device const    T*          new_keys [[buffer(0)]],
    device const    T*          new_values [[buffer(1)]],
    device const    uint*       page_table_data [[buffer(2)]],
    device const    int*        current_token_write_positions [[buffer(3)]],
    device const    uint*       query_to_seq_map_data [[buffer(4)]],
    constant const  FillKVPagesParams& params [[buffer(5)]],
    device          T*          global_k_pool_out [[buffer(6)]],
    device          T*          global_v_pool_out [[buffer(7)]],
    uint3           tg_id [[threadgroup_position_in_grid]],
    uint3           thread_idx_in_group_3d [[thread_position_in_threadgroup]],
    uint3           threads_per_tg_3d [[threads_per_threadgroup]]
) {
    // Define Vec4 using our abstraction
    using Vec4 = typename Vec<T, 4>::Type;

    // --- The rest of the kernel logic is identical, just with types replaced ---
    uint thread_idx_in_group = thread_idx_in_group_3d.x;
    uint threads_per_tg_actual = threads_per_tg_3d.x;
    uint kv_pairs_per_threadgroup = params.kv_pairs_per_threadgroup;

    uint start_token_idx_for_chunk = tg_id.x * kv_pairs_per_threadgroup;

    ulong elements_per_kv_pair_in_page = (ulong)params.num_kv_heads * params.head_dim;
    ulong elements_per_full_page = (ulong)params.tokens_per_page * elements_per_kv_pair_in_page;

    uint num_vec4_to_copy = (uint)(elements_per_kv_pair_in_page / 4);

    for (uint i = 0; i < kv_pairs_per_threadgroup; i++) {
        uint current_token_global_idx = start_token_idx_for_chunk + i;

        if (current_token_global_idx >= params.total_new_tokens_to_write) {
            break;
        }

        uint seq_idx_in_batch = query_to_seq_map_data[current_token_global_idx];
        uint logical_token_pos_in_sequence = (uint)current_token_write_positions[current_token_global_idx];

        uint logical_block_idx = logical_token_pos_in_sequence / params.tokens_per_page;
        uint slot_in_page = logical_token_pos_in_sequence % params.tokens_per_page;

        uint page_table_flat_idx = seq_idx_in_batch * params.page_table_max_logical_blocks + logical_block_idx;
        uint physical_page_id = page_table_data[page_table_flat_idx];

        ulong page_base_offset_in_pool = (ulong)physical_page_id * elements_per_full_page;
        ulong slot_base_offset_in_page = (ulong)slot_in_page * elements_per_kv_pair_in_page;

        ulong k_target_start_offset = page_base_offset_in_pool + slot_base_offset_in_page;
        ulong v_target_start_offset = page_base_offset_in_pool + slot_base_offset_in_page;

        ulong source_kv_pair_offset = (ulong)current_token_global_idx * elements_per_kv_pair_in_page;

        device Vec4* k_write_ptr = (device Vec4*)(global_k_pool_out + k_target_start_offset);
        device const Vec4* new_k_src_ptr = (device const Vec4*)(new_keys + source_kv_pair_offset);

        device Vec4* v_write_ptr = (device Vec4*)(global_v_pool_out + v_target_start_offset);
        device const Vec4* new_v_src_ptr = (device const Vec4*)(new_values + source_kv_pair_offset);

        for (uint v_idx = thread_idx_in_group; v_idx < num_vec4_to_copy; v_idx += threads_per_tg_actual) {
            k_write_ptr[v_idx] = new_k_src_ptr[v_idx];
            v_write_ptr[v_idx] = new_v_src_ptr[v_idx];
        }
    }
}
