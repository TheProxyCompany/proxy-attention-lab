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
 * fill_kv_pages_kernel
 */
[[kernel]] void fill_kv_pages_kernel(
    device const half* new_keys,                            // [[buffer(0)]]
    device const half* new_values,                          // [[buffer(1)]]
    device half* global_k_pool,                             // [[buffer(2)]]
    device half* global_v_pool,                             // [[buffer(3)]]
    device const uint* page_table_data,                     // [[buffer(4)]]
    device const int* current_token_write_positions,        // [[buffer(5)]]
    device const uint* query_to_seq_map_data,               // [[buffer(6)]]
    constant const FillKVPagesParams& params,               // [[buffer(7)]]
    uint tid [[thread_position_in_grid]]
) {
    // Thread tid processes the tid-th new token (0 to params.num_new_tokens - 1)

    // Boundary check
    // if (tid >= params.num_new_tokens) { return; }

    // Step 1: Retrieve metadata for the current token (tid)
    // Get sequence_index_in_batch = query_to_seq_map[tid];
    // Get logical_token_pos_in_sequence = current_token_write_positions[tid];

    // Step 2: Calculate target page and slot
    // uint logical_block_idx = logical_token_pos_in_sequence / params.tokens_per_page;
    // uint slot_in_page = logical_token_pos_in_sequence % params.tokens_per_page;

    // Step 3: Get physical page ID
    // uint page_table_row_offset = sequence_index_in_batch * MAX_LOGICAL_BLOCKS_PER_SEQ;
    // uint physical_page_id = page_table[page_table_row_offset + logical_block_idx];

    // Step 4: Calculate base pointers into global pools
    // Calculate k_target_base_offset using physical_page_id, slot_in_page,
    // params.num_kv_heads, params.head_dim, params.tokens_per_page
    // Calculate v_target_base_offset similarly
    // device half* k_write_ptr = global_k_pool + k_target_base_offset;
    // device half* v_write_ptr = global_v_pool + v_target_base_offset;

    // Step 5: Calculate source pointers for new K/V
    // device const half* new_k_src_ptr = new_keys + tid * params.num_kv_heads * params.head_dim;
    // device const half* new_v_src_ptr = new_values + tid * params.num_kv_heads * params.head_dim;

    // Step 6: Perform the copy/scatter loop
    // Loop from h = 0 to params.num_kv_heads * params.head_dim - 1
    // k_write_ptr[h] = new_k_src_ptr[h];
    // v_write_ptr[h] = new_v_src_ptr[h];
}
