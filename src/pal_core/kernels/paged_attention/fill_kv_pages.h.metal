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
#include "utils.h.metal"

using namespace metal;

// K-cache layout: [pages, heads, head_dim/x, tokens_per_page, x]
//         where x = MEMORY_ALIGNMENT_BYTES / sizeof(T)
// V-cache layout: [pages, heads, head_dim, tokens_per_page]

template <typename T>
[[kernel]] void fill_kv_pages_kernel(
    // --- Data Buffers ---
    device const T*      new_keys                      [[buffer(0)]],
    device const T*      new_values                    [[buffer(1)]],
    device const uint*   page_table                    [[buffer(2)]],
    device const int*    token_write_positions         [[buffer(3)]],
    device const uint*   query_to_seq_map              [[buffer(4)]],
    constant const FillKVPagesParams& params           [[buffer(5)]],
    device T*            global_k_pool_out             [[buffer(6)]],
    device T*            global_v_pool_out             [[buffer(7)]],

    // --- Thread Identifiers ---
    uint3                tg_id                         [[threadgroup_position_in_grid]],
    uint                 local_idx                     [[thread_index_in_threadgroup]]
)
{
    // 1. Identify which tokens this threadgroup is responsible for.
    constexpr int vec_size = MEMORY_ALIGNMENT_BYTES / sizeof(T);
    uint start_token_idx = tg_id.x * params.tokens_per_threadgroup;
    using VecT = typename Vec<T, vec_size>::Type;

    // Calculate total vectors to copy for one token.
    const uint total_vectors_per_token = (params.num_kv_heads * params.head_dim) / vec_size;

    // 2. Loop over the tokens assigned to this threadgroup.
    for (uint i = 0; i < params.tokens_per_threadgroup; ++i) {
        uint token_idx = start_token_idx + i;

        // Boundary check: ensure this token actually exists.
        if (token_idx >= params.total_new_tokens_to_write) {
            return; // This thread has no more work to do.
        }

        // 3. Find the destination physical page and slot for this token.
        uint seq_idx = query_to_seq_map[token_idx];
        uint logical_pos = token_write_positions[token_idx];

        uint logical_block_idx = logical_pos / params.tokens_per_page;
        uint slot_in_page = logical_pos % params.tokens_per_page;

        uint page_table_idx = seq_idx * params.page_table_max_logical_blocks + logical_block_idx;
        uint physical_page_id = page_table[page_table_idx];

        const uint num_threads = params.threads_per_threadgroup;
        device const T* source_k_base = new_keys + (ulong)token_idx * params.num_kv_heads * params.head_dim;
        device const T* source_v_base = new_values + (ulong)token_idx * params.num_kv_heads * params.head_dim;

        ulong page_offset = (ulong)physical_page_id * params.num_kv_heads * params.tokens_per_page * params.head_dim;
        device T* k_page_base = global_k_pool_out + page_offset;
        device T* v_page_base = global_v_pool_out + page_offset;

        for (uint vec_idx = local_idx; vec_idx < total_vectors_per_token; vec_idx += num_threads) {
            // 5a. Read one vector from the source arrays.
            VecT k_vec = ((device const VecT*)source_k_base)[vec_idx];
            VecT v_vec = ((device const VecT*)source_v_base)[vec_idx];

            // 5b. Deconstruct the linear vector index to find the head and element position.
            uint linear_elem_idx = vec_idx * vec_size;
            uint kv_head_idx = linear_elem_idx / params.head_dim;
            uint elem_in_head_idx = linear_elem_idx % params.head_dim;

            // 5c. Write K-cache vector (coalesced)
            {
                uint vec_in_head_idx = elem_in_head_idx / vec_size;

                ulong k_head_offset = (ulong)kv_head_idx * params.head_dim * params.tokens_per_page;
                ulong k_vec_chunk_offset = (ulong)vec_in_head_idx * params.tokens_per_page * vec_size;
                ulong k_slot_offset = (ulong)slot_in_page * vec_size;

                device VecT* k_write_ptr = (device VecT*)(k_page_base + k_head_offset + k_vec_chunk_offset + k_slot_offset);
                *k_write_ptr = k_vec;
            }

            // 5d. Write V-cache vector (strided)
            {
                ulong v_head_offset = (ulong)kv_head_idx * params.head_dim * params.tokens_per_page;
                ulong v_elem_start_offset = (ulong)elem_in_head_idx * params.tokens_per_page;

                device T* v_write_ptr = v_page_base + v_head_offset + v_elem_start_offset + slot_in_page;

                // This is a strided write. Each element is tokens_per_page apart.
                #pragma unroll
                for (int j = 0; j < vec_size; ++j) {
                    v_write_ptr[j * params.tokens_per_page] = v_vec[j];
                }
            }
        } // end vectorized read/write loop

    } // end token loop

} // end kernel
