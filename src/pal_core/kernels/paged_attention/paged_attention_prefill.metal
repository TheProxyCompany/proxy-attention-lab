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
    uint flat_work_item_idx = tg_pos_in_grid.x; // This is the index for the (batch_item, logical_page) pair

    // Safety check: ensure this threadgroup is within the actual number of work items
    if (flat_work_item_idx >= params.num_active_batch_logical_pages) {
        return;
    }

    // For simplicity in this dummy test, let only KV_Head 0 do the writing.
    // In the real kernel, all KV_Heads would participate.
    uint assigned_global_kv_head_idx = tg_pos_in_grid.y;
    if (assigned_global_kv_head_idx != 0) {
        return;
    }

    // Let only one thread in the Threadgroup (TG) perform the write for this dummy test.
    if (local_idx_in_tg == 0) {
        // We'll write dummy data for a fixed query_idx = 0 and q_head_idx = 0.
        // In a real scenario, these would be iterated or derived.
        uint DUMMY_QUERY_TOKEN_IDX = 0;
        uint DUMMY_Q_HEAD_IDX = 0;

        // Calculate the flat index for m_locals_pass1_out and s_locals_pass1_out.
        // Shape: [query_token_count_total][num_q_heads][num_active_batch_logical_pages]
        uint stride_query_dim = params.num_q_heads * params.num_active_batch_logical_pages;
        uint stride_q_head_dim = params.num_active_batch_logical_pages;

        uint write_idx_ms = DUMMY_QUERY_TOKEN_IDX * stride_query_dim +
                            DUMMY_Q_HEAD_IDX * stride_q_head_dim +
                            flat_work_item_idx; // The page-dimension

        // Basic bounds check for safety, though DUMMY indices should be valid for test setup
        if (DUMMY_QUERY_TOKEN_IDX < params.query_token_count_total &&
            DUMMY_Q_HEAD_IDX < params.num_q_heads &&
            flat_work_item_idx < params.num_active_batch_logical_pages) {

            m_locals_pass1_out[write_idx_ms] = (float)(flat_work_item_idx) + 1.0f; // e.g., 1.0, 2.0, 3.0...
            s_locals_pass1_out[write_idx_ms] = (float)(flat_work_item_idx) + 1.0f * 100.0f; // e.g., 100.0, 200.0...
        }

        // Write a dummy value to o_partials_pass1_out as well.
        // Shape: [query_token_count_total][num_q_heads][num_active_batch_logical_pages][head_dim]
        if (params.head_dim > 0) {
            uint write_idx_o_base = write_idx_ms * params.head_dim;
             // Basic bounds check
            if (write_idx_o_base < (params.query_token_count_total * params.num_q_heads * params.num_active_batch_logical_pages * params.head_dim)) {
                o_partials_pass1_out[write_idx_o_base + 0] = ((float)flat_work_item_idx + 0.5f); // Write to first element of head_dim
            }
        }
    }
} // End of kernel
