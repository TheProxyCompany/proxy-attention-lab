// paged_attn_prefill_pass2.metal
// Metal shader implementation for Pass 2 of page-centric prefill architecture.
// This kernel aggregates Pass 1 outputs to produce final normalized attention.
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
 * paged_attn_prefill_pass2_kernel
 * --------------------------------
 * Pass 2 of the page-centric prefill architecture.
 * This kernel:
 * - Reads page-level softmax statistics from Pass 1
 * - Computes global softmax normalization
 * - Aggregates partial V-accumulations
 * - Writes final normalized attention output
 *
 * Each threadgroup produces one final output vector for a specific
 * (query_token, q_head) pair by aggregating contributions from all pages.
 */
[[kernel]] void paged_attn_prefill_pass2_kernel(
    // Pass 1 output buffers
    device      const float* m_pass1_results        [[buffer(17)]],  // Local max scores per page
    device      const float* s_pass1_results        [[buffer(18)]],  // Local sum-exponentials per page
    device      const half*  o_pass1_results        [[buffer(19)]],  // Unnormalized partial V-accumulations
    // Active work items buffer
    device      const uint2* active_work_item_pairs [[buffer(20)]],  // Active (batch_item, logical_page) pairs
    // Parameters
    constant    const PagedAttentionParams& params  [[buffer(7)]],
    // Final output buffer
    device      half* final_output_buffer           [[buffer(8)]],
    // Thread/grid identifiers
    uint actual_simd_width                          [[threads_per_simdgroup]],
    threadgroup float* tg_mem                       [[threadgroup(0)]],
    uint3       tg_pos_in_grid                      [[threadgroup_position_in_grid]],
    uint3       tg_dim                              [[threads_per_threadgroup]],
    uint        local_idx_in_tg                     [[thread_index_in_threadgroup]]
) {
    // This dummy kernel will have only one threadgroup (0,0) and one thread within it (0)
    // write a result for a single output element (query 0, q_head 0, dim 0).
    if (tg_pos_in_grid.x == 0 && tg_pos_in_grid.y == 0 && local_idx_in_tg == 0) {
        uint DUMMY_TARGET_QUERY_TOKEN_IDX = 0;
        uint DUMMY_TARGET_Q_HEAD_IDX = 0;

        float accumulated_m_val = 0.0f;
        float accumulated_s_val = 0.0f;
        half accumulated_o_val_elem0 = 0.0h;

        // Strides for reading intermediate buffers (must match Pass 1 write logic)
        uint stride_query_dim_ms = params.num_q_heads * params.num_active_batch_logical_pages;
        uint stride_q_head_dim_ms = params.num_active_batch_logical_pages;

        // Iterate through all contributions made by Pass 1 (indexed by flat_work_item_idx)
        for (uint page_contribution_idx = 0; page_contribution_idx < params.num_active_batch_logical_pages; ++page_contribution_idx) {
            // In a real kernel, here you'd use active_work_item_pairs[page_contribution_idx]
            // to determine if this page contribution is relevant for the
            // DUMMY_TARGET_QUERY_TOKEN_IDX. For this dummy test, we assume all are relevant.

            uint read_idx_ms = DUMMY_TARGET_QUERY_TOKEN_IDX * stride_query_dim_ms +
                               DUMMY_TARGET_Q_HEAD_IDX * stride_q_head_dim_ms +
                               page_contribution_idx;

            // Basic bounds check
            if (DUMMY_TARGET_QUERY_TOKEN_IDX < params.query_token_count_total &&
                DUMMY_TARGET_Q_HEAD_IDX < params.num_q_heads &&
                page_contribution_idx < params.num_active_batch_logical_pages) {

                accumulated_m_val += m_pass1_results[read_idx_ms];
                accumulated_s_val += s_pass1_results[read_idx_ms];
            }

            if (params.head_dim > 0) {
                uint read_idx_o_base = read_idx_ms * params.head_dim;
                if (read_idx_o_base < (params.query_token_count_total * params.num_q_heads * params.num_active_batch_logical_pages * params.head_dim)) {
                    accumulated_o_val_elem0 += o_pass1_results[read_idx_o_base + 0];
                }
            }
        }

        // Write the accumulated dummy sum to the first few elements of the output vector
        // for (query 0, q_head 0).
        // final_output_buffer is shape [TotalQueryTokens * NumQHeads, HeadDim]
        uint output_item_idx = DUMMY_TARGET_QUERY_TOKEN_IDX * params.num_q_heads + DUMMY_TARGET_Q_HEAD_IDX;
        uint output_base_addr = output_item_idx * params.head_dim;

        if (output_item_idx < (params.query_token_count_total * params.num_q_heads)) {
            if (params.head_dim > 0) final_output_buffer[output_base_addr + 0] = (half)accumulated_m_val;
            if (params.head_dim > 1) final_output_buffer[output_base_addr + 1] = (half)accumulated_s_val;
            if (params.head_dim > 2) final_output_buffer[output_base_addr + 2] = accumulated_o_val_elem0;
            // Zero out remaining elements of this output vector for clarity
            for (uint d = 3; d < params.head_dim; ++d) {
                final_output_buffer[output_base_addr + d] = 0.0h;
            }
        }
    }
} // End of kernel
