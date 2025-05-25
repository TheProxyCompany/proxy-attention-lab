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
    // Early exit for degenerate case
    if (params.head_dim == 0) {
        return;
    }

    // Constants
    constexpr uint PREFILL_PASS1_Q_HEAD_BLOCK_SIZE_CONST = 8;
    constexpr float kEpsilonForZeroGuard = 1e-12f;
    constexpr uintptr_t kAlignmentBytes = 64;
    constexpr uintptr_t kAlignmentMask = kAlignmentBytes - 1;

    // Calculate block start indices for 2D blocking
    uint token_block_start_idx = tg_pos_in_grid.x * params.pass2_token_block_size;
    uint q_head_block_start_idx = tg_pos_in_grid.y * params.pass2_qhead_block_size;

    // Early return if block is completely out of bounds
    if (token_block_start_idx >= params.query_token_count_total ||
        q_head_block_start_idx >= params.num_q_heads) {
        return;
    }

    uint local_thread_idx = local_idx_in_tg;

    // // Process each (token, q_head) pair in the assigned 2D block
    // for (uint t_offset = 0; t_offset < params.pass2_token_block_size; ++t_offset) {
    //     uint target_q_token_idx = token_block_start_idx + t_offset;

    //     // Check token bounds
    //     if (target_q_token_idx >= params.query_token_count_total) {
    //         continue;
    //     }

    // } // End of outer loop (token)

} // End of kernel
