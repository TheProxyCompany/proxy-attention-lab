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
    device      const float* m_pass1_results        [[buffer(13)]],  // Local max scores per page
    device      const float* s_pass1_results        [[buffer(14)]],  // Local sum-exponentials per page
    device      const half*  o_pass1_results        [[buffer(15)]],  // Unnormalized partial V-accumulations
    // Active work items buffer
    device      const uint2* active_work_item_pairs [[buffer(16)]],  // Active (batch_item, logical_page) pairs
    // Parameters
    constant    const PagedAttentionParams& params  [[buffer(7)]],
    // Final output buffer
    device      half* final_output_buffer           [[buffer(17)]],
    // Thread/grid identifiers
    uint actual_simd_width                          [[threads_per_simdgroup]],
    threadgroup float* tg_mem                       [[threadgroup(0)]],
    uint3       tg_pos_in_grid                      [[threadgroup_position_in_grid]],
    uint3       tg_dim                              [[threads_per_threadgroup]],
    uint        local_idx_in_tg                     [[thread_index_in_threadgroup]]
) {
    // --- TGMem Carving for Pass 2 ---
    threadgroup uchar* tg_mem_base_byte_ptr = (threadgroup uchar*)tg_mem;
    uintptr_t current_tg_offset = 0;

    current_tg_offset = (current_tg_offset + kAlignmentMask) & ~kAlignmentMask;
    threadgroup float* O_final_accumulators_base_tg = (threadgroup float*)(tg_mem_base_byte_ptr + current_tg_offset);
    // Each thread gets its own slice of this O_final_accumulators_base_tg
    threadgroup float* O_final_for_this_thread_tg = O_final_accumulators_base_tg + (local_idx_in_tg * params.head_dim);

    // --- 1. Determine the (Query Token Block, Q-Head Block) this TG is responsible for ---
    uint query_token_block_offset = tg_pos_in_grid.x * params.pass2_token_block_size;
    uint q_head_block_offset = tg_pos_in_grid.y * params.pass2_qhead_block_size;

    // --- 2. Each thread iterates over the output items assigned to it within the TG's block ---
    // N_outputs_per_TG is the number of (QueryToken, QHead) pairs this TG processes.
    uint num_outputs_in_tg_block = params.pass2_token_block_size * params.pass2_qhead_block_size;
    uint threads_in_tg_p2 = tg_dim.x; // Total threads in this Pass 2 TG

    for (uint item_idx_processed_by_thread = local_idx_in_tg;
          item_idx_processed_by_thread < num_outputs_in_tg_block;
          item_idx_processed_by_thread += threads_in_tg_p2) {

        // Map the flat item_idx_processed_by_thread to 2D local indices within the block
        uint local_token_idx_in_block = item_idx_processed_by_thread % params.pass2_token_block_size;
        uint local_q_head_idx_in_block = item_idx_processed_by_thread / params.pass2_token_block_size; // Integer division

        // Calculate the absolute master_query_idx and target_global_q_head_idx for this item
        uint master_query_idx = query_token_block_offset + local_token_idx_in_block;
        uint target_global_q_head_idx = q_head_block_offset + local_q_head_idx_in_block;

        // Boundary checks for the specific item this thread is processing
        if (master_query_idx >= params.query_token_count_total || target_global_q_head_idx >= params.num_q_heads) {
            continue;
        }

        // --- 3. Initialize and find M_global for this (master_query_idx, target_global_q_head_idx) item ---
        float M_global_for_this_item = -INFINITY;

        // Loop over all pages that Pass 1 processed.
        // params.num_active_batch_logical_pages is the size of the third dimension of m_pass1_results.
        for (uint page_idx = 0; page_idx < params.num_active_batch_logical_pages; ++page_idx) {
            // Calculate flat index into m_pass1_results.
            // Layout of m_pass1_results: [TotalQueryTokens, NumQHeads, NumActivePages]
            ulong m_s_stride_qhead = params.num_active_batch_logical_pages;
            ulong m_s_stride_query = params.num_q_heads * params.num_active_batch_logical_pages;

            ulong flat_idx_m_pass1 = (ulong)master_query_idx * m_s_stride_query +
                                     (ulong)target_global_q_head_idx * m_s_stride_qhead +
                                     page_idx;

            float current_page_m_local = m_pass1_results[flat_idx_m_pass1];
            M_global_for_this_item = max(M_global_for_this_item, current_page_m_local);
        }

        // --- Step 4: Calculate S_global_for_this_item ---
        float S_global_for_this_item = 0.0f;
        float kahan_c_for_S_global = 0.0f; // Kahan compensation term for S_global

        // fuse with above for loop eventually.
        for (uint page_idx = 0; page_idx < params.num_active_batch_logical_pages; ++page_idx) {
                        // Layout: [TotalQueryTokens, NumQHeads, NumActivePages]
            ulong m_s_stride_qhead = params.num_active_batch_logical_pages;
            ulong m_s_stride_query = params.num_q_heads * params.num_active_batch_logical_pages;

            ulong flat_idx_ms_pass1 = (ulong)master_query_idx * m_s_stride_query +
                                      (ulong)target_global_q_head_idx * m_s_stride_qhead +
                                      page_idx;

            float m_local_p = m_pass1_results[flat_idx_ms_pass1];
            float s_local_p = s_pass1_results[flat_idx_ms_pass1];

            // Calculate exp(m_local_p - M_global_for_this_item)
            // Clamp the argument to exp.
            float rescale_factor_exp_arg = (M_global_for_this_item == -INFINITY && m_local_p == -INFINITY) ?
                                           params.log_exp_min_clamp :
                                           max(m_local_p - M_global_for_this_item, params.log_exp_min_clamp);
            float rescale_factor = precise::exp(rescale_factor_exp_arg);

            // Contribution of this page to S_global
            float s_page_contribution = s_local_p * rescale_factor;

            // Kahan summation for S_global_for_this_item
            float y_kahan = s_page_contribution - kahan_c_for_S_global;
            float t_kahan = S_global_for_this_item + y_kahan;
            kahan_c_for_S_global = (t_kahan - S_global_for_this_item) - y_kahan;
            S_global_for_this_item = t_kahan;
        }

        // --- BEGIN Step 5: O_partial Aggregation & Normalization (using TGMem for O_final) ---

        // Initialize this thread's dedicated O_final accumulator in TGMem to zeros.
        // Each thread zeros its own HeadDim slice.
        for (uint h_idx_init = 0; h_idx_init < params.head_dim; h_idx_init += 4) {
             *((threadgroup float4*)(O_final_for_this_thread_tg + h_idx_init)) = float4(0.0f);
        }

        // Loop over all pages again for this item to aggregate o_partials
        for (uint page_idx = 0; page_idx < params.num_active_batch_logical_pages; ++page_idx) {
            // Calculate flat index for m_pass1_results
            ulong m_s_stride_qhead = params.num_active_batch_logical_pages;
            ulong m_s_stride_query = params.num_q_heads * params.num_active_batch_logical_pages;
            ulong flat_idx_m_pass1 = (ulong)master_query_idx * m_s_stride_query +
                                     (ulong)target_global_q_head_idx * m_s_stride_qhead +
                                     page_idx;
            float m_local_p = m_pass1_results[flat_idx_m_pass1];

            // Calculate rescale_factor: exp(m_local_p - M_global_for_this_item)
            float rescale_factor_exp_arg = (M_global_for_this_item == -INFINITY && m_local_p == -INFINITY) ?
                                           params.log_exp_min_clamp :
                                           max(m_local_p - M_global_for_this_item, params.log_exp_min_clamp);
            float rescale_factor = precise::exp(rescale_factor_exp_arg);

            // If rescale_factor is effectively zero, this page's o_partial won't contribute,
            // so we can skip the expensive HeadDim loop.
            if (rescale_factor < kEpsilonForZeroGuard) {
                continue;
            }

            // Calculate base offset for o_partials_pass1_out for this item and page
            // Layout: [TotalQueries, NumQHeads, NumActivePages, HeadDim]
            ulong o_stride_page = params.head_dim;
            ulong o_stride_qhead = params.num_active_batch_logical_pages * params.head_dim;
            ulong o_stride_query = params.num_q_heads * params.num_active_batch_logical_pages * params.head_dim;

            ulong base_offset_o_partial = (ulong)master_query_idx * o_stride_query +
                                          (ulong)target_global_q_head_idx * o_stride_qhead +
                                          (ulong)page_idx * o_stride_page;

            device const half* o_partial_p_ptr = o_pass1_results + base_offset_o_partial;

            // Aggregate the HeadDim components into this thread's TGMem O accumulator
            // This inner loop is done by THIS thread, operating on its O_final_for_this_thread_tg.
            for (uint h_idx = 0; h_idx < params.head_dim; h_idx += 4) {
                float4 o_partial_chunk_f = float4( *((device const half4*)(o_partial_p_ptr + h_idx)) );

                // Accumulate into this thread's TGMem accumulator
                *((threadgroup float4*)(O_final_for_this_thread_tg + h_idx)) += o_partial_chunk_f * rescale_factor;
            }
        }

        // Normalize the final O_vector (in TGMem) by S_global_for_this_item
        float inv_S_global = (S_global_for_this_item > kSmallDenominatorThreshold) ?
                              (1.0f / S_global_for_this_item) : 0.0f;

        for (uint h_idx = 0; h_idx < params.head_dim; h_idx += 4) {
            *((threadgroup float4*)(O_final_for_this_thread_tg + h_idx)) *= inv_S_global;
        }
        // --- END Step 5: O_partial Aggregation & Normalization ---

        // --- BEGIN Step 6: Write O_final_for_this_thread_tg to final_output_buffer ---
        // Calculate the flat index for this item in the 1D view of [TotalItems, HeadDim].
        ulong output_item_flat_idx = (ulong)master_query_idx * params.num_q_heads + target_global_q_head_idx;
        ulong base_offset_final_output = output_item_flat_idx * params.head_dim;

        device half* final_out_ptr_for_item = final_output_buffer + base_offset_final_output;

        // This thread writes its HeadDim float values (from O_final_for_this_thread_tg)
        // to global memory, converting to half.
        // This is a HeadDim-wide operation.
        for (uint h_idx = 0; h_idx < params.head_dim; h_idx += 4) { // Process in float4 chunks
            // Read the float4 chunk from this thread's TGMem accumulator
            float4 val_f4_to_write = *((threadgroup float4*)(O_final_for_this_thread_tg + h_idx));

            // Convert to half4 and write to global memory
            *((device half4*)(final_out_ptr_for_item + h_idx)) = half4(val_f4_to_write);
        }
        // --- END Step 6: Write O_final_for_this_thread_tg to final_output_buffer ---
    } // End loop over items assigned to this thread

} // End of kernel

// =================================================================================================
// KERNEL DEVELOPMENT NOTES & CURRENT STATUS (As of 2025-05-29, S_global implemented)
// Advisor: K (Jack + K collaboration)
//
// I. KERNEL OBJECTIVE (Pass 2 of Two-Pass Page-Centric Prefill):
//    This kernel consumes the intermediate outputs from Pass 1 (`m_locals_pass1_out`,
//    `s_locals_pass1_out`, `o_partials_pass1_out`) to produce the final normalized
//    attention output vector for each (Query Token, Query Head) pair.
//
// II. CURRENT IMPLEMENTATION STATE:
//    1. Dispatch & TG/Thread Role:
//        - TG dispatched for a block of (QueryTokens, QHeads).
//        - Each thread iterates over a subset of (QueryToken, QHead) items within the TG's block.
//    2. M_global Calculation:
//        - For each (QueryToken, QHead) item, the thread iterates over all active pages
//          from Pass 1, reading `m_pass1_results` to find the true global maximum score
//          (M_global_for_this_item). This logic is complete.
//    3. S_global Calculation:
//        - For each (QueryToken, QHead) item, after M_global_for_this_item is found,
//          the thread re-iterates over all active pages.
//        - It reads `m_pass1_results` and `s_pass1_results`.
//        - It calculates `s_page_contribution = s_local_p * exp(m_local_p - M_global_for_this_item)`.
//        - It accumulates these contributions into `S_global_for_this_item` using Kahan summation.
//          This logic is complete.
//    4. Pass 1 State for Testing:
//        - Currently, Pass 1 (`paged_attn_prefill_kernel.metal`) is being run with its
//          `dot_product_qk` call commented out and `score` hardcoded to `1.0f` (with
//          proper masking for invalid K/V pairs). This is to speed up Pass 1 execution
//          during Pass 2 development and provide simpler intermediate values.
//
// III. PERFORMANCE OBSERVATIONS (Pass 1 `score=1.0f` + Pass 2 up to S_global):
//    - Benchmark (2025-05-29):
//      "cpp_pal_paged_attention_prefill": { "4096.0": 90.9279 } ms.
//    - Compared to Pass 2 only calculating M_global (which was ~87.15 ms at 4096), adding
//      the S_global calculation (which involves another loop over pages and more reads/math)
//      added ~3.77 ms at 4096 tokens. This overhead is noticeable.
//
// IV. NEXT STEPS for Pass 2:
//    1. Implement O_partial Aggregation & Normalization:
//        - Loop over pages again (or fuse with S_global loop).
//        - Read `o_partials_pass1_out` and `m_pass1_results`.
//        - Rescale `o_partial` using `exp(m_local_p - M_global_for_this_item)`.
//        - Accumulate these rescaled O_partial vectors into a final O_vector for the item.
//        - Normalize the final O_vector by dividing by S_global_for_this_item.
//    2. Write to `final_output_buffer`.
//    3. Optimization: Consider fusing the M_global, S_global, and O_partial aggregation
//       loops over pages to reduce redundant reads of `m_pass1_results` and loop overhead.
//
// V. FUTURE CONSIDERATIONS (Relating to Chunked Prefill):
//    - The current two-pass prefill design processes the entire sequence length (up to
//      hardware/buffer limits) in one go for Pass 1, followed by one go for Pass 2.
//    - If "Chunked Prefill" implies breaking a very long sequence into smaller logical
//      "chunks" that are processed independently (e.g., to manage memory for extremely
//      long sequences beyond what even paged attention can hold, or for streaming input),
//      that would likely involve multiple invocations of this entire two-pass kernel pair,
//      or a modified kernel architecture. This is a higher-level strategy beyond the
//      current kernel implementation details. The current Pass 1/Pass 2 are designed
//      for a single "prefill" operation over the given `seq_len_for_this_batch_item`.
// =================================================================================================
// =================================================================================================
// KERNEL DEVELOPMENT NOTES & CURRENT STATUS (As of 2025-05-29, Full Pass 2 Implemented)
// Advisor: K (Jack + K collaboration)
//
// I. KERNEL OBJECTIVE (Pass 2 of Two-Pass Page-Centric Prefill):
//    This kernel consumes the intermediate outputs from Pass 1 (`m_locals_pass1_out`,
//    `s_locals_pass1_out`, `o_partials_pass1_out`) to produce the final normalized
//    attention output vector for each (Query Token, Query Head) pair.
//
// II. IMPLEMENTATION STATE:
//    1. Dispatch & TG/Thread Role:
//        - TG dispatched for a block of (QueryTokens, QHeads) as defined by C++ primitive
//          (params.pass2_token_block_size, params.pass2_qhead_block_size).
//        - Each thread iterates over a subset of (QueryToken, QHead) items within the TG's block.
//    2. M_global Calculation (Step 3 in code):
//        - For each item, iterates over all active pages from Pass 1, reading `m_pass1_results`
//          to find the true global maximum score (M_global_for_this_item).
//    3. S_global Calculation (Step 4 in code):
//        - For each item, re-iterates over pages, reads `m_pass1_results` & `s_pass1_results`.
//        - Calculates `s_page_contribution = s_local_p * exp(m_local_p - M_global_for_this_item)`.
//        - Accumulates into `S_global_for_this_item` using Kahan summation.
//    4. O_partial Aggregation & Normalization (Step 5 in code):
//        - Per-thread `O_final_for_this_thread_tg` accumulator (HeadDim floats) in TGMem zeroed.
//        - Re-iterates over pages, reads `m_pass1_results` & `o_pass1_results`.
//        - Calculates `rescale_factor = exp(m_local_p - M_global_for_this_item)`.
//        - Accumulates `rescaled_o_partial = o_partial_p * rescale_factor` into
//          `O_final_for_this_thread_tg` (HeadDim-wide vector sum).
//        - Normalizes `O_final_for_this_thread_tg` by dividing by `S_global_for_this_item`.
//    5. Final Output Write (Step 6 in code):
//        - Writes the normalized `O_final_for_this_thread_tg` (converted to half) to the
//          `final_output_buffer` at the correct (Item, HeadDim) location.
//
// III. PERFORMANCE OBSERVATIONS (Pass 1 was using `score=1.0f` for these Pass 2 dev steps):
//    - Cost of M_global calc (vs. empty Pass 2): Minimal.
//    - Cost of S_global calc (vs. M_global only): Added ~3.77 ms at 4096 tokens.
//      (e.g., 4096 tokens: from ~87.15 ms to ~90.93 ms).
//    - Cost of O-Agg & Norm (vs. M+S_global only): Added ~34.5 ms at 4096 tokens.
//      (e.g., 4096 tokens: from ~90.93 ms to ~125.42 ms). This step reads the largest
//      intermediate buffer (`o_partials_pass1_out`) and does HeadDim-wide vector math.
//    - Cost of Final Write (vs. M,S,O-Agg-Norm only): Minimal change.
//      (e.g., 4096 tokens: from ~125.42 ms to ~123.21 ms - within variance).
//
// IV. NEXT STEPS & FUTURE OPTIMIZATIONS:
//    1. Correctness Verification: Rigorously test numerical output of the full two-pass
//       system (with actual dot products in Pass 1) against a Python reference.
//       Address any discrepancies (like the current test failures).
//    2. Loop Fusion: The three separate loops over `page_idx` (for M, S, and O) in this
//       kernel are a prime candidate for fusion to reduce redundant global reads of
//       `m_pass1_results` and loop overhead. This should be done after correctness is confirmed.
//    3. TG-Wide Parallel Reduction (If needed for page loop): If `params.num_active_batch_logical_pages`
//       is very large, the current per-thread iteration over all pages might become a bottleneck.
//       A TG-wide parallel reduction strategy (where threads cooperatively reduce over pages
//       for a single output item) could be explored. Current C++ TGMem allocation for Pass 2
//       already provides some scratch space that could support this.
// =================================================================================================
