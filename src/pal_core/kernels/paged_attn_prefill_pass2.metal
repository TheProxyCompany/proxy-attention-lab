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

    // Identify target output vector this TG is responsible for
    uint target_q_token_idx = tg_pos_in_grid.x;
    uint target_q_head_idx = tg_pos_in_grid.y;

    // Check bounds
    if (target_q_token_idx >= params.num_sequences_in_batch || 
        target_q_head_idx >= params.num_q_heads) {
        return;
    }

    uint local_thread_idx = local_idx_in_tg;

    // --- Threadgroup Memory Layout ---
    // We need space for:
    // 1. Thread-local max values for Pass 2a reduction
    // 2. Thread-local sum values for Pass 2b reduction  
    // 3. Thread-local partial output accumulation [threads][head_dim]
    // 4. Global statistics (M_global, S_global)
    // 5. Final output accumulator

    threadgroup float* tg_thread_max_scratch = tg_mem;
    
    uintptr_t current_offset = (uintptr_t)(tg_thread_max_scratch + tg_dim.x);
    current_offset = (current_offset + kAlignmentMask) & ~kAlignmentMask;
    threadgroup float* tg_thread_sum_scratch = (threadgroup float*)current_offset;
    
    current_offset = (uintptr_t)(tg_thread_sum_scratch + tg_dim.x);
    current_offset = (current_offset + kAlignmentMask) & ~kAlignmentMask;
    threadgroup float* tg_global_stats = (threadgroup float*)current_offset; // [0] = M_global, [1] = S_global
    
    current_offset = (uintptr_t)(tg_global_stats + 2);
    current_offset = (current_offset + kAlignmentMask) & ~kAlignmentMask;
    threadgroup float* tg_thread_o_accumulator = (threadgroup float*)current_offset; // [threads][head_dim]
    
    current_offset = (uintptr_t)(tg_thread_o_accumulator + tg_dim.x * params.head_dim);
    current_offset = (current_offset + kAlignmentMask) & ~kAlignmentMask;
    threadgroup float* tg_final_o_accumulator = (threadgroup float*)current_offset; // [head_dim]

    // Initialize thread-local accumulators
    float thread_local_max = -INFINITY;
    float thread_local_sum = 0.0f;
    float thread_local_sum_kahan_comp = 0.0f;
    float thread_local_o_accum[kMaxHeadDimMetal];
    for (uint d = 0; d < params.head_dim; ++d) {
        thread_local_o_accum[d] = 0.0f;
    }

    // For now, assume all active pages contributed to this query
    // In a more sophisticated version, we'd use the Relevant Query Map
    // TODO: This should be passed from C++ - for now use a placeholder
    uint num_contributing_pages = 1; // Will be updated when C++ passes this value

    // Calculate strides for Pass 1 output indexing
    // Assuming layout: [num_active_pages][num_q_head_blocks][q_heads_per_block][optional_head_dim]
    uint num_q_head_blocks_total = (params.num_q_heads + PREFILL_PASS1_Q_HEAD_BLOCK_SIZE_CONST - 1) / 
                                   PREFILL_PASS1_Q_HEAD_BLOCK_SIZE_CONST;
    uint q_head_block_idx = target_q_head_idx / PREFILL_PASS1_Q_HEAD_BLOCK_SIZE_CONST;
    uint q_head_offset_in_block = target_q_head_idx % PREFILL_PASS1_Q_HEAD_BLOCK_SIZE_CONST;

    // --- Pass 2a: Find Global Max (M_global) ---
    // Each thread processes a subset of pages
    for (uint page_iter = local_thread_idx; page_iter < num_contributing_pages; page_iter += tg_dim.x) {
        // Calculate index into Pass 1 m_locals output
        uint m_index = page_iter * num_q_head_blocks_total * PREFILL_PASS1_Q_HEAD_BLOCK_SIZE_CONST +
                       q_head_block_idx * PREFILL_PASS1_Q_HEAD_BLOCK_SIZE_CONST +
                       q_head_offset_in_block;
        
        float m_local = m_pass1_results[m_index];
        thread_local_max = max(thread_local_max, m_local);
    }

    // Store thread-local max for reduction
    tg_thread_max_scratch[local_thread_idx] = thread_local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Threadgroup reduction to find global max
    // Simple tree reduction
    for (uint stride = tg_dim.x / 2; stride > 0; stride >>= 1) {
        if (local_thread_idx < stride) {
            tg_thread_max_scratch[local_thread_idx] = max(tg_thread_max_scratch[local_thread_idx],
                                                          tg_thread_max_scratch[local_thread_idx + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Thread 0 writes final M_global
    if (local_thread_idx == 0) {
        tg_global_stats[0] = tg_thread_max_scratch[0];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // All threads read M_global
    float M_global = tg_global_stats[0];

    // --- Pass 2b: Compute Global Sum (S_global) and Accumulate Scaled Partial Outputs ---
    // Each thread processes a subset of pages
    for (uint page_iter = local_thread_idx; page_iter < num_contributing_pages; page_iter += tg_dim.x) {
        // Calculate indices into Pass 1 outputs
        uint ms_index = page_iter * num_q_head_blocks_total * PREFILL_PASS1_Q_HEAD_BLOCK_SIZE_CONST +
                        q_head_block_idx * PREFILL_PASS1_Q_HEAD_BLOCK_SIZE_CONST +
                        q_head_offset_in_block;
        
        uint o_base_index = ms_index * params.head_dim;

        // Read Pass 1 outputs for this page
        float m_local = m_pass1_results[ms_index];
        float s_local = s_pass1_results[ms_index];
        
        // Calculate scaling factor
        float scale_factor = precise::exp(max(m_local - M_global, params.log_exp_min_clamp));
        
        // Update thread-local sum with Kahan summation
        float term_to_add = s_local * scale_factor;
        float y_kahan = term_to_add - thread_local_sum_kahan_comp;
        float t_kahan = thread_local_sum + y_kahan;
        thread_local_sum_kahan_comp = (t_kahan - thread_local_sum) - y_kahan;
        thread_local_sum = t_kahan;
        
        // Accumulate scaled partial output
        for (uint d = 0; d < params.head_dim; ++d) {
            thread_local_o_accum[d] += float(o_pass1_results[o_base_index + d]) * scale_factor;
        }
    }

    // Store thread-local sum for reduction
    tg_thread_sum_scratch[local_thread_idx] = thread_local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Threadgroup reduction for global sum
    for (uint stride = tg_dim.x / 2; stride > 0; stride >>= 1) {
        if (local_thread_idx < stride) {
            tg_thread_sum_scratch[local_thread_idx] += tg_thread_sum_scratch[local_thread_idx + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Thread 0 writes final S_global
    if (local_thread_idx == 0) {
        tg_global_stats[1] = tg_thread_sum_scratch[0];
    }

    // Store each thread's local output accumulator in threadgroup memory
    threadgroup float* my_o_accum = tg_thread_o_accumulator + local_thread_idx * params.head_dim;
    for (uint d = 0; d < params.head_dim; ++d) {
        my_o_accum[d] = thread_local_o_accum[d];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Reduce output accumulators across all threads
    // Each thread sums one or more dimensions
    for (uint d = local_thread_idx; d < params.head_dim; d += tg_dim.x) {
        float sum = 0.0f;
        // Sum contributions from all threads for this dimension
        for (uint t = 0; t < tg_dim.x; ++t) {
            sum += tg_thread_o_accumulator[t * params.head_dim + d];
        }
        tg_final_o_accumulator[d] = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // --- Final Normalization & Write Output ---
    float S_global = tg_global_stats[1];
    float inv_S_global = (S_global > kEpsilonForZeroGuard) ? (1.0f / S_global) : 0.0f;

    // Calculate output index in final buffer
    // Assuming output layout: [num_tokens][num_q_heads][head_dim]
    uint output_base_idx = target_q_token_idx * params.num_q_heads * params.head_dim +
                          target_q_head_idx * params.head_dim;

    // Cooperatively write final normalized output
    for (uint d = local_thread_idx; d < params.head_dim; d += tg_dim.x) {
        final_output_buffer[output_base_idx + d] = (half)(tg_final_o_accumulator[d] * inv_S_global);
    }

} // End of kernel