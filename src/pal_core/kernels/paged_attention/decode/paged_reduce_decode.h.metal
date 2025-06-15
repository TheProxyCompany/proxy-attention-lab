// paged_reduce_decode.h.metal
// Second-pass reduction kernel for partitioned paged attention.
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
#include "../utils.h.metal"

using namespace metal;

template <typename T, int head_dim, int SIMD_WIDTH>
[[kernel]] void pal_paged_reduce_decode(
    device T*            output_buffer          [[buffer(0)]],
    device const float*  max_logits_in          [[buffer(1)]],
    device const float*  exp_sums_in            [[buffer(2)]],
    device const T*      tmp_in                 [[buffer(3)]],
    device const int*    context_lens_in        [[buffer(4)]],

    constant const PagedAttentionParams& params [[buffer(5)]],
    threadgroup uchar*   tg_mem                 [[threadgroup(0)]],
    uint3                threads_per_threadgroup [[threads_per_threadgroup]],
    uint3                tg_pos_in_grid         [[threadgroup_position_in_grid]],
    uint                 local_idx_in_tg        [[thread_index_in_threadgroup]],
    uint                 simdgroup_idx          [[simdgroup_index_in_threadgroup]],
    uint                 lane_idx               [[thread_index_in_simdgroup]]
) {
    const int seq_idx = tg_pos_in_grid.x;
    const int head_idx = tg_pos_in_grid.y;
    const int num_threads = threads_per_threadgroup.x;
    const int num_simd_groups = num_threads / SIMD_WIDTH;

    const int context_len = context_lens_in[seq_idx];
    const int num_chunks = (context_len + CHUNK_SIZE - 1) / CHUNK_SIZE;

    const int max_tokens = params.max_logical_pages_per_seq * params.tokens_per_page;
    const int max_num_chunks = (max_tokens + CHUNK_SIZE - 1) / CHUNK_SIZE;

    // Early exit if only one chunk: just copy tmp_out to out
    if (num_chunks <= 1) {
        device T* out_ptr = output_buffer +
                            (seq_idx * params.num_q_heads * head_dim) +
                            (head_idx * head_dim);
        const device T* tmp_in_ptr = tmp_in +
                                     ((ulong)seq_idx * params.num_q_heads * max_num_chunks * head_dim) +
                                     (head_idx * max_num_chunks * head_dim);
        #pragma unroll
        for (int i = local_idx_in_tg; i < head_dim; i += num_threads) {
            out_ptr[i] = tmp_in_ptr[i];
        }
        return;
    }

    // --- Threadgroup Memory Layout ---
    threadgroup uchar* current_mem_ptr = tg_mem;
    threadgroup float* shared_max_logits = (threadgroup float*)current_mem_ptr;
    current_mem_ptr += max_num_chunks * sizeof(float);

    threadgroup float* shared_exp_sums = (threadgroup float*)current_mem_ptr;
    current_mem_ptr += max_num_chunks * sizeof(float);

    threadgroup float* reduction_scratch = (threadgroup float*)current_mem_ptr;

    // --- 1. Find Global Max Logit ---
    const device float* max_logits_ptr = max_logits_in +
        (ulong)seq_idx * params.num_q_heads * max_num_chunks +
        (ulong)head_idx * max_num_chunks;

    float thread_max_logit = -INFINITY;
    for (int i = local_idx_in_tg; i < num_chunks; i += num_threads) {
        const float l = max_logits_ptr[i];
        shared_max_logits[i] = l;
        thread_max_logit = max(thread_max_logit, l);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float global_max_logit = page_max(reduction_scratch, thread_max_logit, simdgroup_idx, lane_idx, num_simd_groups, SIMD_WIDTH);

    // --- 2. Calculate Global Exp Sum ---
    const device float* exp_sums_ptr = exp_sums_in +
        (ulong)seq_idx * params.num_q_heads * max_num_chunks +
        (ulong)head_idx * max_num_chunks;

    float thread_exp_sum = 0.0f;
    for (int i = local_idx_in_tg; i < num_chunks; i += num_threads) {
        float l = shared_max_logits[i];
        float rescaled_exp_sum = exp_sums_ptr[i] * exp(max(l - global_max_logit, params.log_exp_min_clamp));
        thread_exp_sum += rescaled_exp_sum;
        shared_exp_sums[i] = rescaled_exp_sum;
    }

    float global_exp_sum = page_sum(reduction_scratch, thread_exp_sum, simdgroup_idx, lane_idx, num_simd_groups, SIMD_WIDTH);
    const float inv_global_exp_sum = 1.0f / (global_exp_sum + 1e-6f);

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // --- 3. Aggregate Final Output ---
    const device T* tmp_base_ptr = tmp_in +
                                  ((ulong)seq_idx * params.num_q_heads + head_idx) *
                                  max_num_chunks * head_dim;

    device T* out_ptr = output_buffer +
                   (seq_idx * params.num_q_heads * head_dim) +
                   (head_idx * head_dim);
    #pragma unroll
    for (int i = local_idx_in_tg; i < head_dim; i += num_threads) {
        float acc = 0.0f;
        #pragma unroll
        for (int j = 0; j < num_chunks; ++j) {
            acc += float(tmp_base_ptr[j * head_dim + i]) * shared_exp_sums[j];
        }
        out_ptr[i] = T(acc * inv_global_exp_sum);
    }
}
