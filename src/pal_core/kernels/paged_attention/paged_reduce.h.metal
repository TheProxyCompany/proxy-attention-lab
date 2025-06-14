// paged_reduce.h.metal
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
#include "pal_types.h.metal"

using namespace metal;

template <typename T, int head_dim>
[[kernel]] void pal_paged_reduce(
    device T*            output_buffer          [[buffer(0)]],
    device const float*  max_logits_in          [[buffer(1)]],
    device const float*  exp_sums_in            [[buffer(2)]],
    device const T*      tmp_in                 [[buffer(3)]],
    device const int*    context_lens_in        [[buffer(4)]],

    constant const PagedAttentionParams& params [[buffer(5)]],
    threadgroup uchar*   tg_mem                 [[threadgroup(0)]],
    uint3                tg_pos_in_grid         [[threadgroup_position_in_grid]],
    uint                 local_idx_in_tg        [[thread_index_in_threadgroup]],
    uint                 simdgroup_idx          [[simdgroup_index_in_threadgroup]],
    uint                 lane_idx               [[thread_index_in_simdgroup]]
) {
    const int seq_idx = tg_pos_in_grid.x;
    const int head_idx = tg_pos_in_grid.y;
    const int num_threads = 256;
    const int num_simd_groups = num_threads / SIMD_WIDTH;

    const int context_len = context_lens_in[seq_idx];
    const int num_chunks = (context_len + CHUNK_SIZE - 1) / CHUNK_SIZE;

    // Calculate max_num_chunks from params
    const int max_tokens = params.max_logical_pages_per_seq * params.tokens_per_page;
    const int max_num_chunks = (max_tokens + CHUNK_SIZE - 1) / CHUNK_SIZE;

    // Early exit if only one chunk
    if (num_chunks == 1) {
        // Just copy tmp_out to out
        device T* out_ptr = output_buffer +
                            (seq_idx * params.num_q_heads * head_dim) +
                            (head_idx * head_dim);
        const device T* tmp_in_ptr = tmp_in +
                                    ((seq_idx * params.num_q_heads * max_num_chunks) +
                                    (head_idx * max_num_chunks)) * head_dim;

        for (int i = local_idx_in_tg; i < head_dim; i += num_threads) {
            out_ptr[i] = tmp_in_ptr[i];
        }
        return;
    }

    // Workspace allocation - sequential layout to avoid collisions
    threadgroup uchar* mem = tg_mem;

    threadgroup float* shared_max_logits = (threadgroup float*)mem;
    mem += num_chunks * sizeof(float);

    threadgroup float* shared_exp_sums = (threadgroup float*)mem;
    mem += num_chunks * sizeof(float);

    threadgroup float* reduction_scratch = (threadgroup float*)mem;
    mem += 2 * num_simd_groups * sizeof(float);

    // Load max logits to shared memory
    const device float* max_logits_ptr = max_logits_in +
        seq_idx * params.num_q_heads * max_num_chunks +
        head_idx * max_num_chunks;

    float max_logit = -INFINITY;
    for (int i = local_idx_in_tg; i < num_chunks; i += num_threads) {
        const float l = max_logits_ptr[i];
        shared_max_logits[i] = l;
        max_logit = max(max_logit, l);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // THREADGROUP BARRIER REASON: Ensure all threads have loaded their max_logits into shared memory before reduction.

    max_logit = page_max(
        reduction_scratch,
        max_logit,
        simdgroup_idx,
        lane_idx,
        num_simd_groups,
        SIMD_WIDTH
    );

    // Load rescaled exp sums to shared memory
    const device float* exp_sums_ptr = exp_sums_in +
        seq_idx * params.num_q_heads * max_num_chunks +
        head_idx * max_num_chunks;

    float global_exp_sum = 0.0f;
    for (int i = local_idx_in_tg; i < num_chunks; i += num_threads) {
        float l = shared_max_logits[i];
        float rescaled_exp_sum = exp_sums_ptr[i] * exp(max(l - max_logit, params.log_exp_min_clamp));
        global_exp_sum += rescaled_exp_sum;
        shared_exp_sums[i] = rescaled_exp_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // THREADGROUP BARRIER REASON: Ensure shared_exp_sums is fully populated and local global_exp_sum is calculated before reduction.

    global_exp_sum = page_sum(
        reduction_scratch,
        global_exp_sum,
        simdgroup_idx,
        lane_idx,
        num_simd_groups,
        SIMD_WIDTH
    );

    const float inv_global_exp_sum = 1.0f / (global_exp_sum + 1e-6f);

    // Aggregate tmp_out to out
    const int out_idx_base = (seq_idx * params.num_q_heads * max_num_chunks) +
                             (head_idx * max_num_chunks);
    const device T* tmp_base_ptr = tmp_in + out_idx_base * head_dim;

    device T* out_ptr = output_buffer +
                   (seq_idx * params.num_q_heads * head_dim) +
                   (head_idx * head_dim);


    using Vec4 = typename Vec<T, 4>::Type;

    for (uint i = local_idx_in_tg; i < head_dim / 4; i += num_threads) {
        float4 acc = {0.0f, 0.0f, 0.0f, 0.0f};
        for (int j = 0; j < num_chunks; ++j) {
            Vec4 tmp_chunk = ((device const Vec4*)(tmp_base_ptr + j * head_dim))[i];
            acc += float4(tmp_chunk) * shared_exp_sums[j];
        }
        Vec4 result;
        from_float(result, acc * inv_global_exp_sum);
        ((device Vec4*)out_ptr)[i] = result;
    }
}
