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

template <typename T, int head_dim, int CHUNK_SIZE>
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
    const int simd_width = params.simd_width;
    const int num_simd_groups = num_threads / simd_width;

    const int context_len = context_lens_in[seq_idx];
    const int num_chunks = (context_len + CHUNK_SIZE - 1) / CHUNK_SIZE;

    // Calculate max_num_chunks from params
    const int max_tokens = params.max_logical_pages_per_seq * params.tokens_per_page;
    const int max_num_chunks = (max_tokens + CHUNK_SIZE - 1) / CHUNK_SIZE;

    // Workspace allocation - sequential layout to avoid collisions
    threadgroup uchar* mem = tg_mem;

    threadgroup float* shared_max_logits = (threadgroup float*)mem;
    mem += max_num_chunks * sizeof(float);

    threadgroup float* shared_exp_sums = (threadgroup float*)mem;
    mem += max_num_chunks * sizeof(float);

    threadgroup float* red_smem = (threadgroup float*)mem;
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

    // Reduce within SIMD group
    max_logit = simd_max(max_logit, simd_width);
    if (lane_idx == 0) {
        red_smem[simdgroup_idx] = max_logit;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Reduce across SIMD groups
    if (simdgroup_idx == 0) {
        max_logit = lane_idx < num_simd_groups ? red_smem[lane_idx] : -INFINITY;
        max_logit = simd_max(max_logit, simd_width);
    }
    // Broadcast to all threads
    if (simdgroup_idx == 0 && lane_idx == 0) {
        red_smem[0] = max_logit;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    max_logit = red_smem[0];

    // Load rescaled exp sums to shared memory
    const device float* exp_sums_ptr = exp_sums_in +
        seq_idx * params.num_q_heads * max_num_chunks +
        head_idx * max_num_chunks;

    float global_exp_sum = 0.0f;
    for (int i = local_idx_in_tg; i < num_chunks; i += num_threads) {
        float l = shared_max_logits[i];
        float rescaled_exp_sum = exp_sums_ptr[i] * exp(l - max_logit);
        global_exp_sum += rescaled_exp_sum;
        shared_exp_sums[i] = rescaled_exp_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Reduce global_exp_sum using templated SIMD helper
    global_exp_sum = simd_sum(global_exp_sum, simd_width);
    if (lane_idx == 0) {
        red_smem[simdgroup_idx + num_simd_groups] = global_exp_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simdgroup_idx == 0) {
        global_exp_sum = lane_idx < num_simd_groups ? red_smem[lane_idx + num_simd_groups] : 0.0f;
        global_exp_sum = simd_sum(global_exp_sum, simd_width);
    }
    if (simdgroup_idx == 0 && lane_idx == 0) {
        red_smem[1] = global_exp_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    global_exp_sum = red_smem[1];

    const float inv_global_exp_sum = 1.0f / (global_exp_sum + 1e-6f);

    // Aggregate tmp_out to out
    const device T* tmp_out_ptr = tmp_in +
        seq_idx * params.num_q_heads * max_num_chunks * head_dim +
        head_idx * max_num_chunks * head_dim;
    device T* out_ptr = output_buffer +
        seq_idx * params.num_q_heads * head_dim +
        head_idx * head_dim;

    using Vec4 = typename Vec<T, 4>::Type;
    const uint head_dim_vec4 = head_dim / 4;

    for (uint i = local_idx_in_tg; i < head_dim_vec4; i += num_threads) {
        float4 acc = {0.0f, 0.0f, 0.0f, 0.0f};
        for (int j = 0; j < num_chunks; ++j) {
            Vec4 tmp_chunk = ((device const Vec4*)(tmp_out_ptr + j * head_dim))[i];
            acc += to_float4(tmp_chunk) * shared_exp_sums[j];
        }
        ((device Vec4*)out_ptr)[i] = from_float4<T>(acc * inv_global_exp_sum);
    }
}
