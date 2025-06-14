// utils.h.metal
// Metal shader header for PAL utilities.
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

#include "../metal/metal_types.h.metal"

// ============================================================================
// Paged Attention-specific utilities
// inspired/borrowed from:
// https://github.com/EricLBuehler/mistral.rs/blob/58df07e2abb758f7c1d4de8f26f24803d7dbee1f/mistralrs-paged-attn/src/metal/kernels/pagedattention.metal
// ============================================================================

// Dot product for threadgroup Q and thread-local K arrays
template <typename QVec, typename KVec>
inline float qk_dot_strided(
    threadgroup const QVec* q_base,
    thread const KVec* k_vecs,
    int num_vecs,
    int lane_offset,
    int subgroup_size
) {
    using AccType = typename FloatVec<QVec>::Type;

    // First vector at q_base[lane_offset * num_vecs + 0]
    AccType qk_vec = mul<AccType, QVec, KVec>(
        q_base[lane_offset * num_vecs],
        k_vecs[0]
    );

    #pragma unroll
    for (int i = 1; i < num_vecs; ++i) {
        qk_vec = fma(
            q_base[lane_offset * num_vecs + i],
            k_vecs[i],
            qk_vec
        );
    }

    float qk = sum(qk_vec);
    return simd_sum(qk, subgroup_size);
}

// page-wide sum for softmax computation
// happens within a SIMD group
inline float page_sum(
    threadgroup float* tg_mem,
    float local_sum,
    uint simd_group_id,
    uint simd_lane_id,
    uint num_simd_groups,
    uint simd_width
) {
    // First reduce within each SIMD group
    local_sum = simd_sum(local_sum, simd_width);

    // SIMD group leaders write to shared memory
    if (simd_lane_id == 0) {
        tg_mem[simd_group_id] = local_sum;
    }

    // Synchronize to ensure all groups have written
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Get the local sum from the shared memory
    local_sum = (simd_lane_id < num_simd_groups) ? tg_mem[simd_lane_id] : 0.0f;

    // Reduce across the SIMD groups
    local_sum = simd_sum(local_sum, num_simd_groups);

    // Broadcast result to all threads
    return simd_broadcast(local_sum, 0);
}

// page-wide max for softmax computation
// happens within a SIMD group
inline float page_max(
    threadgroup float* tg_mem,
    float local_max,
    uint simd_group_id,
    uint simd_lane_id,
    uint num_simd_groups,
    uint simd_width,
    uint subgroup_size = 1
) {
    // First reduce within each SIMD group (simd_width) + (subgroup_size)
    local_max = simd_max(local_max, simd_width, subgroup_size);

    // SIMD group leaders write to shared memory
    if (simd_lane_id == 0) {
        tg_mem[simd_group_id] = local_max;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // THREADGROUP BARRIER REASON: Ensure all groups have written their local_max to shared memory before reduction.

    // Get the local max from the shared memory
    local_max = (simd_lane_id < num_simd_groups) ? tg_mem[simd_lane_id] : -INFINITY;

    // Reduce across the SIMD groups (num_simd_groups)
    local_max = simd_max(local_max, num_simd_groups);

    // Broadcast result to all threads
    return simd_broadcast(local_max, 0);
}
