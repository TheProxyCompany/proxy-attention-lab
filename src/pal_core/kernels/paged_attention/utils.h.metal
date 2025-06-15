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


/**
 * @brief Computes the dot product between a vector in threadgroup memory
 *        and a vector in thread-private registers.
 *
 * This is the core computation for the prefill kernel. Each thread calculates
 * a partial sum, and then a fast, parallel reduction sums these partials
 * into a single final score.
 *
 * @param q_vector Pointer to the start of the Q vector in threadgroup memory.
 * @param k_vector   The K vector, held in per-thread private registers.
 * @param scratchpad  A pointer to a block of threadgroup memory of size
 *                    `tg_dim.x` for the reduction.
 * @return The final, single float value of the dot product.
 */
template <typename T, int HEAD_DIM>
inline float dot_product_shmem_reg(
    threadgroup const T* q_vector,
    thread const T* k_vector,
    threadgroup float* scratchpad,
    uint local_idx_in_tg,
    uint tg_dim_x
) {
    // 1. Parallel Partial Summation
    float partial_score = 0.0f;
    #pragma unroll
    for (int i = local_idx_in_tg; i < HEAD_DIM; i += tg_dim_x) {
        partial_score += (float)q_vector[i] * (float)k_vector[i];
    }

    // 2. Parallel Reduction
    scratchpad[local_idx_in_tg] = partial_score;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // THREADGROUP BARRIER REASON: Ensure all threads have the updated partial_score before the next step.

    // Iteratively reduce the sums in parallel.
    for (uint s = tg_dim_x / 2; s > 0; s >>= 1) {
        if (local_idx_in_tg < s) {
            scratchpad[local_idx_in_tg] += scratchpad[local_idx_in_tg + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // THREADGROUP BARRIER REASON: Ensure all threads have the updated partial_score before the next step.
    }

    // The final result is in scratchpad[0]. All threads return this value.
    return scratchpad[0];
}


/**
 * @brief Updates the online softmax statistics and the output accumulator
 *        for a single query within a tile.
 *
 * This function encapsulates the core logic of the "flash" attention mechanism.
 * It's called by every thread, but a `local_idx_in_tg == 0` check ensures
 * the statistics are only updated once. The subsequent V-aggregation is parallel.
 *
 * @param score The newly computed attention score for this (Q, K) pair.
 * @param v_vector The V vector for the current key, held in registers.
 * @param softmax_stats Pointer to the start of the [max_score, sum_exp] pair
 *                  for this query in threadgroup memory.
 * @param accumulator_ptr Pointer to the start of the output accumulator vector
 *                        for this query in threadgroup memory.
 */
template <typename T, int HEAD_DIM>
inline void update_attention_tile(
    float score,
    thread const T* v_vector,
    threadgroup float* softmax_stats,
    threadgroup float* accumulator,
    threadgroup float* scale_factor_broadcast,
    uint local_idx_in_tg,
    uint tg_dim_x,
    float log_exp_min_clamp
) {
    float new_max_score = -INFINITY;
    if (local_idx_in_tg == 0) {
        float old_max_score = softmax_stats[0];
        new_max_score = max(old_max_score, score);
        float scale_factor = 1.0f;

        // If the max score changed, calculate a rescaling factor.
        if (new_max_score > old_max_score) {
            scale_factor = exp(max(old_max_score - new_max_score, log_exp_min_clamp));
            softmax_stats[1] *= scale_factor; // Rescale sum_exp
        }
        softmax_stats[0] = new_max_score;
        *scale_factor_broadcast = scale_factor; // Use scratchpad to broadcast scale factor
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // THREADGROUP BARRIER REASON: Ensure all threads have the updated stats before the next step.

    // All threads read the broadcasted scale factor and updated max_score
    float scale = *scale_factor_broadcast;
    new_max_score = softmax_stats[0];

    if (scale != 1.0f) {
        #pragma unroll
        for (int i = local_idx_in_tg; i < HEAD_DIM; i += tg_dim_x) {
            accumulator[i] *= scale;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // THREADGROUP BARRIER REASON: Ensure all threads have the updated accumulator before the next step.

    // V-Aggregation
    float prob = exp(max(score - new_max_score, log_exp_min_clamp));
    #pragma unroll
    for (int i = local_idx_in_tg; i < HEAD_DIM; i += tg_dim_x) {
        accumulator[i] += prob * (float)v_vector[i];
    }

    if (local_idx_in_tg == 0) {
        softmax_stats[1] += prob;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // THREADGROUP BARRIER REASON: Ensure all threads have the updated accumulator before the next step.
}
