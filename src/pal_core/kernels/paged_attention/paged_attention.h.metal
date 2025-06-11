// paged_attention.h.metal
// Metal shader header for paged attention operations with tiled V accumulation.
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

constant bool USE_TWO_PASS [[function_constant(0)]];

template <typename T, int head_dim, int CHUNK_SIZE>
[[kernel]] void pal_paged_attention(
    device const T*      queries_in             [[buffer(0)]],
    device const T*      k_cache_pool_in        [[buffer(1)]],
    device const T*      v_cache_pool_in        [[buffer(2)]],
    device const uint*   page_table_in          [[buffer(3)]],
    device const int*    context_lens_in        [[buffer(4)]],
    device T*            output_buffer          [[buffer(5)]],

    // --- Intermediate Buffers for Two-Pass ---
    device float*        max_logits_out         [[buffer(6), function_constant(USE_TWO_PASS)]],
    device float*        exp_sums_out           [[buffer(7), function_constant(USE_TWO_PASS)]],
    device T*            tmp_out                [[buffer(8), function_constant(USE_TWO_PASS)]],

    constant const PagedAttentionParams& params [[buffer(9)]],
    threadgroup uchar*   tg_mem                 [[threadgroup(0)]],
    uint3                tg_pos_in_grid         [[threadgroup_position_in_grid]],
    uint                 local_idx_in_tg        [[thread_index_in_threadgroup]]
) {
    // --- 1. Identification ---
    uint seq_idx = tg_pos_in_grid.x;
    uint q_head_idx = tg_pos_in_grid.y;
    const int num_threads = 256;

    // Determine the KV head this Q head should attend to.
    const int num_q_per_kv = params.num_q_heads / params.num_kv_heads;
    const int kv_head_id = q_head_idx / num_q_per_kv;
    const uint head_dim_vec4 = head_dim / 4;

    // simd group and lane indices
    const int simd_width = params.simd_width;
    const int num_simd_groups = num_threads / simd_width;
    const int simdgroup_idx = local_idx_in_tg / simd_width;
    const int lane_idx = local_idx_in_tg % simd_width;

    // --- 2. Partition Threadgroup Memory ---
    // The threadgroup memory is partitioned into 5 sections:
    // 1. Q tile (head_dim * sizeof(float))
    // 2. K tile (params.tokens_per_page * head_dim * sizeof(T))
    // 3. V tile (params.tokens_per_page * head_dim * sizeof(T))
    // 4. Page table slice (params.max_logical_pages_per_seq * sizeof(uint))
    // 5. Reduction scratch (num_simd_groups * 2 * sizeof(float)) + acc_tile
    // helper – works at compile time

    // --- 2. Partition Threadgroup Memory ---
    threadgroup uchar* current_mem_ptr = tg_mem;

    threadgroup float* q_tile = (threadgroup float*)current_mem_ptr;
    current_mem_ptr = (threadgroup uchar*)ALIGN16(current_mem_ptr + (head_dim * sizeof(float)));

    threadgroup T* k_tile = (threadgroup T*)current_mem_ptr;
    current_mem_ptr = (threadgroup uchar*)ALIGN16(current_mem_ptr + (params.tokens_per_page * head_dim * sizeof(T)));

    threadgroup T* v_tile = (threadgroup T*)current_mem_ptr;
    current_mem_ptr = (threadgroup uchar*)ALIGN16(current_mem_ptr + (params.tokens_per_page * head_dim * sizeof(T)));

    threadgroup float* reduction_scratch = (threadgroup float*)current_mem_ptr;
    current_mem_ptr = (threadgroup uchar*)ALIGN16(current_mem_ptr + (params.tokens_per_page * sizeof(float)));

    threadgroup float* simd_max_scores = (threadgroup float*)current_mem_ptr;
    current_mem_ptr = (threadgroup uchar*)ALIGN16(current_mem_ptr + (num_simd_groups * sizeof(float)));

    threadgroup float* simd_sum_exps = (threadgroup float*)current_mem_ptr;
    current_mem_ptr = (threadgroup uchar*)ALIGN16(current_mem_ptr + (num_simd_groups * sizeof(float)));

    threadgroup float* acc_tile = (threadgroup float*)current_mem_ptr;
    current_mem_ptr = (threadgroup uchar*)ALIGN16(current_mem_ptr + (num_simd_groups * head_dim * sizeof(float)));

    // --- 3. Load the Q Vector ---
    device const T* q_ptr = queries_in +
                            (seq_idx * params.num_q_heads * head_dim) +
                            (q_head_idx * head_dim);

    // Cooperatively load the Q vector into shared memory
    // Convert to float for precision and apply the scaling factor
    using Vec4 = typename Vec<T, 4>::Type;
    for (uint i = local_idx_in_tg; i < head_dim_vec4; i += num_threads) {
        Vec4 q_vec = ((device const Vec4*)q_ptr)[i];
        float4 q_chunk = float4(q_vec);
        ((threadgroup float4*)q_tile)[i] = q_chunk * params.inv_sqrt_head_dim;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // THREADGROUP BARRIER REASON: Ensure the full Q-vector is in shared memory before use.

    // --- 4. Initialize Accumulators & Get Sequence Info ---
    #pragma unroll
    for(uint i = local_idx_in_tg; i < head_dim_vec4; i += num_threads) {
        ((threadgroup float4*)acc_tile)[i] = 0.0f;
    }

    // Online softmax statistics, held in registers.
    float max_score = -INFINITY;
    float local_max = -INFINITY;
    float sum_exp = 0.0f;

    // Get the total number of tokens to process for this sequence.
    const int context_len = context_lens_in[seq_idx];

    // Determine the range of blocks this threadgroup is responsible for.
    const int chunk_idx = tg_pos_in_grid.z;
    const int start_block_idx = USE_TWO_PASS ? (chunk_idx * CHUNK_SIZE) / params.tokens_per_page : 0;
    const int num_context_blocks = (context_len + params.tokens_per_page - 1) / params.tokens_per_page;
    int end_block_idx;
    if (USE_TWO_PASS) {
        int chunk_size_per_page = start_block_idx + (CHUNK_SIZE / params.tokens_per_page);
        end_block_idx = MIN(chunk_size_per_page, num_context_blocks);
    } else {
        end_block_idx = num_context_blocks;
    }

    // --- 5. Attention Loop ---
    // 5.a Define Parallelism Hierarchy ---
    const int subgroup_size = MAX(simd_width / params.tokens_per_page, 1);
    const int num_subgroups_per_simd = simd_width / subgroup_size;
    const int subgroup_idx = lane_idx / subgroup_size;
    const int subgroup_lane_offset = lane_idx % subgroup_size;

    // Strides in elements
    const ulong page_stride = (ulong)params.num_kv_heads * params.tokens_per_page * head_dim;
    const ulong head_stride = (ulong)params.tokens_per_page * head_dim;

    // --- 5.b Main Attention Loop ---
    for (
        int block_idx = start_block_idx + simdgroup_idx;
        block_idx < end_block_idx;
        block_idx += num_simd_groups
    ) {
        // --- Load K & V Tile for this Block ---
        uint physical_page_id = page_table_in[seq_idx * params.max_logical_pages_per_seq + block_idx];
        ulong base_element_offset = (ulong)physical_page_id * page_stride + (ulong)kv_head_id * head_stride;
        // Get direct pointers to the start of the data for this (page, head)
        device const T* k_page_ptr = k_cache_pool_in + base_element_offset;
        device const T* v_page_ptr = v_cache_pool_in + base_element_offset;

        for (uint i = local_idx_in_tg; i < params.tokens_per_page * head_dim_vec4; i += num_threads) {
            // layout [pages, kv_heads, tokens, head_dim],
            // all tokens for this KV head are contiguous
            ((threadgroup Vec4*)k_tile)[i] = ((device const Vec4*)k_page_ptr)[i];
            ((threadgroup Vec4*)v_tile)[i] = ((device const Vec4*)v_page_ptr)[i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // THREADGROUP BARRIER REASON: Ensure K/V tile for this block is fully loaded.

        // --- Compute Scores for all Tokens in the Tile ---
        // Use reduction_scratch as a temporary tile for scores.
        threadgroup float* logits_tile = reduction_scratch;
        // reset local_max
        local_max = -INFINITY;

        // Each subgroup computes the score for one token in the tile.
        const int iterations = (params.tokens_per_page + num_subgroups_per_simd - 1) / num_subgroups_per_simd;
        for (int i = 0; i < iterations; ++i) {
            const int token_in_page = subgroup_idx + i * num_subgroups_per_simd;
            if (token_in_page >= static_cast<int>(params.tokens_per_page)) continue;  // Tail guard

            const int key_token_pos = block_idx * params.tokens_per_page + token_in_page;

            float score = 0.0f;
            if (key_token_pos < context_len) {
                threadgroup const T* k_vec_in_tile = k_tile + token_in_page * head_dim;

                threadgroup const float4* q_vecs = (threadgroup const float4*)q_tile;
                threadgroup const float4* k_vecs = (threadgroup const float4*)k_vec_in_tile;

                score = qk_dot<float4>(q_vecs, k_vecs, head_dim_vec4, subgroup_size);

                local_max = max(local_max, score);
            }

            // The leader of the subgroup writes the final score to the shared logits tile.
            if (subgroup_lane_offset == 0) {
                logits_tile[token_in_page] = score;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // THREADGROUP BARRIER REASON: Ensure all scores for the tile are in shared memory.

        // --- Update Online Softmax Statistics ---
        float tile_max = block_max(simd_max_scores, local_max, simdgroup_idx, lane_idx, num_simd_groups, simd_width);

        // --- Online Softmax Bookkeeping ---
        // When max changes, we must rescale the running sum to maintain correctness
        float prev_max = max_score;
        float new_max = max(prev_max, tile_max);

        // Rescale factors
        float prev_scale = exp(max(prev_max - new_max, params.log_exp_min_clamp));  // scale for accumulated sum
        float tile_scale = exp(max(tile_max - new_max, params.log_exp_min_clamp));  // scale for this tile

        // Update running statistics
        sum_exp *= prev_scale;  // rescale accumulated sum to new max
        max_score = new_max;    // update global max

        // Rescale the accumulator to the new max
        if (prev_scale != 1.0f) {
            threadgroup float* simd_acc_tile = acc_tile + simdgroup_idx * head_dim;
            #pragma unroll
            for (uint v = lane_idx; v < head_dim_vec4; v += simd_width) {
                ((threadgroup float4*)simd_acc_tile)[v] *= prev_scale;
            }
        }

       // --- Compute Softmax Weights ---
        // Parallel exp computation and sum reduction
        float local_sum = 0.0f;
        #pragma unroll
        for (int token_in_page = lane_idx; token_in_page < params.tokens_per_page; token_in_page += simd_width) {
            const int key_token_pos = block_idx * params.tokens_per_page + token_in_page;
            if (key_token_pos < context_len) {
                float score = logits_tile[token_in_page];
                float exp_arg = max(score - new_max, params.log_exp_min_clamp); // CHANGE: use new_max, not tile_max!
                float exp_score = exp(exp_arg);
                logits_tile[token_in_page] = exp_score;
                local_sum += exp_score;
            } else {
                logits_tile[token_in_page] = 0.0f;
            }
        }
        float tile_sum = block_sum(simd_sum_exps, local_sum, simdgroup_idx, lane_idx, num_simd_groups, simd_width);
        sum_exp += tile_sum; // Add tile's contribution to running sum

        // --- V Accumulation ---
        // Each SIMD group accumulates its portion of the output vector
        threadgroup float* simd_acc_tile = acc_tile + simdgroup_idx * head_dim;

        // Process all tokens in the page
        for (int token_in_page = 0; token_in_page < params.tokens_per_page; ++token_in_page) {
            const int key_token_pos = block_idx * params.tokens_per_page + token_in_page;
            if (key_token_pos < context_len) {
                float weight = logits_tile[token_in_page];
                threadgroup const T* v_vec_in_tile = v_tile + token_in_page * head_dim;

                // Each thread accumulates its portion of the V vector
                for (uint v = lane_idx; v < head_dim_vec4; v += simd_width) {
                    Vec4 v_chunk = ((threadgroup const Vec4*)v_vec_in_tile)[v];
                    float4 v_float = float4(v_chunk);
                    float4 weighted_v = v_float * weight;
                    ((threadgroup float4*)simd_acc_tile)[v] += weighted_v;
                }
            }
        }
    } // end of main attention loop

    // --- Cross-SIMD Group Reduction ---
    // Step 1: Store per-SIMD group statistics to shared memory
    if (lane_idx == 0) {
        simd_max_scores[simdgroup_idx] = max_score;
        simd_sum_exps[simdgroup_idx] = sum_exp;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // THREADGROUP BARRIER REASON: Ensure all threads have computed their local sums.

    // Step 2: Compute global max across all SIMD groups
    float final_max_score = (simdgroup_idx < num_simd_groups) ? simd_max_scores[simdgroup_idx] : -INFINITY;
    final_max_score = block_max(simd_max_scores, final_max_score, simdgroup_idx, lane_idx, num_simd_groups, simd_width);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // THREADGROUP BARRIER REASON: Ensure we have the final max score.

    // Step 3: Rescale exp sums and compute global sum
    float rescaled_sum_exp = 0.0f;
    if (lane_idx == 0) {
        float local_max = simd_max_scores[simdgroup_idx];
        float local_sum = simd_sum_exps[simdgroup_idx];
        rescaled_sum_exp = local_sum * exp(max(local_max - final_max_score, params.log_exp_min_clamp));
        simd_sum_exps[simdgroup_idx] = rescaled_sum_exp;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // THREADGROUP BARRIER REASON: Ensure all threads have rescaled their exp sums.

    // Step 4: Compute global sum of rescaled exp values
    float final_sum_exp = (simdgroup_idx < num_simd_groups) ? simd_sum_exps[simdgroup_idx] : 0.0f;
    final_sum_exp = block_sum(simd_sum_exps, final_sum_exp, simdgroup_idx, lane_idx, num_simd_groups, simd_width);

    // Step 5: Rescale and combine V accumulators from all SIMD groups
    // Each SIMD group's accumulator needs to be rescaled by exp(local_max - final_max)
    float rescale_factor = 0.0f;
    if (lane_idx == 0) {
        rescale_factor = exp(max(max_score - final_max_score, params.log_exp_min_clamp));
        simd_max_scores[simdgroup_idx] = rescale_factor; // Reuse array for broadcast
    }
    rescale_factor = simd_max_scores[simdgroup_idx];

    // First, rescale this SIMD group's accumulator
    for (uint v = lane_idx; v < head_dim_vec4; v += simd_width) {
        threadgroup float* simd_acc_tile = acc_tile + simdgroup_idx * head_dim;
        ((threadgroup float4*)simd_acc_tile)[v] *= rescale_factor;
    }

    // Combine all SIMD group accumulators into the first one (acc_tile[0])
    if (simdgroup_idx == 0) {
        for (int sg = 1; sg < num_simd_groups; ++sg) {
            threadgroup float* other_acc = acc_tile + sg * head_dim;
            for (uint v = lane_idx; v < head_dim_vec4; v += simd_width) {
                ((threadgroup float4*)acc_tile)[v] += ((threadgroup float4*)other_acc)[v];
            }
        }
    }

    // --- 6. Finalization and Output ---
    if (USE_TWO_PASS) {
        // --- Two-Pass (Pass 1) Finalization ---
        // Calculate num_chunks from template parameter
        const int max_tokens = params.max_logical_pages_per_seq * params.tokens_per_page;
        const int num_chunks = (max_tokens + CHUNK_SIZE - 1) / CHUNK_SIZE;

        const int out_idx = (seq_idx * params.num_q_heads * num_chunks) +
                            (q_head_idx * num_chunks) +
                            chunk_idx;

        // Only one thread needs to write the scalar stats.
        if (local_idx_in_tg == 0) {
            max_logits_out[out_idx] = final_max_score; // From cross-team reduction
            exp_sums_out[out_idx] = final_sum_exp;     // From cross-team reduction
        }

        // All threads cooperate to write the partial, unnormalized accumulator.
        device T* tmp_out_ptr = tmp_out + out_idx * head_dim;
        for (uint i = local_idx_in_tg; i < head_dim_vec4; i += num_threads) {
            float4 partial_chunk = ((threadgroup float4*)acc_tile)[i];
            Vec4 result;
            from_float(result, partial_chunk);
            ((device Vec4*)tmp_out_ptr)[i] = result;
        }
    } else {
        // --- Single-Pass Finalization ---
        // 1. Final Normalization
        const float inv_sum_exp = 1.0f / (final_sum_exp + 1e-6f);

        // All threads cooperate to normalize the final accumulator in place.
        for (uint i = local_idx_in_tg; i < head_dim_vec4; i += num_threads) {
            ((threadgroup float4*)acc_tile)[i] *= inv_sum_exp;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // THREADGROUP BARRIER REASON: Ensure all threads finish normalization before writing.

        // 2. Final Write
        // All threads cooperate to write the final, normalized result.
        device T* out_ptr = output_buffer +
                        (seq_idx * params.num_q_heads * head_dim) +
                        (q_head_idx * head_dim);

        for (uint i = local_idx_in_tg; i < head_dim_vec4; i += num_threads) {
            float4 final_chunk = ((threadgroup float4*)acc_tile)[i];
            Vec4 result;
            from_float(result, final_chunk);
            ((device Vec4*)out_ptr)[i] = result;
        }
    }

} // end of pal_paged_attention
