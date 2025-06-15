// paged_attention_prefill.h.metal
// Tiled, high-performance prefill kernel for paged attention.
//
// Copyright 2025 The Proxy Company. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may not use this file except in compliance with the License.
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

template <typename T, int HEAD_DIM, int Q_TILE_SIZE>
[[kernel]] void pal_paged_attention_prefill(
    // --- Input Buffers ---
    device const T*      q_prompt_in         [[buffer(0)]],
    device const T*      k_prompt_in         [[buffer(1)]],
    device const T*      v_prompt_in         [[buffer(2)]],
    device const T*      k_cache_paged_in    [[buffer(3)]],
    device const T*      v_cache_paged_in    [[buffer(4)]],
    device const uint*   page_table_in       [[buffer(5)]],
    device const int*    context_lens_in     [[buffer(6)]],

    // --- Parameters & Output ---
    constant PagedAttentionParams& params [[buffer(7)]],
    device T*            output_buffer      [[buffer(8)]],

    // --- Threadgroup Memory & Identifiers ---
    threadgroup uchar*   tg_mem               [[threadgroup(0)]],
    uint3                tg_dim               [[threads_per_threadgroup]],
    uint3                tg_pos_in_grid       [[threadgroup_position_in_grid]],
    uint                 local_idx_in_tg      [[thread_index_in_threadgroup]]
) {
    // ========================================================================
    // --- Phase 1: Initialization & Threadgroup Memory Partitioning ---
    // ========================================================================

    // Unpack threadgroup identifiers
    const uint q_block_idx = tg_pos_in_grid.x;
    const uint seq_idx = tg_pos_in_grid.y;
    const uint head_idx = tg_pos_in_grid.z;

    // Calculate number of new prompt tokens this threadgroup is responsible for.
    const int start_q_idx_global = q_block_idx * Q_TILE_SIZE;
    const int num_queries_in_tile = min((int)Q_TILE_SIZE, (int)params.num_prompt_tokens - start_q_idx_global);

    // Partition the single threadgroup memory buffer into logical sections.
    threadgroup uchar* current_mem_ptr = tg_mem;

    // 1. Query Tile: Stores the block of queries this threadgroup owns.
    threadgroup T* q_tile_shmem = (threadgroup T*)current_mem_ptr;
    current_mem_ptr += Q_TILE_SIZE * HEAD_DIM * sizeof(T);

    // 2. Output Accumulator: Stores the running weighted sum of values. Must be float32.
    threadgroup float* output_accumulator = (threadgroup float*)current_mem_ptr;
    current_mem_ptr += Q_TILE_SIZE * HEAD_DIM * sizeof(float);

    // 3. Softmax Stats: Stores running max_score and sum_exp for each query in the tile.
    threadgroup float* softmax_stats = (threadgroup float*)current_mem_ptr;
    current_mem_ptr += Q_TILE_SIZE * 2 * sizeof(float);

    // 4. Scale Factor Broadcast (1 float total)
    threadgroup float* scale_factor_broadcast = (threadgroup float*)current_mem_ptr;
    current_mem_ptr += sizeof(float);

    // 5. Reduction Scratchpad (1 float per thread)
    threadgroup float* reduction_scratchpad = (threadgroup float*)current_mem_ptr;
    current_mem_ptr += tg_dim.x * sizeof(float);

    // Initialize accumulators and stats to zero/negative infinity.
    // Each thread initializes a portion of the memory.
    #pragma unroll
    for (int i = local_idx_in_tg; i < (Q_TILE_SIZE * HEAD_DIM); i += tg_dim.x) {
        output_accumulator[i] = 0.0f;
    }
    #pragma unroll
    for (int i = local_idx_in_tg; i < (Q_TILE_SIZE * 2); i += tg_dim.x) {
        softmax_stats[i] = (i % 2 == 0) ? -INFINITY : 0.0f; // max_score = -inf, sum_exp = 0
    }

    // ========================================================================
    // --- Phase 2: Load Query Tile into Threadgroup Memory ---
    // ========================================================================
    // Base pointer to the start of the first query vector this threadgroup will load.
    device const T* q_prompt_src_base = q_prompt_in +
                                        (ulong)start_q_idx_global * params.num_q_heads * HEAD_DIM +
                                        (ulong)head_idx * HEAD_DIM;
    // Load the query tile into threadgroup memory.
    #pragma unroll
    for (int q_idx_in_tile = 0; q_idx_in_tile < num_queries_in_tile; ++q_idx_in_tile) {
        // Calculate the global token index for the current query in our tile.
        int global_q_idx = start_q_idx_global + q_idx_in_tile;
        // Calculate the source address for this specific query vector.
        device const T* src = q_prompt_in +
                            (ulong)global_q_idx * params.num_q_heads * HEAD_DIM +
                            (ulong)head_idx * HEAD_DIM;
        // Calculate the destination address in threadgroup memory.
        threadgroup T* dst = q_tile_shmem + q_idx_in_tile * HEAD_DIM;
        // Parallel copy for this single vector.
        #pragma unroll
        for (int i = local_idx_in_tg; i < HEAD_DIM; i += tg_dim.x) {
            dst[i] = src[i];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // THREADGROUP BARRIER REASON: Ensure the full Q-vector is in shared memory before use.

    // ========================================================================
    // --- Phase 3: Main Attention Loop (Streaming K/V) ---
    // ========================================================================
    const int history_len = context_lens_in[seq_idx];
    const int total_context_len = history_len + params.num_prompt_tokens;

    // Determine the correct KV head index for GQA
    const int num_q_per_kv = params.num_q_heads / params.num_kv_heads;
    const int kv_head_idx = head_idx / num_q_per_kv;

    // --- Part A: Attend to Prompt History (Causal) ---
    for (int kv_idx_prompt = 0; kv_idx_prompt < params.num_prompt_tokens; ++kv_idx_prompt) {
        // --- Step 3a: Load one K and one V vector from the prompt buffer ---
        // temporary registers for K and V
        T k_vec_reg[HEAD_DIM];
        T v_vec_reg[HEAD_DIM];

        // Calculate source pointers for this K/V vector
        device const T* k_src = k_prompt_in +
                            (ulong)kv_idx_prompt * params.num_kv_heads * HEAD_DIM +
                            (ulong)kv_head_idx * HEAD_DIM;
        device const T* v_src = v_prompt_in +
                            (ulong)kv_idx_prompt * params.num_kv_heads * HEAD_DIM +
                            (ulong)kv_head_idx * HEAD_DIM;

        // Parallel copy into registers
        #pragma unroll
        for (int i = local_idx_in_tg; i < HEAD_DIM; i += tg_dim.x) {
            k_vec_reg[i] = k_src[i];
            v_vec_reg[i] = v_src[i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (local_idx_in_tg == 0) {
            output_buffer[0] = (T)k_vec_reg[0];
            output_buffer[1] = (T)v_vec_reg[0];
            output_buffer[2] = (T)k_vec_reg[1];
        }
        return;

        // --- Step 3b & 3c: Compute Scores, Mask, and Update Accumulators ---
        // Loop through each of the queries in our tile.
        for (int q_idx_in_tile = 0; q_idx_in_tile < num_queries_in_tile; ++q_idx_in_tile) {
            int global_q_idx = start_q_idx_global + q_idx_in_tile;

            if (kv_idx_prompt > global_q_idx) {
                continue; // causal mask, skip this query
            }

             // Get a pointer to the specific query vector in our tile.
            threadgroup const T* q_vec_ptr = q_tile_shmem + q_idx_in_tile * HEAD_DIM;

            // Call our new helper to compute the dot product efficiently.
            float score = dot_product_shmem_reg<T, HEAD_DIM>(
                q_vec_ptr,
                k_vec_reg,
                reduction_scratchpad,
                local_idx_in_tg,
                tg_dim.x
            );

            // Get pointers to this specific query's statistics and accumulator.
            threadgroup float* stats_ptr = softmax_stats + q_idx_in_tile * 2;
            threadgroup float* accumulator_ptr = output_accumulator + q_idx_in_tile * HEAD_DIM;

            // Call our second helper to perform the online softmax update and V-aggregation.
            update_attention_tile<T, HEAD_DIM>(
                score,
                v_vec_reg,
                stats_ptr,
                accumulator_ptr,
                scale_factor_broadcast,
                local_idx_in_tg,
                tg_dim.x,
                params.log_exp_min_clamp
            );
        } // end of query loop
    } // end of new prompt token loop
    return;

    // ========================================================================
    // --- TEMPORARY DEBUG: Write the unnormalized accumulator to the output ---
    // ========================================================================

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Calculate the base address for this threadgroup's output
device T* out_ptr_base = output_buffer +
                       (ulong)start_q_idx_global * params.num_q_heads * HEAD_DIM +
                       (ulong)head_idx * HEAD_DIM;

// Each thread writes its portion of the accumulator tile
for (int q_idx_in_tile = 0; q_idx_in_tile < num_queries_in_tile; ++q_idx_in_tile) {

    device T* out_q_ptr = out_ptr_base + (ulong)q_idx_in_tile * params.num_q_heads * HEAD_DIM;
    threadgroup float* acc_q_ptr = output_accumulator + q_idx_in_tile * HEAD_DIM;

    for (int i = local_idx_in_tg; i < HEAD_DIM; i += tg_dim.x) {
        out_q_ptr[i] = (T)acc_q_ptr[i];
    }
}

return; // Stop here for the test


    // ========================================================================
    // --- Phase 4: Finalization and Output ---
    // ========================================================================

    threadgroup_barrier(mem_flags::mem_threadgroup); // Ensure all accumulations are done.

    // TODO: Implement the final normalization and write to global `output_buffer`.
    // - Normalize each of the 16 vectors in `output_accumulator` by its final `sum_exp`.
    // - Write the final 16 output vectors to the correct global memory locations.
}
