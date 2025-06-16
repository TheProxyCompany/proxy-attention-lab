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

template <typename T, int HEAD_DIM, int Q_TILE_SIZE, int SIMD_WIDTH>
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
    uint                 simdgroup_idx       [[simdgroup_index_in_threadgroup]],
    uint                 local_idx_in_tg     [[thread_index_in_threadgroup]]
) {
    // ========================================================================
    // --- Phase 1: Initialization & Threadgroup Memory Partitioning ---
    // ========================================================================
    // One SIMD-group (32 Threads) processes one query vector (HEAD_DIM) from a tile.
    const uint query_index_in_tile = simdgroup_idx;
    // what thread in the SIMD group is this?
    const uint query_lane_index = local_idx_in_tg % SIMD_WIDTH;

    // Unpack other identifiers
    const uint sequence_index = tg_pos_in_grid.y;
    const uint head_index = tg_pos_in_grid.z;

    // Query index relative to the start of its own sequence's prompt
    const int query_index_in_sequence = tg_pos_in_grid.x * Q_TILE_SIZE + query_index_in_tile;

    // True global index into the flattened (batch * prompt_len) buffer
    const int global_query_index = (int)sequence_index * params.num_prompt_tokens + query_index_in_sequence;
    // swizzle_size = MEMORY_ALIGNMENT_BYTES / sizeof(T);
    constexpr int ELEMS_PER_SWIZZLE = MEMORY_ALIGNMENT_BYTES / sizeof(T);
    const int elements_per_thread = HEAD_DIM / SIMD_WIDTH;
    const uint tg_size = tg_dim.x * tg_dim.y * tg_dim.z;

    // Calculate sequence offsets for batched prompt data
    ulong q_sequence_offset = (ulong)sequence_index * params.num_prompt_tokens * params.num_q_heads * HEAD_DIM;
    ulong kv_sequence_offset = (ulong)sequence_index * params.num_prompt_tokens * params.num_kv_heads * HEAD_DIM;

    // Partition the single threadgroup memory buffer into logical sections.
    threadgroup uchar* current_mem_ptr = tg_mem;

    // 1. Query Tile: Stores the block of queries this threadgroup owns.
    threadgroup T* query_tile = (threadgroup T*)current_mem_ptr;
    current_mem_ptr += Q_TILE_SIZE * HEAD_DIM * sizeof(T);

    // 2. Output Accumulator: Stores the running weighted sum of values for the whole tile.
    threadgroup float* output_accumulator = (threadgroup float*)current_mem_ptr;
    current_mem_ptr += Q_TILE_SIZE * HEAD_DIM * sizeof(float);

    // 3. Softmax Stats: Stores running max_score and sum_exp for each query in the tile.
    threadgroup float* softmax_stats = (threadgroup float*)current_mem_ptr;
    current_mem_ptr += Q_TILE_SIZE * 2 * sizeof(float);

    // 4. Reduction Scratchpad: Stores the running max_score and sum_exp for each query in the tile.
    threadgroup float* reduction_scratchpad = (threadgroup float*)current_mem_ptr;
    current_mem_ptr += Q_TILE_SIZE * SIMD_WIDTH * sizeof(float);

    // Each SIMD group only initializes the memory for its assigned query.
    // Calculate offsets to this query's specific memory regions.
    threadgroup float* query_accumulator = output_accumulator + query_index_in_tile * HEAD_DIM;
    threadgroup float* query_softmax_stats = softmax_stats + query_index_in_tile * 2;

    // Initialize accumulators and stats to zero/negative infinity.
    // Each thread initializes a portion of the memory.
    #pragma unroll
    for (int i = 0; i < elements_per_thread; ++i) {
        int idx = query_lane_index + i * SIMD_WIDTH;
        query_accumulator[idx] = 0.0f;
    }
    // The first thread in the SIMD group initializes the max_score and sum_exp.
    if (query_lane_index == 0) {
        query_softmax_stats[0] = -INFINITY;
        query_softmax_stats[1] = 0.0f;
    }

    // ========================================================================
    // --- Phase 2: Load Query Tile into Threadgroup Memory ---
    // ========================================================================
    // Load the query tile into threadgroup memory cooperatively.
    #pragma unroll
    for (uint i = local_idx_in_tg; i < (Q_TILE_SIZE * HEAD_DIM); i += tg_size) {
        int query_index_in_tile = i / HEAD_DIM;
        int head_index_in_tile = i % HEAD_DIM;
        int query_index_in_tile_for_load = tg_pos_in_grid.x * Q_TILE_SIZE + query_index_in_tile;
        query_tile[i] = (query_index_in_tile_for_load < params.num_prompt_tokens) ?
            q_prompt_in[q_sequence_offset + query_index_in_tile_for_load * params.num_q_heads * HEAD_DIM + head_index * HEAD_DIM + head_index_in_tile]
            : 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // THREADGROUP BARRIER REASON: Ensure the full Q-vector is in shared memory before use.

    // exit if this SIMD group's assigned query is beyond the prompt length.
    if (query_index_in_sequence >= params.num_prompt_tokens) return;

    // ========================================================================
    // --- Phase 3: Main Attention Loop (Streaming K/V) ---
    // ========================================================================

    const int history_len = context_lens_in[sequence_index];
    const int total_context_len = history_len + params.num_prompt_tokens;

    // Determine the correct KV head index for GQA
    const int num_q_per_kv = params.num_q_heads / params.num_kv_heads;
    const int kv_head_idx = head_index / num_q_per_kv;

    //////////////////////////////////////////////////////////////////////////////
    // --- Part A: Attend to Prompt History (Causal) ---
    //////////////////////////////////////////////////////////////////////////////

    for (int kv_idx_prompt = 0; kv_idx_prompt < params.num_prompt_tokens; ++kv_idx_prompt) {
        // --- 1. Load one K vector from the PROMPT buffer into registers ---
        // Each thread loads its piece of the K vector. This is a private, per-thread array.
        thread T key_vector_registers[elements_per_thread];

        // --- Step 3a: Load one K vector from the prompt buffer ---
        device const T* key_vector_device_ptr = k_prompt_in + kv_sequence_offset +
                            (ulong)kv_idx_prompt * params.num_kv_heads * HEAD_DIM +
                            (ulong)kv_head_idx * HEAD_DIM;
        // Parallel copy into registers
        #pragma unroll
        for (int i = 0; i < elements_per_thread; ++i) {
            int idx = query_lane_index + i * SIMD_WIDTH;
            key_vector_registers[i] = key_vector_device_ptr[idx];
        }

        // --- 2. Apply Causal Mask ---
        // If this K pair is "in the future" for this SIMD group's query, skip it.
        if (kv_idx_prompt > query_index_in_sequence) continue;

        // --- 3. Compute Dot Product ---
        // Get a pointer to this SIMD group's query in threadgroup memory.
        threadgroup const T* current_query_vector = query_tile + query_index_in_tile * HEAD_DIM;

        // Each thread computes its partial score.
        float partial_score = 0.0f;
        for (int i = 0; i < elements_per_thread; ++i) {
            int idx = query_lane_index + i * SIMD_WIDTH;
            partial_score += (float)current_query_vector[idx] * (float)key_vector_registers[i];
        }
        // Reduce the partial scores across the 32 threads in this SIMD group.
        float score = simd_sum(partial_score) * params.inv_sqrt_head_dim;

        // --- 4. Update Accumulator ---
        // Only the first thread in the SIMD group performs the scalar updates.
        if (query_lane_index == 0) {
            threadgroup float* stats = query_softmax_stats;
            float old_max_score = stats[0];
            float new_max_score = max(old_max_score, score);

            float scale = 1.0f;
            if (new_max_score > old_max_score) {
                scale = exp(max(old_max_score - new_max_score, params.log_exp_min_clamp));
            }
            // rescale and clamp the exponent arg to avoid overflow
            float probability = exp(max(score - new_max_score, params.log_exp_min_clamp));

            // Update shared stats
            stats[0] = new_max_score;
            stats[1] = stats[1] * scale + probability;

            // Broadcast the scale factor to the other 31 threads using the scratchpad.
            reduction_scratchpad[query_index_in_tile] = scale;
            reduction_scratchpad[query_index_in_tile + Q_TILE_SIZE] = probability;
        }
        // All threads wait for the leader to finish the softmax update
        simdgroup_barrier(mem_flags::mem_threadgroup);

        // --- 5. Load V-vector (Just-In-Time) ---
        thread T value_vector_registers[elements_per_thread];
        // Calculate base pointers for the physical page and Value head
        device const T* value_head_ptr = v_prompt_in + kv_sequence_offset +
                                    (ulong)kv_idx_prompt * params.num_kv_heads * HEAD_DIM +
                                    (ulong)kv_head_idx * HEAD_DIM;

        // Parallel copy into registers
        #pragma unroll
        for (int i = 0; i < elements_per_thread; ++i) {
            int idx = query_lane_index + i * SIMD_WIDTH;
            value_vector_registers[i] = value_head_ptr[idx];
        }

        // --- 6. V-Aggregation ---
        // All 32 threads read the scale factor broadcast by their leader.
        float final_scale = reduction_scratchpad[query_index_in_tile];
        float final_probability = reduction_scratchpad[query_index_in_tile + Q_TILE_SIZE];

        // Barrier to ensure all threads have pulled the broadcast values
        simdgroup_barrier(mem_flags::mem_none);

        // All 32 threads participate in updating the accumulator.
        #pragma unroll
        for (int i = 0; i < elements_per_thread; ++i) {
            int idx = query_lane_index + i * SIMD_WIDTH;
            query_accumulator[idx] = query_accumulator[idx] * final_scale +
                                    final_probability * (float)value_vector_registers[i];
        }
    } // end of new prompt token loop
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // THREADGROUP BARRIER REASON: Ensure all threads have finished the loop over new prompt tokens.

    //////////////////////////////////////////////////////////////////////////////
    // --- Part B: Attend to History (Non-Causal) ---
    //////////////////////////////////////////////////////////////////////////////

    // history_len is unchanged from part A
    const int num_history_pages = (history_len + params.tokens_per_page - 1) / params.tokens_per_page;

    // K cache is in 5D layout:
    // swizzle_size = MEMORY_ALIGNMENT_BYTES / sizeof(T);
    // [pages][num_kv_heads][head_dim / swizzle_size][tokens_per_page][swizzle_size]
    //
    // V cache is in 4D layout:
    // [pages][num_kv_heads][head_dim][tokens_per_page]
    for (int page_idx = 0; page_idx < num_history_pages; ++page_idx) {
        // --- 1. Get Physical Page ID ---
        const uint page_table_idx = sequence_index * params.max_logical_pages_per_seq + page_idx;
        const uint physical_page_id = page_table_in[page_table_idx];

         // --- 2. Loop over Tokens on this Page ---
        const int start_token_on_page = page_idx * params.tokens_per_page;
        const int end_token_on_page = min(start_token_on_page + (int)params.tokens_per_page, history_len);

        for (int token_index = 0; token_index < params.tokens_per_page; ++token_index) {
            const int global_token_index = start_token_on_page + token_index;
            const ulong page_base_offset = (ulong)physical_page_id * params.num_kv_heads *
                                            params.tokens_per_page * HEAD_DIM;
            if (global_token_index >= history_len) {
                break; // partial page, stop here
            }

            // --- 3. Load Key Vector from Paged Cache into Registers ---
            thread T key_vector_registers[elements_per_thread];
            // Calculate base pointers for the physical page and Key head
            const ulong STRIDE_HEAD = (HEAD_DIM / ELEMS_PER_SWIZZLE) * params.tokens_per_page * ELEMS_PER_SWIZZLE;
            const ulong STRIDE_PAGE = params.num_kv_heads * STRIDE_HEAD;

            // Calculate base pointer for the start of the physical page
            device const T* page_ptr = k_cache_paged_in + (ulong)physical_page_id * STRIDE_PAGE;

            // Calculate base pointer for the specific KV head on that page
            device const T* key_head_ptr = page_ptr + (ulong)kv_head_idx * STRIDE_HEAD;

            // Parallel de-swizzling load for the K-vector
            #pragma unroll
            for (int i = 0; i < elements_per_thread; ++i) {
                int h_dim_offset = query_lane_index + i * SIMD_WIDTH;
                int swizzle_chunk = h_dim_offset / ELEMS_PER_SWIZZLE;
                int swizzle_offset = h_dim_offset % ELEMS_PER_SWIZZLE;

                ulong key_load_offset = (ulong)swizzle_chunk * params.tokens_per_page * ELEMS_PER_SWIZZLE +
                                        (ulong)token_index * ELEMS_PER_SWIZZLE +
                                        swizzle_offset;

                key_vector_registers[i] = key_head_ptr[key_load_offset];
            }

            // --- 4a. Compute Score ---
            threadgroup const T* current_query_vector = query_tile + query_index_in_tile * HEAD_DIM;
            float partial_score = 0.0f;
            #pragma unroll
            for (int i = 0; i < elements_per_thread; ++i) {
                int idx = query_lane_index + i * SIMD_WIDTH;
                partial_score += (float)current_query_vector[idx] * (float)key_vector_registers[i];
            }
            float score = simd_sum(partial_score) * params.inv_sqrt_head_dim;

            // --- 4b. Update Softmax Statistics ---
            // Only the first thread in the SIMD group performs the updates.
            if (query_lane_index == 0) {
                threadgroup float* stats = query_softmax_stats;
                float old_max_score = stats[0];
                float new_max_score = max(old_max_score, score);

                float scale = 1.0f;
                if (new_max_score > old_max_score) {
                    scale = exp(max(old_max_score - new_max_score, params.log_exp_min_clamp));
                }

                float probability = exp(max(score - new_max_score, params.log_exp_min_clamp));

                // Update shared stats
                stats[0] = new_max_score;
                stats[1] = stats[1] * scale + probability;

                // Broadcast the scale and probability using the scratchpad
                reduction_scratchpad[query_index_in_tile] = scale;
                reduction_scratchpad[query_index_in_tile + Q_TILE_SIZE] = probability;
            }
            // All threads wait for the leader
            simdgroup_barrier(mem_flags::mem_threadgroup);

            // --- 5. Load V-vector (Just-In-Time) ---
            thread T value_vector_registers[elements_per_thread];
            // Calculate base pointers for the physical page and Value head
            device const T* value_head_ptr = v_cache_paged_in + page_base_offset +
                                        (ulong)kv_head_idx * params.tokens_per_page * HEAD_DIM;

            // Strided load for V (non-swizzled layout)
            #pragma unroll
            for (int i = 0; i < elements_per_thread; ++i) {
                int head_dim_offset = query_lane_index + i * SIMD_WIDTH;
                ulong value_load_offset = (ulong)head_dim_offset * params.tokens_per_page + token_index;
                value_vector_registers[i] = value_head_ptr[value_load_offset];
            }

            // --- 6. V-Aggregation for History Token ---
            // All threads in SIMD group read the scale and probability factors broadcast by their leader.
            float final_scale = reduction_scratchpad[query_index_in_tile];
            float final_probability = reduction_scratchpad[query_index_in_tile + Q_TILE_SIZE];

            // Barrier to ensure all threads have pulled the broadcast values
            simdgroup_barrier(mem_flags::mem_none);

            // All threads participate in updating the accumulator in parallel.
            // acc = (acc * scale) + (probability * V)
            #pragma unroll
            for (int i = 0; i < elements_per_thread; ++i) {
                int idx = query_lane_index + i * SIMD_WIDTH;
                query_accumulator[idx] = query_accumulator[idx] * final_scale +
                                        final_probability * (float)value_vector_registers[i];
            }
        } // end of tokens in page loop
    } // end of pages loop
    threadgroup_barrier(mem_flags::mem_threadgroup); // Ensure all accumulations are done.

    // ========================================================================
    // --- Phase 4: Finalization and Output ---
    // ========================================================================

    // --- 1. Final Normalization ---
    float sum_exp = query_softmax_stats[1];
    float inv_sum_exp = 1.0f / (sum_exp + 1e-6f);

    // Each thread in the SIMD group normalizes its portion of the accumulator.
    #pragma unroll
    for (int i = 0; i < elements_per_thread; ++i) {
        int idx = query_lane_index + i * SIMD_WIDTH;
        query_accumulator[idx] *= inv_sum_exp;
    }

    // --- 2. Write Output to Global Memory ---
    device T* output_ptr_base = output_buffer +
                            (ulong)global_query_index * params.num_q_heads * HEAD_DIM +
                            (ulong)head_index * HEAD_DIM;

    // Each thread in the SIMD group writes its slice of the final vector.
    #pragma unroll
    for (int i = 0; i < elements_per_thread; ++i) {
        int idx = query_lane_index + i * SIMD_WIDTH;
        output_ptr_base[idx] = (T)query_accumulator[idx];
    }
} // end of kernel
