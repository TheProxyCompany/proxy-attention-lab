// paged_attention.metal
// Metal shader implementation for paged attention operations with tiled V accumulation.
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

// Compile-time constants for kernel configuration
constant static const uint kMaxSimdGroupsPerThreadgroup = 8;  // Max possible SIMD groups (256/32 = 8)
constant static const uint kMaxAccumulationTile = 64;         // Size of the fixed tile for V-accumulation

/**
 *  paged_attn_kernel
 *  ----------------------------------------
 *  Implements paged attention for transformer models with key-value memory pooling.
 *
 *  This kernel computes attention scores between query vectors and key vectors stored in
 *  paged memory, then weights value vectors to produce the final output. The implementation
 *  handles:
 *  - Multi-headed attention (MHA)
 *  - Grouped query attention (GQA)
 *  - Multi-query attention (MQA)
 *  - Efficient vectorized memory access
 *  - Tiled value accumulation for large head dimensions
 *  - SIMD-accelerated parallel reductions
 *
 *  Thread Mapping:
 *  - One threadgroup processes one query item (single token+head pair)
 *  - Each thread in the threadgroup collaboratively processes a portion of history
 *  - SIMD groups are used for efficient parallel reductions
 *
 *  Memory Layout:
 *  - Query shape: [N_tokens × H_q × D] or [N] when H_q==1
 *  - Key/Value cache: [Pages × TokensPerPage × H_kv × D]
 *  - Page table: [Sequences × MaxBlocksPerSequence]
 *
 *  Algorithm Stages:
 *  1. Collaboratively load and pre-scale query vector into shared memory
 *  2. Parallel scan through history tokens, computing max scores
 *  3. SIMD-group-based reduction to find global max and compute softmax normalization
 *  4. Second pass through history to fetch and accumulate weighted value vectors
 *  5. Tiled processing for handling large head dimensions
 */
[[kernel]] void paged_attn_kernel(
    device      const half* queries_in              [[buffer(0)]],
    device      const half* k_cache_pool_in         [[buffer(1)]],
    device      const half* v_cache_pool_in         [[buffer(2)]],
    device      const uint* page_table_in           [[buffer(3)]],
    device      const int* sequence_lengths_in      [[buffer(4)]],
    device      const int* query_to_seq_map_in      [[buffer(5)]],
    device      const int* query_token_offset_in    [[buffer(6)]],
    constant    const PagedAttentionParams& params  [[buffer(7)]],
    device      half* output_buffer                 [[buffer(8)]],
    threadgroup float* tg_mem                       [[threadgroup(0)]],
    uint3       tg_pos_in_grid                      [[threadgroup_position_in_grid]],
    uint3       tg_dim                              [[threads_per_threadgroup]],
    uint        local_idx_in_tg                     [[thread_index_in_threadgroup]],
    uint        simd_lane_id                        [[thread_index_in_simdgroup]],
    uint        simd_group_id                       [[simdgroup_index_in_threadgroup]]
) {
    // --- Create a local, corrected scale value using rsqrt for better performance ---
    float kernel_scale;
    if (params.head_dim > 0) {
        kernel_scale = rsqrt((float)params.head_dim);  // Use rsqrt for 1/sqrt()
    } else {
        kernel_scale = 1.0f;  // Fallback for head_dim == 0 to avoid division by zero/NaN
    }

    // All thread parameters are now properly used in the kernel

    // --- Thread Identifiers ---
    uint global_item_idx = tg_pos_in_grid.x;    // Identifies the query-head item
    uint local_thread_idx = local_idx_in_tg;    // Thread ID within this group

    // --- Carve the dynamic threadgroup buffer into logical sub-arrays ---
    // Threadgroup scratch arrays for reductions.
    // TODO(O3/PAL-TDD8): Investigate potential bank conflicts for these arrays if
    // their sizes (threads_per_item_group or kMaxSimdGroupsPerThreadgroup)
    // align poorly with Metal's memory bank width (typically 32 bytes, i.e., 8 floats).
    // Profiling in TDD-8 should guide whether 2-way interleaving (e.g., float2)
    // or padding is necessary for G_partial_max_scores or other scratch arrays
    // to mitigate contention if observed.
    threadgroup float* q_shmem = tg_mem;  // head_dim floats
    threadgroup float* G_partial_max_scores = q_shmem + params.head_dim;  // threads_per_tg floats
    threadgroup float* G_simd_reduced_maxes = G_partial_max_scores + tg_dim.x;
    threadgroup float* G_simd_reduced_adjusted_sum_exps =
        G_simd_reduced_maxes + kMaxSimdGroupsPerThreadgroup;
    threadgroup float* G_final_max_for_item =
        G_simd_reduced_adjusted_sum_exps + kMaxSimdGroupsPerThreadgroup;
    threadgroup float* G_final_sum_exp_for_item = G_final_max_for_item + 1;
    threadgroup float4* G_simd_group_v_sums = (threadgroup float4*)(G_final_sum_exp_for_item + 1);

    // --- Determine Q-vector pointer for this item ---
    device const half* q_vector_item_ptr;
    if (params.num_q_heads > 1) {  // 3D Q array [Tokens, QHeads, Dim]
                                   // global_item_idx = token_idx * num_q_heads + q_head_idx
        uint item_token_idx = global_item_idx / params.num_q_heads;
        uint item_q_head_idx = global_item_idx % params.num_q_heads;
        ulong query_base_offset = (ulong)item_token_idx * params.num_q_heads * params.head_dim +
                                 (ulong)item_q_head_idx * params.head_dim;
        q_vector_item_ptr = queries_in + query_base_offset;
    } else {  // Original Q was 1D/2D, params.num_q_heads = 1
        q_vector_item_ptr = queries_in + (global_item_idx * params.head_dim);
    }

    // --- Stage Q-vector into shared memory and pre-scale with kernel_scale ---
    for (uint i = local_thread_idx; i < params.head_dim; i += tg_dim.x) {
        q_shmem[i] = (float)q_vector_item_ptr[i] * kernel_scale;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);  // Synchronize after Q-staging

    // --- Determine this item's overall history and sequence length ---
    // For 3D queries [NumTokens, NumQHeads, HeadDim], query_to_seq_map and query_token_offset
    // map to tokens, not to the combined (token,head) pairs
    uint token_idx_for_sideband_lookup;
    if (params.num_q_heads > 1) {  // 3D queries case
        token_idx_for_sideband_lookup = global_item_idx / params.num_q_heads;
    } else {  // 1D/2D queries case - direct mapping
        token_idx_for_sideband_lookup = global_item_idx;
    }

    // Get sequence index and validate
    uint item_seq_idx_in_batch = (uint)query_to_seq_map_in[token_idx_for_sideband_lookup];
    if (item_seq_idx_in_batch >= params.num_sequences_in_batch) {
        if (local_thread_idx == 0) {
            output_buffer[global_item_idx] = 0.0h;
        }
        return;
    }

    // Get token position and validate
    int item_signed_query_token_offset = query_token_offset_in[token_idx_for_sideband_lookup];
    if (item_signed_query_token_offset < 0) {
        if (local_thread_idx == 0) {
            output_buffer[global_item_idx] = 0.0h;
        }
        return;
    }

    uint item_current_q_token_logical_pos = (uint)item_signed_query_token_offset;
    uint item_actual_sequence_length = (uint)sequence_lengths_in[item_seq_idx_in_batch];
    uint item_effective_history_length = min(item_current_q_token_logical_pos,
                                           item_actual_sequence_length);

    // --- Parallel History Scan Setup ---
    float thread_local_max_score = -INFINITY;
    float thread_local_sum_exp = 0.0f;
    bool thread_processed_any_valid_score = false;
    bool thread_first_valid_score_in_chunk = true;

    // Define thread-local V accumulator tile
    float v_accum_tile[kMaxAccumulationTile];  // Fixed-size tile for V accumulation

    if (item_effective_history_length > 0) {
        // Distribute history tokens among threads in the group
        uint num_hist_tokens_per_thread =
            (item_effective_history_length + tg_dim.x - 1) / tg_dim.x;
        uint hist_start_idx = local_thread_idx * num_hist_tokens_per_thread;
        uint hist_end_idx = min((local_thread_idx + 1) * num_hist_tokens_per_thread,
                              item_effective_history_length);

        // --- This thread's loop over its assigned history chunk ---
        for (uint hist_token_idx = hist_start_idx; hist_token_idx < hist_end_idx; ++hist_token_idx) {
            uint target_historical_logical_token_pos = hist_token_idx;

            // Convert logical token position to page table coordinates
            uint logical_block_idx = target_historical_logical_token_pos / params.tokens_per_page;
            uint token_slot_in_page = target_historical_logical_token_pos % params.tokens_per_page;

            // Skip if beyond page table bounds
            if (logical_block_idx >= params.max_logical_blocks_per_seq) {
                break;
            }

            // Look up physical page ID from page table
            uint page_table_flat_idx =
                item_seq_idx_in_batch * params.max_logical_blocks_per_seq + logical_block_idx;
            uint physical_page_id = page_table_in[page_table_flat_idx];

            // Skip invalid physical pages
            if (physical_page_id >= params.num_physical_pages_in_pool) {
                continue;
            }

            // --- KV Head Selection ---
            uint q_head_for_kv_map_within_item = 0;
            if (params.num_q_heads > 1) {  // Original Q was 3D
                q_head_for_kv_map_within_item = global_item_idx % params.num_q_heads;
            }

            uint target_kv_head_idx = 0;
            if (params.num_kv_heads > 0) {
                if (params.num_q_heads > params.num_kv_heads) {  // GQA case
                    uint gqa_factor = params.num_q_heads / params.num_kv_heads;
                    target_kv_head_idx = q_head_for_kv_map_within_item / gqa_factor;
                } else if (params.num_q_heads < params.num_kv_heads) {  // MQA case
                    target_kv_head_idx = 0;  // For MQA, always use kv_head 0
                } else {  // MHA case (1:1 mapping)
                    target_kv_head_idx = q_head_for_kv_map_within_item;
                }

                // Safety check
                if (target_kv_head_idx >= params.num_kv_heads) {
                    target_kv_head_idx = target_kv_head_idx % params.num_kv_heads;
                }
            }

            // --- K-Vector Address Calculation ---
            ulong k_elements_per_token_slot_per_kv_head = (ulong)params.head_dim;
            ulong k_elements_per_token_slot_all_kv_heads =
                (ulong)params.num_kv_heads * k_elements_per_token_slot_per_kv_head;
            ulong k_elements_per_physical_page =
                (ulong)params.tokens_per_page * k_elements_per_token_slot_all_kv_heads;
            ulong k_page_base_offset_in_elements =
                (ulong)physical_page_id * k_elements_per_physical_page;
            ulong k_token_slot_base_offset_in_elements =
                (ulong)token_slot_in_page * k_elements_per_token_slot_all_kv_heads;
            ulong k_kv_head_base_offset_in_elements =
                (ulong)target_kv_head_idx * k_elements_per_token_slot_per_kv_head;
            ulong k_vector_start_idx = k_page_base_offset_in_elements +
                                      k_token_slot_base_offset_in_elements +
                                      k_kv_head_base_offset_in_elements;
            device const half* k_vector_ptr = k_cache_pool_in + k_vector_start_idx;

            // --- Compute Dot Product Q·K^T with alignment check and vectorization ---
            float current_score_float = 0.0f;

            // Check for 8-byte alignment for packed_half4
            bool use_vectorized_load = ((uintptr_t)k_vector_ptr % 8 == 0);

            if (params.head_dim > 0) {
                if (use_vectorized_load && (params.head_dim % 4 == 0)) {
                    // Vectorized path - process 4 elements at a time
                    device const packed_half4* k_ptr_h4 =
                        reinterpret_cast<device const packed_half4*>(k_vector_ptr);
                    for (uint i = 0; i < params.head_dim / 4; ++i) {
                        // Load 4 half K elements, convert to float4
                        float4 k_vec_f4 = float4(k_ptr_h4[i]);
                        // Q is already float in q_shmem. Load 4 floats.
                        float4 q_vec_f4 = {
                            q_shmem[i * 4 + 0],
                            q_shmem[i * 4 + 1],
                            q_shmem[i * 4 + 2],
                            q_shmem[i * 4 + 3]
                        };

                        float dp = dot(q_vec_f4, k_vec_f4);
                        current_score_float += dp;
                    }
                } else {
                    // Scalar fallback path
                    for (uint i = 0; i < params.head_dim; ++i) {
                        current_score_float += q_shmem[i] * (float)k_vector_ptr[i];
                    }
                }
            }

            // Note: current_score_float is already scaled because q_shmem was pre-scaled

            // --- Online Log-Sum-Exp update for this thread's local accumulation ---
            if (thread_first_valid_score_in_chunk) {
                thread_local_max_score = current_score_float;
                thread_local_sum_exp = 1.0f;
                thread_first_valid_score_in_chunk = false;
            } else {
                float new_potential_max = max(thread_local_max_score, current_score_float);
                thread_local_sum_exp = thread_local_sum_exp *
                                     exp(max(thread_local_max_score - new_potential_max, -16.0f)) +
                                     exp(max(current_score_float - new_potential_max, -16.0f));
                thread_local_max_score = new_potential_max;
            }

            thread_processed_any_valid_score = true;
        }  // End of history token loop
    }  // End of effective history check

    // --- Initialize Shared Memory for Reduction ---
    // Each thread writes its local max score to its slot in G_partial_max_scores
    if (thread_processed_any_valid_score) {
        G_partial_max_scores[local_thread_idx] = thread_local_max_score;
    } else {
        G_partial_max_scores[local_thread_idx] = -INFINITY;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);  // Ensure all local maxes are written

    // --- Perform Threadgroup Reduction for Max Score ---
    // 1. SIMD-group level reduction
    float simd_max_val = simd_max(thread_local_max_score);  // All threads in SIMD group get the same max

    if (simd_lane_id == 0) {  // One thread per SIMD group writes its group's max
        G_simd_reduced_maxes[simd_group_id] = simd_max_val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 2. Reduce across SIMD group results (thread 0 does the final reduction)
    float final_global_max_score = -INFINITY;
    if (local_thread_idx == 0) {
        if (item_effective_history_length == 0) {  // Handle no history for the item
            final_global_max_score = 0.0f;
        } else {
            final_global_max_score = G_simd_reduced_maxes[0];  // Start with first SIMD group's max

            // Use the actual number of SIMD groups in this threadgroup
            uint num_simd_groups = (tg_dim.x + 31) / 32;  // Ceiling division by 32
            for (uint i = 1; i < num_simd_groups; ++i) {
                final_global_max_score = max(final_global_max_score, G_simd_reduced_maxes[i]);
            }

            // Handle the case where all history chunks were empty/invalid
            if (final_global_max_score == -INFINITY) {
                final_global_max_score = 0.0f;
            }
        }

        // Store for broadcasting to all threads
        *G_final_max_for_item = final_global_max_score;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // All threads read the global max score
    float final_max_score_for_item_from_shared = *G_final_max_for_item;

    // --- Adjust and Reduce sum_exp_score ---
    float adjusted_thread_local_sum_exp = 0.0f;

    if (thread_processed_any_valid_score) {
        // Calculate adjustment based on difference between thread's max and global max
        float max_diff = thread_local_max_score - final_max_score_for_item_from_shared;

        // Scale the thread's sum_exp by exp(max_diff)
        adjusted_thread_local_sum_exp = thread_local_sum_exp * exp(max_diff);
    }

    // SIMD-group sum for the adjusted sum_exp values
    float simd_sum_val = simd_sum(adjusted_thread_local_sum_exp);

    if (simd_lane_id == 0) {  // One thread per SIMD group writes its group's sum
        G_simd_reduced_adjusted_sum_exps[simd_group_id] = simd_sum_val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Thread 0 computes final global sum_exp_score
    float final_global_sum_exp_score = 0.0f;
    if (local_thread_idx == 0) {
        // Use the actual number of SIMD groups in this threadgroup
        uint num_simd_groups = (tg_dim.x + 31) / 32;  // Ceiling division by 32
        for (uint i = 0; i < num_simd_groups; ++i) {
            final_global_sum_exp_score += G_simd_reduced_adjusted_sum_exps[i];
        }

        // Write global max score and sum_exp to threadgroup memory for all threads to use
        *G_final_max_for_item = final_global_max_score;
        *G_final_sum_exp_for_item = final_global_sum_exp_score;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);  // Ensure all threads see the global values

    // All threads read the global values
    float global_max_score = *G_final_max_for_item;
    float global_sum_exp_score = *G_final_sum_exp_for_item;

    // --- Early exit for zero history case ---
    if (item_effective_history_length == 0) {
        if (local_thread_idx == 0) {
            // Write zeros to the entire output V-vector for this item
            for (uint i = 0; i < params.head_dim; ++i) {
                output_buffer[global_item_idx * params.head_dim + i] = 0.0h;
            }
        }
        return;  // All threads in the group exit early
    }

    // --- Outer Loop for Tiled V-Accumulation ---
    for (uint base_dim_offset = 0; base_dim_offset < params.head_dim;
         base_dim_offset += params.max_accum_tile_runtime) {
        uint current_tile_len = min(params.max_accum_tile_runtime, params.head_dim - base_dim_offset);

        // Zero out v_accum_tile for the current tile
        for (uint i = 0; i < current_tile_len; ++i) {
            v_accum_tile[i] = 0.0f;
        }

        // --- Second Pass: V-fetching and accumulation using global stats ---
        if (item_effective_history_length > 0) {
            // Distribute history tokens among threads (same as in first pass)
            uint num_hist_tokens_per_thread =
                (item_effective_history_length + tg_dim.x - 1) / tg_dim.x;
            uint hist_start_idx = local_thread_idx * num_hist_tokens_per_thread;
            uint hist_end_idx = min((local_thread_idx + 1) * num_hist_tokens_per_thread,
                                  item_effective_history_length);

            // Each thread rescans its assigned history chunk
            for (uint hist_token_idx = hist_start_idx; hist_token_idx < hist_end_idx;
                 ++hist_token_idx) {
                uint target_historical_logical_token_pos = hist_token_idx;

                // Convert logical position to page coordinates
                uint logical_block_idx = target_historical_logical_token_pos / params.tokens_per_page;
                uint token_slot_in_page = target_historical_logical_token_pos % params.tokens_per_page;

                // Skip if beyond page table bounds
                if (logical_block_idx >= params.max_logical_blocks_per_seq) {
                    break;
                }

                // Look up physical page ID from page table
                uint page_table_flat_idx =
                    item_seq_idx_in_batch * params.max_logical_blocks_per_seq + logical_block_idx;
                uint physical_page_id = page_table_in[page_table_flat_idx];

                // Skip invalid physical pages
                if (physical_page_id >= params.num_physical_pages_in_pool) {
                    continue;
                }

                // --- KV Head Selection (same as first pass) ---
                uint q_head_for_kv_map_within_item = 0;
                if (params.num_q_heads > 1) {
                    q_head_for_kv_map_within_item = global_item_idx % params.num_q_heads;
                }

                uint target_kv_head_idx = 0;
                if (params.num_kv_heads > 0) {
                    if (params.num_q_heads > params.num_kv_heads) {  // GQA
                        uint gqa_factor = params.num_q_heads / params.num_kv_heads;
                        target_kv_head_idx = q_head_for_kv_map_within_item / gqa_factor;
                    } else if (params.num_q_heads < params.num_kv_heads) {  // MQA
                        target_kv_head_idx = 0;
                    } else {  // MHA
                        target_kv_head_idx = q_head_for_kv_map_within_item;
                    }
                    if (target_kv_head_idx >= params.num_kv_heads) {
                        target_kv_head_idx = target_kv_head_idx % params.num_kv_heads;
                    }
                }

                // --- K-Vector Address Calculation ---
                ulong k_elements_per_token_slot_per_kv_head = (ulong)params.head_dim;
                ulong k_elements_per_token_slot_all_kv_heads =
                    (ulong)params.num_kv_heads * k_elements_per_token_slot_per_kv_head;
                ulong k_elements_per_physical_page =
                    (ulong)params.tokens_per_page * k_elements_per_token_slot_all_kv_heads;
                ulong k_page_base_offset_in_elements =
                    (ulong)physical_page_id * k_elements_per_physical_page;
                ulong k_token_slot_base_offset_in_elements =
                    (ulong)token_slot_in_page * k_elements_per_token_slot_all_kv_heads;
                ulong k_kv_head_base_offset_in_elements =
                    (ulong)target_kv_head_idx * k_elements_per_token_slot_per_kv_head;
                ulong k_vector_start_idx = k_page_base_offset_in_elements +
                                          k_token_slot_base_offset_in_elements +
                                          k_kv_head_base_offset_in_elements;
                device const half* k_vector_ptr = k_cache_pool_in + k_vector_start_idx;

                // --- Recompute attention score with alignment check ---
                float score = 0.0f;
                bool use_vectorized_k_load_pass2 = ((uintptr_t)k_vector_ptr % 8 == 0);

                if (params.head_dim > 0) {
                    if (use_vectorized_k_load_pass2 && (params.head_dim % 4 == 0)) {
                        // Vectorized path
                        device const packed_half4* k_ptr_h4 =
                            reinterpret_cast<device const packed_half4*>(k_vector_ptr);
                        for (uint i = 0; i < params.head_dim / 4; ++i) {
                            float4 k_vec_f4 = float4(k_ptr_h4[i]);
                            float4 q_vec_f4 = {
                                q_shmem[i * 4 + 0],
                                q_shmem[i * 4 + 1],
                                q_shmem[i * 4 + 2],
                                q_shmem[i * 4 + 3]
                            };
                            score += dot(q_vec_f4, k_vec_f4);
                        }
                    } else {
                        // Scalar fallback path
                        for (uint i = 0; i < params.head_dim; ++i) {
                            score += q_shmem[i] * (float)k_vector_ptr[i];
                        }
                    }
                }

                // Calculate softmax probability using global stats
                float softmax_prob_i = 0.0f;
                if (global_sum_exp_score > 1e-9f) {
                    softmax_prob_i = exp(max(score - global_max_score, -16.0f)) / global_sum_exp_score;
                }

                // --- V-Vector Address Calculation ---
                ulong v_elements_per_token_slot_per_kv_head = (ulong)params.head_dim;
                ulong v_elements_per_token_slot_all_kv_heads =
                    (ulong)params.num_kv_heads * v_elements_per_token_slot_per_kv_head;
                ulong v_elements_per_physical_page =
                    (ulong)params.tokens_per_page * v_elements_per_token_slot_all_kv_heads;
                ulong v_page_base_offset_in_elements =
                    (ulong)physical_page_id * v_elements_per_physical_page;
                ulong v_token_slot_base_offset_in_elements =
                    (ulong)token_slot_in_page * v_elements_per_token_slot_all_kv_heads;
                ulong v_kv_head_base_offset_in_elements =
                    (ulong)target_kv_head_idx * v_elements_per_token_slot_per_kv_head;
                ulong v_vector_start_idx = v_page_base_offset_in_elements +
                                          v_token_slot_base_offset_in_elements +
                                          v_kv_head_base_offset_in_elements;
                device const half* v_vector_ptr = v_cache_pool_in + v_vector_start_idx;

                // --- V-Accumulation for the current tile with alignment check ---
                bool use_vectorized_v_load_pass2 = ((uintptr_t)v_vector_ptr % 8 == 0);
                bool is_base_offset_aligned = (base_dim_offset % 4 == 0);
                bool is_tile_len_multiple_of_4 = (current_tile_len % 4 == 0);

                if (current_tile_len > 0) {
                    if (use_vectorized_v_load_pass2 && (params.head_dim % 4 == 0) &&
                        is_base_offset_aligned && is_tile_len_multiple_of_4) {
                        // Fully vectorized V-accumulation for the tile
                        for (uint k_tile_h4 = 0; k_tile_h4 < current_tile_len / 4; ++k_tile_h4) {
                            uint v_component_global_idx_base = base_dim_offset + k_tile_h4 * 4;
                            device const packed_half4* v_ptr_h4_tile =
                                reinterpret_cast<device const packed_half4*>(
                                    v_vector_ptr + v_component_global_idx_base);
                            float4 v_vec_f4_tile = float4(v_ptr_h4_tile[0]);  // Load float4 chunk

                            // Accumulate into the v_accum_tile
                            v_accum_tile[k_tile_h4 * 4 + 0] += softmax_prob_i * v_vec_f4_tile.x;
                            v_accum_tile[k_tile_h4 * 4 + 1] += softmax_prob_i * v_vec_f4_tile.y;
                            v_accum_tile[k_tile_h4 * 4 + 2] += softmax_prob_i * v_vec_f4_tile.z;
                            v_accum_tile[k_tile_h4 * 4 + 3] += softmax_prob_i * v_vec_f4_tile.w;
                        }
                    } else {
                        // Scalar fallback V-accumulation for the tile
                        for (uint k_tile_scalar = 0; k_tile_scalar < current_tile_len; ++k_tile_scalar) {
                            uint v_component_global_idx = base_dim_offset + k_tile_scalar;
                            if (v_component_global_idx < params.head_dim) {
                                float v_comp = (float)v_vector_ptr[v_component_global_idx];
                                v_accum_tile[k_tile_scalar] += softmax_prob_i * v_comp;
                            }
                        }
                    }
                }
            }  // End of history token loop
        }  // End of effective history check

        // --- NEW SIMD-Group V-Component Reduction for the CURRENT TILE (v_accum_tile) ---
        // This barrier ensures all threads have finished accumulating into their v_accum_tile
        // before reduction and output writes begin. This is the *only* barrier needed for this tile's V-reduction.
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Process v_accum_tile in float4 chunks
        // Loop 'i' iterates over the starting index of each float4 chunk within the current tile.
        for (uint i = 0; i < current_tile_len; i += 4) {
            float4 partial_v_chunk = float4(0.0f); // Initialize to zero

            // Safely load into partial_v_chunk, respecting current_tile_len
            // This ensures we don't read past the valid data in v_accum_tile for the last, possibly partial, chunk.
            if (i < current_tile_len)     partial_v_chunk.x = v_accum_tile[i+0];
            if (i+1 < current_tile_len)   partial_v_chunk.y = v_accum_tile[i+1];
            if (i+2 < current_tile_len)   partial_v_chunk.z = v_accum_tile[i+2];
            if (i+3 < current_tile_len)   partial_v_chunk.w = v_accum_tile[i+3];

            // Perform SIMD-group reduction for each component of the float4.
            // The result (reduced_v_chunk) will be identical for all threads within the same SIMD-group.
            float4 reduced_simd_group_sum_chunk;
            reduced_simd_group_sum_chunk.x = simd_sum(partial_v_chunk.x);
            reduced_simd_group_sum_chunk.y = simd_sum(partial_v_chunk.y);
            reduced_simd_group_sum_chunk.z = simd_sum(partial_v_chunk.z);
            reduced_simd_group_sum_chunk.w = simd_sum(partial_v_chunk.w);

            // Lane 0 of each SIMD group writes its SIMD group's sum to the shared G_simd_group_v_sums array.
            if (simd_lane_id == 0) {
                G_simd_group_v_sums[simd_group_id] = reduced_simd_group_sum_chunk;
            }

            // Barrier to ensure all SIMD groups have written their sums to G_simd_group_v_sums
            // before thread 0 proceeds with the final summation.
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Thread 0 of the entire threadgroup performs the final sum across SIMD group results.
            if (local_thread_idx == 0) {
                float4 final_sum_for_chunk = float4(0.0f);
                uint num_simd_groups_in_tg = (tg_dim.x + 31) / 32; // Assumes SIMD width of 32

                for (uint sg_idx = 0; sg_idx < num_simd_groups_in_tg; ++sg_idx) {
                    final_sum_for_chunk += G_simd_group_v_sums[sg_idx];
                }

                // Write the final summed chunk to the output buffer, respecting current_tile_len bounds for the write.
                uint output_base_idx = global_item_idx * params.head_dim + base_dim_offset + i;
                if (i < current_tile_len)     output_buffer[output_base_idx + 0] = (half)final_sum_for_chunk.x;
                if (i+1 < current_tile_len)   output_buffer[output_base_idx + 1] = (half)final_sum_for_chunk.y;
                if (i+2 < current_tile_len)   output_buffer[output_base_idx + 2] = (half)final_sum_for_chunk.z;
                if (i+3 < current_tile_len)   output_buffer[output_base_idx + 3] = (half)final_sum_for_chunk.w;
            }

            // Barrier to ensure thread 0's write to output_buffer is complete before any thread
            // from any SIMD group potentially overwrites G_simd_group_v_sums in the next 'i' iteration.
            // This is critical if G_simd_group_v_sums is reused across 'i' iterations.
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        // No final barrier needed here because the iteration loop already ends with a barrier
    }  // End of base_dim_offset loop for tiling
}  // End of kernel
