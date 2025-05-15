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

constant static const uint kMaxHeadDimMetal = 256; // Kernel's internal max, C++ validates params.head_dim against this
constant static const uint kHeadDimProcessingChunk = 64; // Chunk size for processing head_dim in Pass 2


constant static const uint kMaxAccumulationTile = 64; // Matches kMaxAccumulationTile in C++ code

using namespace metal;

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
 *  - Tiled value accumulation for large head dimensions (via two-pass approach)
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
 *  Algorithm Stages (Two-Pass):
 *  Pass 1:
 *  1. Collaboratively load and pre-scale query vector into shared memory.
 *  2. Single-pass history scan performing online softmax to calculate m_final_global and d_final_global.
 *     - Each thread computes its local m_local and d_local.
 *     - Threadgroup reduction to find m_final_global (overall max score) and d_final_global (overall normalization sum).
 *  Pass 2:
 *  1. Iterate over head_dim in chunks of kHeadDimProcessingChunk.
 *  2. For each chunk:
 *     - Each thread initializes a local o_tile[kHeadDimProcessingChunk].
 *     - Re-scan history:
 *       - Compute Q·K score.
 *       - Calculate attention weight p = exp(score - m_final_global) / d_final_global.
 *       - Accumulate V contributions into o_tile: o_tile += p * V_chunk.
 *     - Reduce o_tile across the threadgroup and write to output buffer.
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

    // --- Basic input validation ---
    // Early exit for degenerate case where head_dim is zero
    if (params.head_dim == 0) {
        // Zero the output and exit (no need for loop if head_dim is 0, but for safety if it's called)
        // Output buffer is implicitly zero if no writes occur, but explicit zeroing can be done if needed.
        return;
    }

    // --- Thread-Local Accumulators for Pass 1 (Online Softmax stats) ---
    float m_local = -INFINITY; // Maximum score accumulator
    float d_local = 0.0f;      // Sum of scaled exponentials accumulator

    // Note: o_local (large array) is removed. Output O will be computed in Pass 2 using o_tile.

    // --- Thread Identifiers ---
    uint global_item_idx = tg_pos_in_grid.x;    // Identifies the query-head item
    uint local_thread_idx = local_idx_in_tg;    // Thread ID within this group
    const uint num_simd_groups = max(1u, (tg_dim.x + 31u) >> 5); // Calculate number of actual SIMD groups

    // --- Carve the dynamic threadgroup buffer into logical sub-arrays ---
    // Layout for Pass 1 and Pass 2 reductions.
    // q_shmem, G_partial_max_scores, G_simd_reduced_maxes, G_simd_reduced_adjusted_sum_exps,
    // G_final_max_for_item, G_final_sum_exp_for_item are primarily for Pass 1.
    // G_simd_group_v_sums is used in Pass 2 for o_tile reduction.

    threadgroup float* q_shmem = tg_mem;  // head_dim floats

    uintptr_t current_offset = (uintptr_t)(q_shmem + params.head_dim);
    current_offset = (current_offset + 15u) & ~15u;
    threadgroup float* G_partial_max_scores = (threadgroup float*)current_offset;  // threads_per_tg floats

    current_offset = (uintptr_t)(G_partial_max_scores + tg_dim.x);
    current_offset = (current_offset + 15u) & ~15u;
    threadgroup float* G_simd_reduced_maxes = (threadgroup float*)current_offset; // num_simd_groups floats

    current_offset = (uintptr_t)(G_simd_reduced_maxes + num_simd_groups);
    current_offset = (current_offset + 15u) & ~15u;
    threadgroup float* G_simd_reduced_adjusted_sum_exps = (threadgroup float*)current_offset; // num_simd_groups floats

    current_offset = (uintptr_t)(G_simd_reduced_adjusted_sum_exps + num_simd_groups);
    current_offset = (current_offset + 15u) & ~15u;
    threadgroup float* G_final_max_for_item = (threadgroup float*)current_offset; // 1 float

    current_offset = (uintptr_t)(G_final_max_for_item + 1);
    current_offset = (current_offset + 15u) & ~15u;
    threadgroup float* G_final_sum_exp_for_item = (threadgroup float*)current_offset; // 1 float

    current_offset = (uintptr_t)(G_final_sum_exp_for_item + 1);
    current_offset = (current_offset + 15u) & ~15u;
    threadgroup float4* G_simd_group_v_sums = (threadgroup float4*)current_offset; // num_simd_groups float4s (for Pass 2 o_tile reduction)

    // --- Determine Q-vector pointer for this item ---
    device const half* q_vector_item_ptr;
    if (params.num_q_heads > 1) {
        uint item_token_idx = global_item_idx / params.num_q_heads;
        uint item_q_head_idx = global_item_idx % params.num_q_heads;
        ulong query_base_offset = (ulong)item_token_idx * params.num_q_heads * params.head_dim +
                                 (ulong)item_q_head_idx * params.head_dim;
        q_vector_item_ptr = queries_in + query_base_offset;
    } else {
        q_vector_item_ptr = queries_in + (global_item_idx * params.head_dim);
    }

    // --- Stage Q-vector into shared memory and pre-scale with kernel_scale ---
    for (uint i = local_thread_idx; i < params.head_dim; i += tg_dim.x) {
        q_shmem[i] = (float)q_vector_item_ptr[i] * kernel_scale;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // --- Determine this item's overall history and sequence length ---
    uint token_idx_for_sideband_lookup;
    if (params.num_q_heads > 1) {
        token_idx_for_sideband_lookup = global_item_idx / params.num_q_heads;
    } else {
        token_idx_for_sideband_lookup = global_item_idx;
    }

    uint item_seq_idx_in_batch = (uint)query_to_seq_map_in[token_idx_for_sideband_lookup];
    if (item_seq_idx_in_batch >= params.num_sequences_in_batch) {
        // Zero the full output vector collaboratively and exit (handled by Pass 2 loop not running if history is 0)
        // For safety, ensure output is zero if exiting early.
        // The Pass 2 loop for h_offset will handle zeroing if item_effective_history_length is 0.
        return;
    }

    int item_signed_query_token_offset = query_token_offset_in[token_idx_for_sideband_lookup];
    if (item_signed_query_token_offset < 0) {
        // Similar to above, Pass 2 will handle zeroing.
        return;
    }

    uint item_current_q_token_logical_pos = (uint)item_signed_query_token_offset;
    uint item_actual_sequence_length = (uint)sequence_lengths_in[item_seq_idx_in_batch];
    uint item_effective_history_length = min(item_current_q_token_logical_pos,
                                         item_actual_sequence_length);

    // --- PASS 1: Calculate m_final_global and d_final_global ---
    if (item_effective_history_length == 0) {
        // If no history, m_final_global = 0, d_final_global = 0 (or 1 to avoid div by zero, but 0 means no contribution).
        // Output will be zeroed in Pass 2.
        if (local_thread_idx == 0) {
            *G_final_max_for_item = 0.0f;
            *G_final_sum_exp_for_item = 0.0f; // Or a safe non-zero like 1.0f if inv_d is used later
        }
        threadgroup_barrier(mem_flags::mem_threadgroup); // Ensure all threads see this before potentially exiting/starting Pass 2
    } else {
        // --- Distribute history tokens among threads in the group for Pass 1---
    uint num_hist_tokens_per_thread =
        (item_effective_history_length + tg_dim.x - 1) / tg_dim.x;
    uint hist_start_idx = local_thread_idx * num_hist_tokens_per_thread;
    uint hist_end_idx = min((local_thread_idx + 1) * num_hist_tokens_per_thread,
                          item_effective_history_length);

        // --- Pass 1: History scan for m_local and d_local ---
    for (uint hist_token_idx = hist_start_idx; hist_token_idx < hist_end_idx; ++hist_token_idx) {
        uint target_historical_logical_token_pos = hist_token_idx;
        uint logical_block_idx = target_historical_logical_token_pos / params.tokens_per_page;
        uint token_slot_in_page = target_historical_logical_token_pos % params.tokens_per_page;

            if (logical_block_idx >= params.max_logical_blocks_per_seq) break;

            uint page_table_flat_idx = item_seq_idx_in_batch * params.max_logical_blocks_per_seq + logical_block_idx;
        uint physical_page_id = page_table_in[page_table_flat_idx];

            if (physical_page_id >= params.num_physical_pages_in_pool) continue;

            uint q_head_for_kv_map_within_item = (params.num_q_heads > 1) ? (global_item_idx % params.num_q_heads) : 0;
        uint target_kv_head_idx = 0;
        if (params.num_kv_heads > 0) {
                if (params.num_q_heads > params.num_kv_heads) { // GQA
                    target_kv_head_idx = q_head_for_kv_map_within_item / (params.num_q_heads / params.num_kv_heads);
                } else { // MHA or MQA (num_q_heads <= num_kv_heads)
                target_kv_head_idx = q_head_for_kv_map_within_item;
            }
                if (target_kv_head_idx >= params.num_kv_heads) target_kv_head_idx %= params.num_kv_heads; // Safety
            }

            ulong k_base_offset = (ulong)physical_page_id * params.tokens_per_page * params.num_kv_heads * params.head_dim +
                                  (ulong)token_slot_in_page * params.num_kv_heads * params.head_dim +
                                  (ulong)target_kv_head_idx * params.head_dim;
            device const half* k_vector_ptr = k_cache_pool_in + k_base_offset;

        float current_score_fp32 = 0.0f;
            bool use_vectorized_load_h4 = (params.head_dim % 4 == 0);

        if (params.head_dim > 0) {
                if (use_vectorized_load_h4) {
                    device const packed_half4* k_ptr_h4 = reinterpret_cast<device const packed_half4*>(k_vector_ptr);
                for (uint i = 0; i < params.head_dim / 4; ++i) {
                    float4 k_vec_f4 = float4(k_ptr_h4[i]);
                        float4 q_vec_f4 = { q_shmem[i*4+0], q_shmem[i*4+1], q_shmem[i*4+2], q_shmem[i*4+3] };
                        current_score_fp32 += dot(q_vec_f4, k_vec_f4);
                }
            } else {
                for (uint i = 0; i < params.head_dim; ++i) {
                    current_score_fp32 += q_shmem[i] * (float)k_vector_ptr[i];
                }
            }
        }

        float new_m_local = max(m_local, current_score_fp32);
            float alpha = exp(m_local - new_m_local);
        float exponent_for_p = current_score_fp32 - new_m_local;
            float p_val = exp(exponent_for_p);

        d_local = d_local * alpha + p_val;
        m_local = new_m_local;
        } // End of Pass 1 history scan loop

    // --- Threadgroup Reduction for m_local (to find m_final_global) ---
        G_partial_max_scores[local_thread_idx] = m_local;
    threadgroup_barrier(mem_flags::mem_threadgroup);

        float simd_max_m = simd_max(m_local);
    if (simd_lane_id == 0 && simd_group_id < num_simd_groups) {
        G_simd_reduced_maxes[simd_group_id] = simd_max_m;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

        float m_final_global_val = -INFINITY; // Renamed to avoid clash with later variable
    if (local_thread_idx == 0) {
            m_final_global_val = G_simd_reduced_maxes[0];
            for (uint i = 1; i < num_simd_groups; ++i) {
                m_final_global_val = max(m_final_global_val, G_simd_reduced_maxes[i]);
            }
            if (m_final_global_val == -INFINITY) m_final_global_val = 0.0f;
            *G_final_max_for_item = m_final_global_val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
        m_final_global_val = *G_final_max_for_item;

        // --- Rescale d_local and Reduce d_local (for d_final_global) ---
        float d_rescale_exponent = m_local - m_final_global_val; // Use thread's m_local and final global m
        float d_local_rescaled = d_local * exp(d_rescale_exponent);

    float simd_sum_d_rescaled = simd_sum(d_local_rescaled);
    if (simd_lane_id == 0 && simd_group_id < num_simd_groups) {
        G_simd_reduced_adjusted_sum_exps[simd_group_id] = simd_sum_d_rescaled;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

        float d_final_global_val = 0.0f; // Renamed
    if (local_thread_idx == 0) {
        for (uint i = 0; i < num_simd_groups; ++i) {
                d_final_global_val += G_simd_reduced_adjusted_sum_exps[i];
            }
            *G_final_sum_exp_for_item = d_final_global_val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
        // d_final_global_val is now in G_final_sum_exp_for_item, readable by all threads.
    } // End of if(item_effective_history_length > 0) for Pass 1

    // All threads read the finalized m_final_global and d_final_global
    float m_final_global = *G_final_max_for_item;
    float d_final_global = *G_final_sum_exp_for_item;
    float inv_d_final_global = (d_final_global > 1e-9f) ? (1.0f / d_final_global) : 0.0f;

    // --- PASS 2: Compute O in tiles ---
    // Thread-local tile for accumulating V contributions for a head_dim chunk
    thread float o_tile[kHeadDimProcessingChunk];

    if (item_effective_history_length == 0) { // If no history, output must be zero
        for (uint h_offset = 0; h_offset < params.head_dim; h_offset += kHeadDimProcessingChunk) {
            if (local_thread_idx == 0) { // Only one thread needs to write zeros for the whole chunk
                for (uint i = 0; i < kHeadDimProcessingChunk; ++i) {
                    uint dim_idx = h_offset + i;
                    if (dim_idx < params.head_dim) {
                         output_buffer[global_item_idx * params.head_dim + dim_idx] = 0.0h;
                    }
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup); // ensure writes complete if other threads skip Pass 2 loop
        return; // Exit after zeroing
    }


    // Loop over head_dim in chunks of kHeadDimProcessingChunk
    for (uint h_offset = 0; h_offset < params.head_dim; h_offset += kHeadDimProcessingChunk) {
        // Initialize o_tile for this chunk
        for (uint i = 0; i < kHeadDimProcessingChunk; ++i) {
            o_tile[i] = 0.0f;
        }

        // Pass 2: Re-scan history to accumulate V into o_tile
        // Each thread processes its assigned history tokens
        uint num_hist_tokens_per_thread_pass2 = (item_effective_history_length + tg_dim.x - 1) / tg_dim.x;
        uint hist_start_idx_pass2 = local_thread_idx * num_hist_tokens_per_thread_pass2;
        uint hist_end_idx_pass2 = min((local_thread_idx + 1) * num_hist_tokens_per_thread_pass2, item_effective_history_length);

        for (uint hist_token_idx = hist_start_idx_pass2; hist_token_idx < hist_end_idx_pass2; ++hist_token_idx) {
            uint target_historical_logical_token_pos = hist_token_idx;
            uint logical_block_idx = target_historical_logical_token_pos / params.tokens_per_page;
            uint token_slot_in_page = target_historical_logical_token_pos % params.tokens_per_page;

            if (logical_block_idx >= params.max_logical_blocks_per_seq) break;

            uint page_table_flat_idx = item_seq_idx_in_batch * params.max_logical_blocks_per_seq + logical_block_idx;
            uint physical_page_id = page_table_in[page_table_flat_idx];

            if (physical_page_id >= params.num_physical_pages_in_pool) continue;

            uint q_head_for_kv_map_within_item = (params.num_q_heads > 1) ? (global_item_idx % params.num_q_heads) : 0;
            uint target_kv_head_idx = 0;
            if (params.num_kv_heads > 0) {
                 if (params.num_q_heads > params.num_kv_heads) { // GQA
                    target_kv_head_idx = q_head_for_kv_map_within_item / (params.num_q_heads / params.num_kv_heads);
                } else { // MHA or MQA
                    target_kv_head_idx = q_head_for_kv_map_within_item;
                }
                if (target_kv_head_idx >= params.num_kv_heads) target_kv_head_idx %= params.num_kv_heads; // Safety
            }

            ulong k_base_offset = (ulong)physical_page_id * params.tokens_per_page * params.num_kv_heads * params.head_dim +
                                  (ulong)token_slot_in_page * params.num_kv_heads * params.head_dim +
                                  (ulong)target_kv_head_idx * params.head_dim;
            device const half* k_vector_ptr = k_cache_pool_in + k_base_offset;

            ulong v_base_offset = (ulong)physical_page_id * params.tokens_per_page * params.num_kv_heads * params.head_dim +
                                  (ulong)token_slot_in_page * params.num_kv_heads * params.head_dim +
                                  (ulong)target_kv_head_idx * params.head_dim;
            device const half* v_vector_full_ptr = v_cache_pool_in + v_base_offset;


            // Compute Q·K score (already scaled as q_shmem is pre-scaled)
            float current_score_fp32 = 0.0f;
            bool use_vectorized_load_h4_pass2 = (params.head_dim % 4 == 0);

            if (params.head_dim > 0) {
                if (use_vectorized_load_h4_pass2) {
                    device const packed_half4* k_ptr_h4 = reinterpret_cast<device const packed_half4*>(k_vector_ptr);
                    for (uint i = 0; i < params.head_dim / 4; ++i) {
                        float4 k_vec_f4 = float4(k_ptr_h4[i]);
                        float4 q_vec_f4 = { q_shmem[i*4+0], q_shmem[i*4+1], q_shmem[i*4+2], q_shmem[i*4+3] };
                        current_score_fp32 += dot(q_vec_f4, k_vec_f4);
                    }
                } else { // Scalar
                    for (uint i = 0; i < params.head_dim; ++i) {
                        current_score_fp32 += q_shmem[i] * (float)k_vector_ptr[i];
                    }
                }
            }

            // Calculate final attention probability p_val = exp(score - m_final) / d_final
            float exp_score_minus_m_final = exp(current_score_fp32 - m_final_global);
            float p_attn_weight = exp_score_minus_m_final * inv_d_final_global;


            // Accumulate V contributions into o_tile for the current head_dim chunk
            // V vector pointer for the current h_offset chunk
            device const half* v_vector_chunk_ptr = v_vector_full_ptr + h_offset;

            bool use_vectorized_v_load_h4 = (kHeadDimProcessingChunk % 4 == 0) && (h_offset % 4 == 0) && (params.head_dim >= h_offset + 4);


            if (use_vectorized_v_load_h4) {
                for (uint i = 0; i < kHeadDimProcessingChunk / 4; ++i) {
                     if (h_offset + i * 4 + 3 < params.head_dim) { // Boundary check
                        float4 v_chunk_f4 = float4(reinterpret_cast<device const packed_half4*>(v_vector_chunk_ptr)[i]);
                        o_tile[i*4+0] += p_attn_weight * v_chunk_f4.x;
                        o_tile[i*4+1] += p_attn_weight * v_chunk_f4.y;
                        o_tile[i*4+2] += p_attn_weight * v_chunk_f4.z;
                        o_tile[i*4+3] += p_attn_weight * v_chunk_f4.w;
                     } else {
                        for(uint j=0; j < 4; ++j) {
                            if (h_offset + i * 4 + j < params.head_dim) {
                                o_tile[i*4+j] += p_attn_weight * (float)v_vector_chunk_ptr[i*4+j];
                            }
                        }
                    }
                }
            } else { // Scalar V accumulation
                for (uint i = 0; i < kHeadDimProcessingChunk; ++i) {
                    if (h_offset + i < params.head_dim) { // Boundary for current head_dim element
                        o_tile[i] += p_attn_weight * (float)v_vector_chunk_ptr[i];
                    }
                }
            }
        } // End of Pass 2 history scan for this o_tile

        // --- Reduce o_tile across threadgroup and write to output ---
        // This reduction sums the o_tile contributions from all threads for the current h_offset chunk.
        // The result is already normalized as p_attn_weight included inv_d_final_global.
        for (uint h_chunk_idx_in_tile = 0; h_chunk_idx_in_tile < kHeadDimProcessingChunk; h_chunk_idx_in_tile += 4) {
            float4 o_chunk_thread_local = float4(0.0f);
            // Check bounds against kHeadDimProcessingChunk and actual params.head_dim for this specific tile processing.
            if (h_chunk_idx_in_tile < kHeadDimProcessingChunk)     o_chunk_thread_local.x = o_tile[h_chunk_idx_in_tile];
            if (h_chunk_idx_in_tile + 1 < kHeadDimProcessingChunk) o_chunk_thread_local.y = o_tile[h_chunk_idx_in_tile + 1];
            if (h_chunk_idx_in_tile + 2 < kHeadDimProcessingChunk) o_chunk_thread_local.z = o_tile[h_chunk_idx_in_tile + 2];
            if (h_chunk_idx_in_tile + 3 < kHeadDimProcessingChunk) o_chunk_thread_local.w = o_tile[h_chunk_idx_in_tile + 3];

            float4 simd_sum_o_chunk;
            simd_sum_o_chunk.x = simd_sum(o_chunk_thread_local.x);
            simd_sum_o_chunk.y = simd_sum(o_chunk_thread_local.y);
            simd_sum_o_chunk.z = simd_sum(o_chunk_thread_local.z);
            simd_sum_o_chunk.w = simd_sum(o_chunk_thread_local.w);

            if (simd_lane_id == 0 && simd_group_id < num_simd_groups) {
                G_simd_group_v_sums[simd_group_id] = simd_sum_o_chunk;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (local_thread_idx == 0) {
                float4 o_final_chunk_summed = float4(0.0f);
            for (uint sg = 0; sg < num_simd_groups; ++sg) {
                    o_final_chunk_summed += G_simd_group_v_sums[sg];
                }

                // Write final output for this chunk of head_dim
                uint out_base = global_item_idx * params.head_dim + h_offset + h_chunk_idx_in_tile;
                if (h_offset + h_chunk_idx_in_tile < params.head_dim)     output_buffer[out_base]     = (half)(o_final_chunk_summed.x);
                if (h_offset + h_chunk_idx_in_tile + 1 < params.head_dim) output_buffer[out_base + 1] = (half)(o_final_chunk_summed.y);
                if (h_offset + h_chunk_idx_in_tile + 2 < params.head_dim) output_buffer[out_base + 2] = (half)(o_final_chunk_summed.z);
                if (h_offset + h_chunk_idx_in_tile + 3 < params.head_dim) output_buffer[out_base + 3] = (half)(o_final_chunk_summed.w);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup); // Ensure writes complete before next tile iter or exit
        }
    } // End of h_offset loop for Pass 2
} // End of kernel
