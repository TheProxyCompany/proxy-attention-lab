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
constant static const uint kLocalAccumulationTileSize = 64;  // Fixed tile size for reduction phase only


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
 *  2. Single-pass history scan performing online softmax and V-aggregation
 *  3. Threadgroup reduction for final output
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
        // Zero the output and exit
        for (uint i = local_idx_in_tg; i < params.head_dim; i += tg_dim.x) {
            ulong output_idx = (ulong)tg_pos_in_grid.x * params.head_dim + i;
            output_buffer[output_idx] = 0.0h;
        }
        return;
    }

    // --- Thread-Local Accumulators for Online Softmax & V-Aggregation ---
    float m_local = -INFINITY; // Maximum score accumulator
    float d_local = 0.0f;      // Sum of scaled exponentials accumulator

    // o_local will store the full V-weighted sum for all head_dim components.
    // This is sized to store up to kMaxHeadDimMetal elements, with C++ validation ensuring
    // params.head_dim <= kMaxHeadDimMetal.
    thread float o_local[kMaxHeadDimMetal];

    // --- Thread Identifiers ---
    uint global_item_idx = tg_pos_in_grid.x;    // Identifies the query-head item
    uint local_thread_idx = local_idx_in_tg;    // Thread ID within this group
    const uint num_simd_groups = max(1u, (tg_dim.x + 31u) >> 5); // Calculate number of actual SIMD groups

    // --- Carve the dynamic threadgroup buffer into logical sub-arrays ---
    // --- Threadgroup Memory Layout & Host Calculation ---
    // The host must allocate enough threadgroup memory. Calculation (example for host):
    // const uint num_simd_groups_host = (threads_per_threadgroup_x + 31u) / 32u;
    // size_t q_shmem_bytes = params.head_dim * sizeof(float);
    // size_t G_partial_max_scores_bytes = threads_per_threadgroup_x * sizeof(float);
    // size_t G_simd_reduced_maxes_bytes = num_simd_groups_host * sizeof(float);
    // size_t G_simd_reduced_adjusted_sum_exps_bytes = num_simd_groups_host * sizeof(float);
    // size_t G_final_max_for_item_bytes = 1 * sizeof(float);
    // size_t G_final_sum_exp_for_item_bytes = 1 * sizeof(float);
    //
    // uintptr_t offset_before_v_sums = q_shmem_bytes + G_partial_max_scores_bytes +
    //                                G_simd_reduced_maxes_bytes + G_simd_reduced_adjusted_sum_exps_bytes +
    //                                G_final_max_for_item_bytes + G_final_sum_exp_for_item_bytes;
    // uintptr_t aligned_offset_for_v_sums = (offset_before_v_sums + 15u) & ~15u; // Align to 16 bytes
    // size_t G_simd_group_v_sums_bytes = num_simd_groups_host * sizeof(float4);
    // size_t total_tg_bytes = aligned_offset_for_v_sums + G_simd_group_v_sums_bytes;
    // commandEncoder.setThreadgroupMemoryLength(total_tg_bytes, 0);

    // Carve up threadgroup memory with proper 16-byte alignment for each section

    // Start with q_shmem at base address
    threadgroup float* q_shmem = tg_mem;  // head_dim floats

    // Align the next section (G_partial_max_scores)
    uintptr_t partial_max_offset = (uintptr_t)(q_shmem + params.head_dim);
    uintptr_t partial_max_aligned = (partial_max_offset + 15u) & ~15u;
    threadgroup float* G_partial_max_scores = (threadgroup float*)partial_max_aligned;  // threads_per_tg floats

    // Align G_simd_reduced_maxes
    uintptr_t simd_maxes_offset = (uintptr_t)(G_partial_max_scores + tg_dim.x);
    uintptr_t simd_maxes_aligned = (simd_maxes_offset + 15u) & ~15u;
    threadgroup float* G_simd_reduced_maxes = (threadgroup float*)simd_maxes_aligned;

    // Align G_simd_reduced_adjusted_sum_exps
    uintptr_t sum_exps_offset = (uintptr_t)(G_simd_reduced_maxes + num_simd_groups);
    uintptr_t sum_exps_aligned = (sum_exps_offset + 15u) & ~15u;
    threadgroup float* G_simd_reduced_adjusted_sum_exps = (threadgroup float*)sum_exps_aligned;

    // Align G_final_max_for_item
    uintptr_t final_max_offset = (uintptr_t)(G_simd_reduced_adjusted_sum_exps + num_simd_groups);
    uintptr_t final_max_aligned = (final_max_offset + 15u) & ~15u;
    threadgroup float* G_final_max_for_item = (threadgroup float*)final_max_aligned;

    // Align G_final_sum_exp_for_item
    uintptr_t final_sum_offset = (uintptr_t)(G_final_max_for_item + 1);
    uintptr_t final_sum_aligned = (final_sum_offset + 15u) & ~15u;
    threadgroup float* G_final_sum_exp_for_item = (threadgroup float*)final_sum_aligned;

    // Align G_simd_group_v_sums (already used float4, so alignment is critical)
    uintptr_t g_sums_offset = (uintptr_t)(G_final_sum_exp_for_item + 1);
    uintptr_t g_sums_aligned = (g_sums_offset + 15u) & ~15u;
    threadgroup float4* G_simd_group_v_sums = (threadgroup float4*)g_sums_aligned;

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
        // Zero the full output vector collaboratively
        for (uint i = local_thread_idx; i < params.head_dim; i += tg_dim.x) {
            output_buffer[global_item_idx * params.head_dim + i] = 0.0h;
        }
        return;
    }

    // Get token position and validate
    int item_signed_query_token_offset = query_token_offset_in[token_idx_for_sideband_lookup];
    if (item_signed_query_token_offset < 0) {
        // Zero the full output vector collaboratively
        for (uint i = local_thread_idx; i < params.head_dim; i += tg_dim.x) {
            output_buffer[global_item_idx * params.head_dim + i] = 0.0h;
        }
        return;
    }

    uint item_current_q_token_logical_pos = (uint)item_signed_query_token_offset;
    uint item_actual_sequence_length = (uint)sequence_lengths_in[item_seq_idx_in_batch];
    uint item_effective_history_length = min(item_current_q_token_logical_pos,
                                         item_actual_sequence_length);

    // --- Early exit for zero history case ---
    if (item_effective_history_length == 0) {
        // Write zeros to the entire output V-vector for this item collaboratively
        for (uint i = local_thread_idx; i < params.head_dim; i += tg_dim.x) {
            output_buffer[global_item_idx * params.head_dim + i] = 0.0h;
        }
        return;  // All threads in the group exit early
    }

    // --- Initialize o_local before history scan ---
    // Initialize all params.head_dim components of o_local
    for (uint i = 0; i < params.head_dim; ++i) {
        o_local[i] = 0.0f; // Accesses up to params.head_dim, guarded by C++ validation
    }

    // --- Distribute history tokens among threads in the group ---
    uint num_hist_tokens_per_thread =
        (item_effective_history_length + tg_dim.x - 1) / tg_dim.x;
    uint hist_start_idx = local_thread_idx * num_hist_tokens_per_thread;
    uint hist_end_idx = min((local_thread_idx + 1) * num_hist_tokens_per_thread,
                          item_effective_history_length);

    // --- Single-pass history scan with online softmax and V-aggregation ---
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
        float current_score_fp32 = 0.0f;

        // Use vectorized load if head_dim is a multiple of 4
        bool use_vectorized_load = (params.head_dim % 4 == 0);

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
                    current_score_fp32 += dp;
                }
            } else {
                // Scalar fallback path
                for (uint i = 0; i < params.head_dim; ++i) {
                    current_score_fp32 += q_shmem[i] * (float)k_vector_ptr[i];
                }
            }
        }

        // Note: current_score_fp32 is already scaled because q_shmem was pre-scaled

        // --- Online update for m_local and d_local ---
        float new_m_local = max(m_local, current_score_fp32);
        float alpha = fast::exp(m_local - new_m_local);
        float exponent_for_p = current_score_fp32 - new_m_local;
        // NEW: Branch-free version from O3 that allows underflow to zero
        float p_val = (exponent_for_p < params.log_exp_min_clamp) ? 0.0f : fast::exp(exponent_for_p);

        d_local = d_local * alpha + p_val;
        m_local = new_m_local;

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

        // --- Online o_local update with continuous rescaling for FULL head_dim ---
        bool use_vectorized_v_load = (params.head_dim % 4 == 0);
        if (use_vectorized_v_load) {
            device const packed_half4* v_ptr_h4 = reinterpret_cast<device const packed_half4*>(v_vector_ptr);
            for (uint i = 0; i < params.head_dim / 4; ++i) {
                float4 v_chunk = float4(v_ptr_h4[i]);
                float4 o_chunk_old = float4(o_local[i * 4 + 0], o_local[i * 4 + 1],
                                           o_local[i * 4 + 2], o_local[i * 4 + 3]);
                float4 o_chunk_new = o_chunk_old * alpha + p_val * v_chunk;
                o_local[i * 4 + 0] = o_chunk_new.x;
                o_local[i * 4 + 1] = o_chunk_new.y;
                o_local[i * 4 + 2] = o_chunk_new.z;
                o_local[i * 4 + 3] = o_chunk_new.w;
            }
        } else { // Scalar update
            for (uint i = 0; i < params.head_dim; ++i) {
                o_local[i] = o_local[i] * alpha + p_val * (float)v_vector_ptr[i];
            }
        }
    } // End of history scan loop

    // --- Threadgroup Reduction for m_local (to find m_final_global) ---
    G_partial_max_scores[local_thread_idx] = m_local; // Each thread writes its local max
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float simd_max_m = simd_max(m_local); // All threads in SIMD group get the same max
    if (simd_lane_id == 0 && simd_group_id < num_simd_groups) {
        G_simd_reduced_maxes[simd_group_id] = simd_max_m;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float m_final_global = -INFINITY;
    if (local_thread_idx == 0) {
        if (item_effective_history_length == 0) { // Should be handled by early exit now, but good for safety
            m_final_global = 0.0f;
        } else {
            m_final_global = G_simd_reduced_maxes[0];
            for (uint i = 1; i < num_simd_groups; ++i) {
                m_final_global = max(m_final_global, G_simd_reduced_maxes[i]);
            }
            if (m_final_global == -INFINITY) { // All history chunks were empty/invalid
                m_final_global = 0.0f;
            }
        }
        *G_final_max_for_item = m_final_global; // Store for all threads
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    m_final_global = *G_final_max_for_item; // All threads read the final global max

    // --- Rescale d_local and o_local, then Reduce d_local ---
    // Apply the same branch-free underflow to zero behavior
    float rescale_exponent = m_local - m_final_global;
    float d_local_rescaled = d_local * ((rescale_exponent < params.log_exp_min_clamp) ? 0.0f : fast::exp(rescale_exponent));

    // Use G_simd_reduced_adjusted_sum_exps and G_final_sum_exp_for_item for d_final_global
    float simd_sum_d_rescaled = simd_sum(d_local_rescaled);
    if (simd_lane_id == 0 && simd_group_id < num_simd_groups) {
        G_simd_reduced_adjusted_sum_exps[simd_group_id] = simd_sum_d_rescaled;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float d_final_global = 0.0f;
    if (local_thread_idx == 0) {
        for (uint i = 0; i < num_simd_groups; ++i) {
            d_final_global += G_simd_reduced_adjusted_sum_exps[i];
        }
        *G_final_sum_exp_for_item = d_final_global; // Store for all threads (or just for thread0 use)
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Only thread0 strictly needs d_final_global, but if others might, they can read it:
    // d_final_global = *G_final_sum_exp_for_item;

    // --- Rescale o_local after history scan ---
    // Apply scaling factor to all o_local components with branch-free underflow to zero
    float final_o_rescale_factor = (rescale_exponent < params.log_exp_min_clamp) ? 0.0f : fast::exp(rescale_exponent);
    for (uint i = 0; i < params.head_dim; ++i) {
        o_local[i] *= final_o_rescale_factor;
    }

    // Final normalization factor (inverse of sum of exps)
    float inv_d = (d_final_global > 1e-9f) ? (1.0f / d_final_global) : 0.0f;

    // --- Reduce FULL o_local and Write Output in chunks of 4 ---
    for (uint h_chunk_idx = 0; h_chunk_idx < params.head_dim; h_chunk_idx += 4) {
        // Load from o_local for the current chunk (will be fully accumulated and rescaled already)
        float4 o_chunk_thread_local_rescaled = float4(0.0f);
        if (h_chunk_idx < params.head_dim)     o_chunk_thread_local_rescaled.x = o_local[h_chunk_idx];
        if (h_chunk_idx + 1 < params.head_dim) o_chunk_thread_local_rescaled.y = o_local[h_chunk_idx + 1];
        if (h_chunk_idx + 2 < params.head_dim) o_chunk_thread_local_rescaled.z = o_local[h_chunk_idx + 2];
        if (h_chunk_idx + 3 < params.head_dim) o_chunk_thread_local_rescaled.w = o_local[h_chunk_idx + 3];

        // Use SIMD group reduction to sum this chunk across threads
        float4 simd_sum_chunk;
        simd_sum_chunk.x = simd_sum(o_chunk_thread_local_rescaled.x);
        simd_sum_chunk.y = simd_sum(o_chunk_thread_local_rescaled.y);
        simd_sum_chunk.z = simd_sum(o_chunk_thread_local_rescaled.z);
        simd_sum_chunk.w = simd_sum(o_chunk_thread_local_rescaled.w);

        // SIMD group leaders write to shared memory
        if (simd_lane_id == 0 && simd_group_id < num_simd_groups) {
            G_simd_group_v_sums[simd_group_id] = simd_sum_chunk;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Thread 0 combines SIMD group sums and writes final output
        if (local_thread_idx == 0) {
            float4 o_final_chunk = float4(0.0f);
            for (uint sg = 0; sg < num_simd_groups; ++sg) {
                o_final_chunk += G_simd_group_v_sums[sg];
            }

            // Apply normalization factor and write to output
            o_final_chunk *= inv_d;

            uint out_base = global_item_idx * params.head_dim + h_chunk_idx;
            if (h_chunk_idx < params.head_dim)     output_buffer[out_base] = (half)(o_final_chunk.x);
            if (h_chunk_idx + 1 < params.head_dim) output_buffer[out_base + 1] = (half)(o_final_chunk.y);
            if (h_chunk_idx + 2 < params.head_dim) output_buffer[out_base + 2] = (half)(o_final_chunk.z);
            if (h_chunk_idx + 3 < params.head_dim) output_buffer[out_base + 3] = (half)(o_final_chunk.w);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
} // End of kernel
