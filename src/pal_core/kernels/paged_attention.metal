//  Proxy Attention Lab – Paged‑attention kernel
//  Finds the maximum scaled dot product between a Q-vector and K-vectors in history.
//  This version handles GQA, 64‑bit indexing, and runtime bounds checks.
//  Author: The Proxy Company

#include <metal_stdlib>
#include "paged_attention.h.metal"

using namespace metal;

// Compile-time constants for kernel configuration
constant static const uint MAX_SIMD_GROUPS_PER_TG = 8; // Max possible SIMD groups (256/32 = 8)


/**
 *  paged_attn_kernel
 *  ----------------------------------------
 *  Parameters (buffers):
 *    0  queries_in      – half[N_tokens × H_q × D] or half[N] when H_q==1
 *    1  k_cache_pool_in – half[Pages × T_pp × H_kv × D]
 *    2  v_cache_pool_in – half[…], currently unused
 *    3  page_table_in   – uint[Seqs × MaxBlocks]
 *    4  sequence_lengths_in – int[Seqs] (bounds checking)
 *    5  query_to_seq_map_in – int[N_threads]
 *    6  query_token_offset_in – int[N_threads]
 *    7  PagedAttentionParams (constant)
 *    8  output_buffer   – half[same layout as queries_in]
 *
 *  One thread processes exactly one (token,q‑head) pair.
 *  For each pair, the kernel:
 *  1. Identifies the current token position (from query_token_offset_in)
 *  2. Loops through historical token positions from 0 to current position - 1
 *  3. Computes the scaled dot product with each historical K-vector
 *  4. Finds and returns the maximum score
 *  On invalid indices the thread writes zeros (safe‑exit contract).
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
    (void)v_cache_pool_in;
    (void)simd_lane_id; (void)simd_group_id; (void)tg_dim;

    // --- Thread Identifiers ---
    uint global_item_idx = tg_pos_in_grid.x;    // Identifies the query-head item
    uint local_thread_idx = local_idx_in_tg;    // Thread ID within this group (0 to THREADS_PER_ITEM_GROUP_CONST - 1)


    // --- Carve the dynamic threadgroup buffer into logical sub-arrays ---
    threadgroup float* q_shmem                           = tg_mem;                    // head_dim floats
    threadgroup float* G_partial_max_scores              = q_shmem + params.head_dim; // threads_per_tg floats
    threadgroup float* G_simd_reduced_maxes              = G_partial_max_scores + tg_dim.x;
    threadgroup float* G_simd_reduced_adjusted_sum_exps  = G_simd_reduced_maxes + MAX_SIMD_GROUPS_PER_TG;
    threadgroup float* G_final_max_for_item              = G_simd_reduced_adjusted_sum_exps + MAX_SIMD_GROUPS_PER_TG;

    // --- Determine Q-vector pointer for this item ---
    // This logic is adapted from the original single-thread-per-item kernel.
    // 'params' struct is now required.
    device const half* q_vector_item_ptr;
    if (params.num_q_heads > 1) { // Indicates original 3D Q array [Tokens, QHeads, Dim]
                                   // global_item_idx = token_idx * num_q_heads + q_head_idx
        uint item_token_idx = global_item_idx / params.num_q_heads;
        uint item_q_head_idx = global_item_idx % params.num_q_heads;
        ulong query_base_offset = (ulong)item_token_idx * params.num_q_heads * params.head_dim +
                                  (ulong)item_q_head_idx * params.head_dim;
        q_vector_item_ptr = queries_in + query_base_offset;
    } else { // Original Q was 1D/2D, params.num_q_heads = 1. global_item_idx is the direct item index.
        q_vector_item_ptr = queries_in + (global_item_idx * params.head_dim);
    }

    // --- Stage Q-vector into Shared Memory ---
    // Each thread in the group cooperatively loads a portion.
    // Since we're using dynamic threadgroup memory allocation, no need to check against a fixed limit.
    // The C++ side will ensure the requested memory size is within device limits.

    for (uint i = local_thread_idx; i < params.head_dim; i += tg_dim.x) {
        q_shmem[i] = (float)q_vector_item_ptr[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup); // Synchronize after Q-staging

    // --- Determine this item's overall history and sequence length ---
    // For 3D queries [NumTokens, NumQHeads, HeadDim], query_to_seq_map and query_token_offset
    // map to tokens, not to the combined (token,head) pairs
    uint token_idx_for_sideband_lookup;
    if (params.num_q_heads > 1) { // 3D queries case
        token_idx_for_sideband_lookup = global_item_idx / params.num_q_heads;
    } else { // 1D/2D queries case - direct mapping
        token_idx_for_sideband_lookup = global_item_idx;
    }

    uint item_seq_idx_in_batch = (uint)query_to_seq_map_in[token_idx_for_sideband_lookup];
    if (item_seq_idx_in_batch >= params.num_sequences_in_batch) {
        if (local_thread_idx == 0) { output_buffer[global_item_idx] = 0.0h; }
        return;
    }

    int item_signed_query_token_offset = query_token_offset_in[token_idx_for_sideband_lookup];
    if (item_signed_query_token_offset < 0) {
        if (local_thread_idx == 0) { output_buffer[global_item_idx] = 0.0h; }
        return;
    }
    uint item_current_q_token_logical_pos = (uint)item_signed_query_token_offset;
    uint item_actual_sequence_length = (uint)sequence_lengths_in[item_seq_idx_in_batch];
    uint item_effective_history_length = min(item_current_q_token_logical_pos, item_actual_sequence_length);

    // --- Parallel History Scan Setup ---
    float thread_local_max_score = -INFINITY;
    float thread_local_sum_exp = 0.0f;
    bool thread_processed_any_valid_score = false;
    bool thread_first_valid_score_in_chunk = true;

    if (item_effective_history_length > 0) {
        // Distribute history tokens among threads in the group
        // Each thread processes approx. item_effective_history_length / tg_dim.x tokens
        uint num_hist_tokens_per_thread = (item_effective_history_length + tg_dim.x - 1) / tg_dim.x;
        uint hist_start_idx = local_thread_idx * num_hist_tokens_per_thread;
        uint hist_end_idx = min((local_thread_idx + 1) * num_hist_tokens_per_thread, item_effective_history_length);

        // --- This thread's loop over its assigned history chunk ---
        for (uint hist_token_idx = hist_start_idx; hist_token_idx < hist_end_idx; ++hist_token_idx) {
            uint target_historical_logical_token_pos = hist_token_idx;

            uint logical_block_idx = target_historical_logical_token_pos / params.tokens_per_page;
            uint token_slot_in_page = target_historical_logical_token_pos % params.tokens_per_page;

            if (logical_block_idx >= params.max_logical_blocks_per_seq) {
                break; // This thread's chunk goes beyond page table for this item
            }

            uint page_table_flat_idx = item_seq_idx_in_batch * params.max_logical_blocks_per_seq + logical_block_idx;
            uint physical_page_id = page_table_in[page_table_flat_idx];

            if (physical_page_id >= params.num_physical_pages_in_pool) {
                continue; // Skip this history position - invalid physical page
            }

            // KV Head Selection (adapted from original kernel - uses global_item_idx to determine its Q-head nature)
            uint q_head_for_kv_map_within_item = 0;
            if (params.num_q_heads > 1) { // Original Q was 3D
                q_head_for_kv_map_within_item = global_item_idx % params.num_q_heads;
            } // else: original Q was 1D/2D, params.num_q_heads=1, so effective q_head_for_kv_map is 0.

            uint target_kv_head_idx = 0;
            if (params.num_kv_heads > 0) {
                if (params.num_q_heads > params.num_kv_heads) { // GQA (num_q_heads > num_kv_heads)
                    uint gqa_factor = params.num_q_heads / params.num_kv_heads;
                    target_kv_head_idx = q_head_for_kv_map_within_item / gqa_factor;
                } else if (params.num_q_heads < params.num_kv_heads) { // MQA (num_q_heads < num_kv_heads)
                    target_kv_head_idx = 0; // For MQA, always use kv_head 0 regardless of q_head
                } else { // MHA (num_q_heads == num_kv_heads)
                    target_kv_head_idx = q_head_for_kv_map_within_item; // Direct 1:1 mapping for MHA
                }
                if (target_kv_head_idx >= params.num_kv_heads) { // Safety
                     target_kv_head_idx = target_kv_head_idx % params.num_kv_heads;
                }
            }

            // K-Vector Address Calculation (from original kernel)
            ulong k_elements_per_token_slot_per_kv_head = (ulong)params.head_dim;
            ulong k_elements_per_token_slot_all_kv_heads = (ulong)params.num_kv_heads * k_elements_per_token_slot_per_kv_head;
            ulong k_elements_per_physical_page = (ulong)params.tokens_per_page * k_elements_per_token_slot_all_kv_heads;
            ulong k_page_base_offset_in_elements = (ulong)physical_page_id * k_elements_per_physical_page;
            ulong k_token_slot_base_offset_in_elements = (ulong)token_slot_in_page * k_elements_per_token_slot_all_kv_heads;
            ulong k_kv_head_base_offset_in_elements = (ulong)target_kv_head_idx * k_elements_per_token_slot_per_kv_head;
            ulong k_vector_start_idx = k_page_base_offset_in_elements +
                                       k_token_slot_base_offset_in_elements +
                                       k_kv_head_base_offset_in_elements;
            device const half* k_vector_ptr = k_cache_pool_in + k_vector_start_idx;

            // --- Compute Dot Product Q·K^T ---
            // Q is from q_shmem, K is from k_vector_ptr
            float current_score_float = 0.0f;

            // Alignment check for K (Q from q_shmem is float, no packed_half4 for Q here yet)
            #ifdef PAL_DEBUG
            if (((uintptr_t)k_vector_ptr & 0x7) != 0 && (params.head_dim % 4 == 0)) {
                // Potential performance issue if not aligned for packed_half4 load
            }
            #endif

            // Assuming params.head_dim is a multiple of 4 (validated in C++)
            // And params.head_dim <= MAX_HEAD_DIM_FOR_SHARED_MEM
            if (params.head_dim > 0) { // Ensure head_dim is positive
                device const packed_half4* k_ptr_h4 = reinterpret_cast<device const packed_half4*>(k_vector_ptr);
                for (uint i = 0; i < params.head_dim / 4; ++i) {
                    // Load 4 half K elements, convert to float4
                    float4 k_vec_f4 = float4(k_ptr_h4[i]);
                    // Q is already float in q_shmem. Load 4 floats.
                    // This assumes q_shmem can be treated as float4 array implicitly
                    // Or load individually: float4 q_vec_f4 = float4(q_shmem[i*4], q_shmem[i*4+1], ...);
                    float4 q_vec_f4 = {q_shmem[i * 4 + 0], q_shmem[i * 4 + 1], q_shmem[i * 4 + 2], q_shmem[i * 4 + 3]};

                    float dp = dot(q_vec_f4, k_vec_f4);
                    current_score_float += dp;
                }
            }

            current_score_float *= params.scale;

            // Online Log-Sum-Exp update for this thread's local accumulation
            if (thread_first_valid_score_in_chunk) {
                thread_local_max_score = current_score_float;
                thread_local_sum_exp = 1.0f;
                thread_first_valid_score_in_chunk = false;
            } else {
                float new_potential_max = max(thread_local_max_score, current_score_float);
                thread_local_sum_exp = thread_local_sum_exp * exp(max(thread_local_max_score - new_potential_max, -16.0f)) +
                                    exp(max(current_score_float - new_potential_max, -16.0f));
                thread_local_max_score = new_potential_max;
            }

            thread_processed_any_valid_score = true;
        } // end for hist_token_idx
    } // end if item_effective_history_length > 0

    // --- Initialize Shared Memory for Reduction ---
    // Each thread writes its local max score to its slot in G_partial_max_scores
    if (thread_processed_any_valid_score) {
        G_partial_max_scores[local_thread_idx] = thread_local_max_score;
    } else {
        // If this thread had a non-empty chunk assigned but found no valid KVs,
        // or if its chunk was empty (hist_start_idx >= hist_end_idx).
        G_partial_max_scores[local_thread_idx] = -INFINITY;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup); // Ensure all local maxes are written

    // --- Perform Threadgroup Reduction for Max Score ---
    // 1. SIMD-group level reduction
    float simd_max_val = simd_max(thread_local_max_score); // All threads in SIMD group get the same max

    if (simd_lane_id == 0) { // One thread per SIMD group writes its group's max
        G_simd_reduced_maxes[simd_group_id] = simd_max_val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 2. Reduce across SIMD group results (thread 0 does the final reduction)
    float final_global_max_score = -INFINITY;
    if (local_thread_idx == 0) {
        if (item_effective_history_length == 0) { // Handle no history for the item
            final_global_max_score = 0.0f;
        } else {
            final_global_max_score = G_simd_reduced_maxes[0]; // Start with first SIMD group's max

            // Use the actual number of SIMD groups in this threadgroup: tg_dim.x / 32
            uint num_simd_groups = (tg_dim.x + 31) / 32; // Ceiling division by 32
            for (uint i = 1; i < num_simd_groups; ++i) {
                final_global_max_score = max(final_global_max_score, G_simd_reduced_maxes[i]);
            }

            // If after all reductions, max is still -INF (all history chunks were empty/invalid) set to 0.
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
        // If thread_local_max_score < final_max_score_for_item_from_shared, this reduces the thread's contribution
        adjusted_thread_local_sum_exp = thread_local_sum_exp * exp(max_diff);
    }

    // SIMD-group sum for the adjusted sum_exp values
    float simd_sum_val = simd_sum(adjusted_thread_local_sum_exp);

    if (simd_lane_id == 0) { // One thread per SIMD group writes its group's sum
        G_simd_reduced_adjusted_sum_exps[simd_group_id] = simd_sum_val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Thread 0 computes final global sum_exp_score
    float final_global_sum_exp_score = 0.0f;
    if (local_thread_idx == 0) {
        // Use the actual number of SIMD groups in this threadgroup: tg_dim.x / 32
        uint num_simd_groups = (tg_dim.x + 31) / 32; // Ceiling division by 32
        for (uint i = 0; i < num_simd_groups; ++i) {
            final_global_sum_exp_score += G_simd_reduced_adjusted_sum_exps[i];
        }

        // --- Output in planar layout ---
        // Always write the proper max score for the item
        output_buffer[global_item_idx] = (half)final_global_max_score;
        uint output_pitch = params.total_items_in_dispatch;
        if (output_pitch == 0) output_pitch = 1;
        // Don't check size since we don't have that method
        output_buffer[global_item_idx + output_pitch] = (half)final_global_sum_exp_score;
    }
} // End of kernel body
