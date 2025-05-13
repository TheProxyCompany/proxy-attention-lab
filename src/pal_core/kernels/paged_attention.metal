//  Proxy Attention Lab – Paged‑attention kernel
//  Finds the maximum scaled dot product between a Q-vector and K-vectors in history.
//  This version handles GQA, 64‑bit indexing, and runtime bounds checks.
//  Author: The Proxy Company

#include <metal_stdlib>
#include "paged_attention.h.metal"

using namespace metal;

// Write zero scalar for the current thread – keeps NaNs out on early exit
static void zero_out_thread_output(
    device half* output_buffer,
    uint global_query_thread_idx
) {
    output_buffer[global_query_thread_idx] = 0.0h;
}


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
    device      const half* v_cache_pool_in         [[buffer(2)]], // Unused in this iteration
    device      const uint* page_table_in           [[buffer(3)]],
    device      const int* sequence_lengths_in      [[buffer(4)]],
    device      const int* query_to_seq_map_in      [[buffer(5)]],
    device      const int* query_token_offset_in    [[buffer(6)]],
    constant    const PagedAttentionParams* params  [[buffer(7)]],
    device      half* output_buffer                 [[buffer(8)]],
    uint3       thread_pos_in_grid                  [[thread_position_in_grid]],
    uint3       grid_size                           [[threads_per_grid]]
) {
    (void)v_cache_pool_in;
    // sequence_lengths_in is now being used, so no need for the (void) cast

    uint global_query_thread_idx = thread_pos_in_grid.x;

    if (global_query_thread_idx >= grid_size.x) {
        return;
    }

    // --- Determine Q-vector information for this thread ---
    device const half* q_vector_ptr;
    uint token_idx = 0;
    uint q_head_idx = 0;

    if (params->num_q_heads > 1) {
        token_idx = global_query_thread_idx / params->num_q_heads;
        q_head_idx = global_query_thread_idx % params->num_q_heads;
        ulong query_base_offset = (ulong)token_idx * params->num_q_heads * params->head_dim +
                                  (ulong)q_head_idx * params->head_dim;
        q_vector_ptr = queries_in + query_base_offset;
    } else {
        q_vector_ptr = queries_in + (global_query_thread_idx * params->head_dim);
    }

    // --- Sequence and Token Mapping ---
    uint seq_idx_in_batch = (uint)query_to_seq_map_in[global_query_thread_idx];
    if (seq_idx_in_batch >= params->num_sequences_in_batch) {
        zero_out_thread_output(output_buffer, global_query_thread_idx);
        return;
    }

    int signed_query_token_offset = query_token_offset_in[global_query_thread_idx];
    if (signed_query_token_offset < 0) {
        zero_out_thread_output(output_buffer, global_query_thread_idx);
        return;
    }

    // --- Get current token position and sequence length ---
    uint current_q_token_logical_pos = (uint)signed_query_token_offset;
    uint actual_sequence_length = (uint)sequence_lengths_in[seq_idx_in_batch];

    // --- Determine effective history length ---
    // History is from token 0 up to, but not including, current_q_token_logical_pos.
    // Also, cannot exceed actual_sequence_length.
    uint effective_history_length = min(current_q_token_logical_pos, actual_sequence_length);

    // --- KV Head Selection ---
    uint current_q_head_idx_for_kv_map = 0; // Default for 1D/2D queries or single Q-head case
    if (params->num_q_heads > 1) { // This implies a 3D query input where Q-heads are explicit
        current_q_head_idx_for_kv_map = global_query_thread_idx % params->num_q_heads;
    }
    // For 1D/2D queries, params->num_q_heads is 1. current_q_head_idx_for_kv_map remains 0.
    // This ensures that for MQA with 2D queries, we always try to map "the first" Q-head.

    uint target_kv_head_idx = 0;
    if (params->num_kv_heads > 0) { // Must have at least one KV head
        if (params->num_q_heads >= params->num_kv_heads) { // GQA (num_q_heads > num_kv_heads) or MHA (num_q_heads == num_kv_heads)
            // num_q_heads % num_kv_heads == 0 is validated on C++ side for GQA.
            uint gqa_factor = params->num_q_heads / params->num_kv_heads;
            target_kv_head_idx = current_q_head_idx_for_kv_map / gqa_factor;
        } else { // MQA: num_q_heads < num_kv_heads
                 // This includes the case where C++ set params->num_q_heads = 1 for 2D queries.
                 // In this MQA case, query head 'i' maps to key/value head 'i'.
                 // Since current_q_head_idx_for_kv_map will be 0 if params->num_q_heads was 1,
                 // this correctly maps to target_kv_head_idx = 0.
            target_kv_head_idx = current_q_head_idx_for_kv_map;
        }

        // Final safety clamp, though the logic above should ensure target_kv_head_idx is valid.
        if (target_kv_head_idx >= params->num_kv_heads) {
             target_kv_head_idx = target_kv_head_idx % params->num_kv_heads; // Or handle as an error/default to 0
        }
    }

    // --- Initialize max score to negative infinity for finding maximum ---
    float max_score_float = -INFINITY;  // Accumulate in float for better precision

    // If no history to process, set default score
    if (effective_history_length == 0) {
        max_score_float = 0.0f;
    } else {
        // --- Loop through history positions ---
        for (uint hist_token_idx = 0; hist_token_idx < effective_history_length; ++hist_token_idx) {
            uint target_historical_logical_token_pos = hist_token_idx;

            // Calculate logical block and token slot
            uint logical_block_idx = target_historical_logical_token_pos / params->tokens_per_page;
            uint token_slot_in_page = target_historical_logical_token_pos % params->tokens_per_page;

            // Check if logical block index is valid
            if (logical_block_idx >= params->max_logical_blocks_per_seq) {
                // This implies we've gone beyond the blocks described in the page table
                // for this sequence's history, so we break rather than continue
                break; // No more valid blocks for this sequence's history
            }

            // --- Page Table Lookup ---
            uint page_table_flat_idx = seq_idx_in_batch * params->max_logical_blocks_per_seq + logical_block_idx;
            uint physical_page_id = page_table_in[page_table_flat_idx];

            if (physical_page_id >= params->num_physical_pages_in_pool) {
                continue; // Skip this history position - invalid physical page
            }

            // --- K-Vector Address Calculation ---
            ulong k_elements_per_token_slot_per_kv_head = (ulong)params->head_dim;
            ulong k_elements_per_token_slot_all_kv_heads = (ulong)params->num_kv_heads * k_elements_per_token_slot_per_kv_head;
            ulong k_elements_per_physical_page = (ulong)params->tokens_per_page * k_elements_per_token_slot_all_kv_heads;

            ulong k_page_base_offset_in_elements = (ulong)physical_page_id * k_elements_per_physical_page;
            ulong k_token_slot_base_offset_in_elements = (ulong)token_slot_in_page * k_elements_per_token_slot_all_kv_heads;
            ulong k_kv_head_base_offset_in_elements = (ulong)target_kv_head_idx * k_elements_per_token_slot_per_kv_head;

            ulong k_vector_start_idx = k_page_base_offset_in_elements +
                                       k_token_slot_base_offset_in_elements +
                                       k_kv_head_base_offset_in_elements;

            // --- Get pointer to K-vector for this history position ---
            device const half* k_vector_ptr = k_cache_pool_in + k_vector_start_idx;

            // --- Compute Dot Product Q·K^T for this history token ---
            float current_score_float = 0.0f; // Accumulate in float for precision

            // C++ primitive now ensures params->head_dim > 0 and params->head_dim % 4 == 0.
            // Check 8-byte alignment for packed_half4 access
#ifdef PAL_DEBUG
            // Note: Metal kernel pointers using packed_half4 should be 8-byte aligned for optimal performance
            // The base arrays should be aligned by MLX's allocator
            // If we had a critical path, we could add a fallback to scalar path here
            bool q_ptr_aligned = (((uintptr_t)q_vector_ptr & 0x7) == 0);
            bool k_ptr_aligned = (((uintptr_t)k_vector_ptr & 0x7) == 0);
            // Metal doesn't support thread_printf in this context
            // This alignment check is for development only
#endif
            device const packed_half4* q_ptr_h4 = reinterpret_cast<device const packed_half4*>(q_vector_ptr);
            device const packed_half4* k_ptr_h4 = reinterpret_cast<device const packed_half4*>(k_vector_ptr);

            for (uint i = 0; i < params->head_dim / 4; ++i) {
                float4 q_vec_f4 = float4(q_ptr_h4[i]);
                float4 k_vec_f4 = float4(k_ptr_h4[i]);
                current_score_float += dot(q_vec_f4, k_vec_f4);
            }

            // Apply scale
            current_score_float *= params->scale;

            // Update max score using float comparison for better precision
            max_score_float = max(max_score_float, current_score_float);
        }
    }

    // If we processed history but didn't find any valid vectors (all were skipped),
    // max_score_float would still be -INFINITY
    if (max_score_float == -INFINITY) {
        max_score_float = 0.0f;
    }

    // Write max score to output, converting to half at the end
    output_buffer[global_query_thread_idx] = (half)max_score_float;
}
