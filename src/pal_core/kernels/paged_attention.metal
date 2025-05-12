//  Proxy Attention Lab – Paged‑attention kernel
//  Fetches a K‑vector for a (token,q‑head) pair and writes a reduced result.
//  This version handles GQA, 64‑bit indexing, and runtime bounds checks.
//  Author: The Proxy Company

#include <metal_stdlib>
#include "paged_attention.h.metal"

using namespace metal;

// Write zero(s) for the current thread – keeps NaNs out on early exit
static void zero_out_thread_output(
    device half* output_buffer,
    uint global_query_thread_idx,
    constant const PagedAttentionParams* params
) {
    if (params->num_q_heads > 1) {
        uint token_idx = global_query_thread_idx / params->num_q_heads;
        uint q_head_idx = global_query_thread_idx % params->num_q_heads;
        ulong out_base_offset = (ulong)token_idx * params->num_q_heads * params->head_dim +
                               (ulong)q_head_idx * params->head_dim;
        for (uint i = 0; i < params->head_dim; ++i) {
            output_buffer[out_base_offset + i] = 0.0h;
        }
    } else {
        output_buffer[global_query_thread_idx] = 0.0h;
    }
}


/**
 *  paged_attn_kernel
 *  ----------------------------------------
 *  Parameters (buffers):
 *    0  queries_in      – half[N_tokens × H_q × D] or half[N] when H_q==1
 *    1  k_cache_pool_in – half[Pages × T_pp × H_kv × D]
 *    2  v_cache_pool_in – half[…], currently unused
 *    3  page_table_in   – uint[Seqs × MaxBlocks]
 *    4  sequence_lengths_in – int[Seqs] (bounds checks, optional)
 *    5  query_to_seq_map_in – int[N_threads]
 *    6  query_token_offset_in – int[N_threads]
 *    7  PagedAttentionParams (constant)
 *    8  output_buffer   – half[same layout as queries_in]
 *
 *  One thread processes exactly one (token,q‑head) pair.
 *  On invalid indices the thread writes zeros (safe‑exit contract).
 */
[[kernel]] void paged_attn_kernel(
    device      const half* queries_in              [[buffer(0)]],
    device      const half* k_cache_pool_in         [[buffer(1)]],
    device      const half* v_cache_pool_in         [[buffer(2)]], // Unused in this iteration
    device      const uint* page_table_in           [[buffer(3)]],
    device      const int* sequence_lengths_in      [[buffer(4)]], // Unused in this iteration's core logic
    device      const int* query_to_seq_map_in      [[buffer(5)]],
    device      const int* query_token_offset_in    [[buffer(6)]],
    constant    const PagedAttentionParams* params  [[buffer(7)]],
    device      half* output_buffer                 [[buffer(8)]],
    uint3       thread_pos_in_grid                  [[thread_position_in_grid]],
    uint3       grid_size                           [[threads_per_grid]]
) {
    (void)v_cache_pool_in;
    (void)sequence_lengths_in;
    (void)params->scale;

    uint global_query_thread_idx = thread_pos_in_grid.x;

    if (global_query_thread_idx >= grid_size.x) {
        return;
    }

    // --- Determine Q-vector information for this thread ---
    device const half* q_vector_for_thread_start_ptr;
    half query_first_element_val;

    if (params->num_q_heads > 1) {
        uint token_idx = global_query_thread_idx / params->num_q_heads;
        uint q_head_idx = global_query_thread_idx % params->num_q_heads;
        ulong query_base_offset = (ulong)token_idx * params->num_q_heads * params->head_dim +
                                  (ulong)q_head_idx * params->head_dim;
        q_vector_for_thread_start_ptr = queries_in + query_base_offset;
        query_first_element_val = q_vector_for_thread_start_ptr[0];
    } else {
        q_vector_for_thread_start_ptr = queries_in + (global_query_thread_idx * params->head_dim);
        query_first_element_val = queries_in[global_query_thread_idx];
    }

    // --- Sequence and Token Mapping ---
    uint seq_idx_in_batch = (uint)query_to_seq_map_in[global_query_thread_idx];
    if (seq_idx_in_batch >= params->num_sequences_in_batch) {
        zero_out_thread_output(output_buffer, global_query_thread_idx, params);
        return;
    }

    int signed_query_token_offset = query_token_offset_in[global_query_thread_idx];
    if (signed_query_token_offset < 0) {
        zero_out_thread_output(output_buffer, global_query_thread_idx, params);
        return;
    }
    uint target_historical_logical_token_pos = (uint)signed_query_token_offset;

    uint logical_block_idx = target_historical_logical_token_pos / params->tokens_per_page;
    uint token_slot_in_page = target_historical_logical_token_pos % params->tokens_per_page;

    if (logical_block_idx >= params->max_logical_blocks_per_seq) {
        zero_out_thread_output(output_buffer, global_query_thread_idx, params);
        return;
    }

    // --- Page Table Lookup ---
    uint page_table_flat_idx = seq_idx_in_batch * params->max_logical_blocks_per_seq + logical_block_idx;
    uint physical_page_id = page_table_in[page_table_flat_idx];

    if (physical_page_id >= params->num_physical_pages_in_pool) {
        zero_out_thread_output(output_buffer, global_query_thread_idx, params);
        return;
    }

    // --- KV Head Selection ---
    uint current_q_head_idx_for_kv_map = 0;
    if (params->num_q_heads > 1) {
        current_q_head_idx_for_kv_map = global_query_thread_idx % params->num_q_heads;
    } else {
        current_q_head_idx_for_kv_map = global_query_thread_idx % params->num_kv_heads;
    }

    uint target_kv_head_idx = 0;
    if (params->num_kv_heads > 0) {
        if (params->num_q_heads >= params->num_kv_heads) {
            uint gqa_factor = params->num_q_heads / params->num_kv_heads;
            target_kv_head_idx = current_q_head_idx_for_kv_map / gqa_factor;
        } else {
            target_kv_head_idx = current_q_head_idx_for_kv_map;
        }
        if (target_kv_head_idx >= params->num_kv_heads) {
             target_kv_head_idx = 0;
        }
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

    // --- Fetch and Sum K-Vector ---
    half k_vector_sum = 0.0h;
    for (uint element_idx = 0; element_idx < params->head_dim; ++element_idx) {
        k_vector_sum += k_cache_pool_in[k_vector_start_idx + element_idx];
    }

    // --- Output Calculation ---
    if (params->num_q_heads > 1) {
        uint token_idx = global_query_thread_idx / params->num_q_heads;
        uint q_head_idx = global_query_thread_idx % params->num_q_heads;
        ulong out_base_offset = (ulong)token_idx * params->num_q_heads * params->head_dim +
                               (ulong)q_head_idx * params->head_dim;
        for (uint i = 0; i < params->head_dim; ++i) {
            output_buffer[out_base_offset + i] = query_first_element_val + k_vector_sum;
        }
    } else {
        output_buffer[global_query_thread_idx] = query_first_element_val + k_vector_sum;
    }
}
