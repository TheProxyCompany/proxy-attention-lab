#include <metal_stdlib>

#include "paged_attention.h.metal"

using namespace metal;

[[kernel]] void paged_attn_kernel(
    device      const half* queries_in              [[buffer(0)]],
    device      const half* k_cache_pool_in         [[buffer(1)]],
    device      const half* v_cache_pool_in         [[buffer(2)]],
    device      const uint* page_table_in           [[buffer(3)]],
    device      const int* sequence_lengths_in      [[buffer(4)]],
    device      const int* query_to_seq_map_in      [[buffer(5)]],
    device      const int* query_token_offset_in    [[buffer(6)]],
    constant    const PagedAttentionParams* params  [[buffer(7)]],
    device      half* output_buffer                 [[buffer(8)]],
    uint3       thread_pos_in_grid                  [[thread_position_in_grid]],
    uint3       grid_size                           [[threads_per_grid]]
) {
    uint global_query_thread_idx = thread_pos_in_grid.x;
    if (global_query_thread_idx >= grid_size.x) {
        return; // Thread is out of bounds for the query data processed by this dispatch
    }

    half query_val = queries_in[global_query_thread_idx];

    // 1. Determine sequence index for this thread
    uint seq_idx_in_batch = (uint)query_to_seq_map_in[global_query_thread_idx];

    // 2. For this test, we always fetch K for logical_token_pos 0 of the sequence.
    uint target_historical_logical_token_pos = 0;
    uint logical_block_idx = target_historical_logical_token_pos / params->tokens_per_page; // Should be 0
    uint token_slot_in_page = target_historical_logical_token_pos % params->tokens_per_page;    // Should be 0

    // 3. Get physical_page_id from page_table_in
    //    page_table_in is effectively [NumBatchSeq][MaxLogicalBlocksPerSeq]
    //    The C++ primitive sets params->max_logical_blocks_per_seq from page_table.shape(1).
    uint page_table_flat_idx = seq_idx_in_batch * params->max_logical_blocks_per_seq + logical_block_idx;
    uint physical_page_id = page_table_in[page_table_flat_idx];

    // 4. Calculate address of the target K vector element
    //    K data for: physical_page_id, token_slot_in_page, target_kv_head_idx = 0, target_element_in_head = 0
    uint target_kv_head_idx = 0;
    uint target_element_in_head = 0;

    // params->head_dim is now correctly the K/V data head dimension.
    uint k_elements_per_token_slot_per_kv_head = params->head_dim;
    uint k_elements_per_token_slot_all_kv_heads = params->num_kv_heads * k_elements_per_token_slot_per_kv_head;
    uint k_elements_per_physical_page = params->tokens_per_page * k_elements_per_token_slot_all_kv_heads;

    uint k_page_base_offset_in_elements = physical_page_id * k_elements_per_physical_page;

    uint k_token_slot_base_offset_in_elements = token_slot_in_page * k_elements_per_token_slot_all_kv_heads;

    uint k_kv_head_base_offset_in_elements = target_kv_head_idx * params->head_dim; // Using params->head_dim

    uint final_k_flat_idx = k_page_base_offset_in_elements + \
                            k_token_slot_base_offset_in_elements + \
                            k_kv_head_base_offset_in_elements + \
                            target_element_in_head;

    half k_element_val = k_cache_pool_in[final_k_flat_idx];

    // 5. Output for this test
    output_buffer[global_query_thread_idx] = query_val + k_element_val;
}
