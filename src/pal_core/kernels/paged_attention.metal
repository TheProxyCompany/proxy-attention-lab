#include <metal_stdlib>

#include "paged_attention.h.metal"

using namespace metal;

[[kernel]] void paged_attn_kernel(
    device const half* queries_in             [[buffer(0)]],
    device const half* k_cache_pool_in        [[buffer(1)]], // MATCHED
    device const half* v_cache_pool_in        [[buffer(2)]], // MATCHED
    device const uint* page_table_in          [[buffer(3)]],
    device const int* sequence_lengths_in    [[buffer(4)]], // ADDED
    device const int* query_to_seq_map_in    [[buffer(5)]], // ADDED
    device const int* query_token_offset_in  [[buffer(6)]], // ADDED
    constant const PagedAttentionParams* params [[buffer(7)]], // MATCHED index
    device       half* output_buffer            [[buffer(8)]], // MATCHED index
    uint global_query_thread_idx          [[thread_position_in_grid]] // MATCHED name
) {
    // For this first test, let's just use params->head_dim and queries_in.
    // We are not using k_cache_pool_in, v_cache_pool_in, page_table_in,
    // sequence_lengths_in, query_to_seq_map_in, query_token_offset_in yet.

    // Using global_query_thread_idx now
    if (global_query_thread_idx < 10 && global_query_thread_idx < params->head_dim) {
        output_buffer[global_query_thread_idx] = queries_in[global_query_thread_idx] + (half)(params->head_dim);
    } else if (global_query_thread_idx < 10) {
        output_buffer[global_query_thread_idx] = queries_in[global_query_thread_idx];
    }
    else {
        output_buffer[global_query_thread_idx] = 0.0h;
    }
}
