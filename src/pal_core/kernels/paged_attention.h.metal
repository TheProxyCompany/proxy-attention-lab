#include <metal_stdlib>
#include "../include/shaders/paged_attention_types.h"

using namespace metal;

[[kernel]] void paged_attn_kernel(
    device      const half* queries_in               [[buffer(0)]],
    device      const half* k_cache_pool_in          [[buffer(1)]], // Global K data
    device      const half* v_cache_pool_in          [[buffer(2)]], // Global V data
    device      const uint* page_table_in            [[buffer(3)]], // Contains physical_page_ids
    device      const int* sequence_lengths_in       [[buffer(4)]], // Actual length of each seq in batch
    device      const int* query_to_seq_map_in       [[buffer(5)]], // Maps global query token index to its seq_idx_in_batch
    device      const int* query_token_offset_in     [[buffer(6)]], // Logical offset of Q token within its sequence
    constant    const PagedAttentionParams* params   [[buffer(7)]], // Struct with config
    device      half* output_buffer                  [[buffer(8)]], // Output
    uint        global_query_thread_idx              [[thread_position_in_grid]]
);
