#include <metal_stdlib>
#include "../include/shaders/paged_attention_types.h"

using namespace metal;

[[kernel]] void paged_attn_kernel(
    device      const half* queries_in,
    device      const half* k_cache_pool_in,
    device      const half* v_cache_pool_in,
    device      const uint* page_table_in,
    device      const int* sequence_lengths_in,
    device      const int* query_to_seq_map_in,
    device      const int* query_token_offset_in,
    constant    const PagedAttentionParams* params,
    device      half* output_buffer,
    uint3       thread_pos_in_grid,
    uint3       grid_size
);
