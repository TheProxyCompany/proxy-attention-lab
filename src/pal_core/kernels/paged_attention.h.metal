// paged_attention.h.metal
// Metal kernel declaration for paged attention implementation.
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
#include "../include/shaders/paged_attention_types.h"

using namespace metal;

/**
 * @brief Main kernel for paged attention computation.
 *
 * @param queries_in Query vectors [N_tokens × H_q × D] or [N] when H_q==1
 * @param k_cache_pool_in Key cache [Pages × T_pp × H_kv × D]
 * @param v_cache_pool_in Value cache [Pages × T_pp × H_kv × D]
 * @param page_table_in Page mapping table [Seqs × MaxBlocks]
 * @param sequence_lengths_in Sequence lengths [Seqs]
 * @param query_to_seq_map_in Maps queries to sequence indices [N_threads]
 * @param query_token_offset_in Position of each query in its sequence [N_threads]
 * @param params Parameters struct controlling kernel execution
 * @param output_buffer Output buffer for attention results
 * @param tg_pos_in_grid Threadgroup position in grid
 * @param tg_dim Threadgroup dimensions
 * @param local_idx_in_tg Thread index in threadgroup
 * @param simd_lane_id Thread index in SIMD group
 * @param simd_group_id SIMD group index in threadgroup
 */
[[kernel]] void paged_attn_kernel(
    device      const half* queries_in,
    device      const half* k_cache_pool_in,
    device      const half* v_cache_pool_in,
    device      const uint* page_table_in,
    device      const int* sequence_lengths_in,
    device      const int* query_to_seq_map_in,
    device      const int* query_token_offset_in,
    constant    const PagedAttentionParams& params,
    device      half* output_buffer,
    threadgroup float* tg_mem,
    uint3       tg_pos_in_grid,
    uint3       tg_dim,
    uint        local_idx_in_tg,
    uint        simd_lane_id,
    uint        simd_group_id
);
