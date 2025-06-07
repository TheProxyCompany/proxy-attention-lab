// paged_attention_kernels.metal
// Instantiates specialized versions of PAL kernels from template headers.
//
// Copyright 2025 The Proxy Company. All Rights Reserved.
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

#include "fill_kv_pages.h.metal"
#include "paged_attention_fused.h.metal"
// #include "paged_attention_2pass.h.metal"

[[kernel]] void get_device_info() {
    // used for fetching a metal compute pipeline state
    // for the current device to get the max threads per group
    // and simd group size
}

// --- Instantiation Macro for fill_kv_pages ---
// This macro creates a specialized version of the kernel with a unique host name
// that the C++ code can look up at runtime.
#define INSTANTIATE_FILL_KV_PAGES(TYPE, SUFFIX)                                                                                 \
    template [[host_name("fill_kv_pages_kernel_" #SUFFIX)]] [[kernel]] void                                                     \
    fill_kv_pages_kernel<TYPE>(                                                                                                 \
        device const    TYPE*   new_keys                    [[buffer(0)]],                                                      \
        device const    TYPE*   new_values                  [[buffer(1)]],                                                      \
        device const    uint*   page_table_data             [[buffer(2)]],                                                      \
        device const    int*    current_token_write_positions[[buffer(3)]],                                                     \
        device const    uint*   query_to_seq_map_data       [[buffer(4)]],                                                      \
        constant const  FillKVPagesParams& params           [[buffer(5)]],                                                      \
        device          TYPE*   global_k_pool_out           [[buffer(6)]],                                                      \
        device          TYPE*   global_v_pool_out           [[buffer(7)]],                                                      \
        uint3           tg_id                              [[threadgroup_position_in_grid]],                                    \
        uint3           thread_idx_in_group_3d             [[thread_position_in_threadgroup]],                                  \
        uint3           threads_per_tg_3d                  [[threads_per_threadgroup]]                                          \
    );

// --- Create the concrete specializations ---
INSTANTIATE_FILL_KV_PAGES(half,        float16);
INSTANTIATE_FILL_KV_PAGES(bfloat16_t,  bfloat16);

// --- Instantiation Macro for paged_attn_fused_kernel ---
#define INSTANTIATE_PAGED_ATTN_FUSED(TYPE, SUFFIX)                                                                              \
    template [[host_name("paged_attn_fused_kernel_" #SUFFIX)]] [[kernel]] void                                                  \
    paged_attn_fused_kernel<TYPE>(                                                                                              \
        device const TYPE* queries_in                [[buffer(0)]],                                                             \
        device const TYPE* k_cache_pool_in           [[buffer(1)]],                                                             \
        device const TYPE* v_cache_pool_in           [[buffer(2)]],                                                             \
        device const uint* page_table_in             [[buffer(3)]],                                                             \
        device const int*  sequence_lengths_in       [[buffer(4)]],                                                             \
        device const int*  query_to_seq_map_in       [[buffer(5)]],                                                             \
        device const int*  query_token_offset_in     [[buffer(6)]],                                                             \
        constant const PagedAttentionParams& params  [[buffer(7)]],                                                             \
        device TYPE*       output_buffer             [[buffer(8)]],                                                             \
        uint              actual_simd_width          [[threads_per_simdgroup]],                                                 \
        threadgroup float* tg_mem                    [[threadgroup(0)]],                                                        \
        uint3             tg_pos_in_grid             [[threadgroup_position_in_grid]],                                          \
        uint3             tg_dim                     [[threads_per_threadgroup]],                                               \
        uint              local_idx_in_tg            [[thread_index_in_threadgroup]],                                           \
        uint              simd_lane_id               [[thread_index_in_simdgroup]],                                             \
        uint              simd_group_id              [[simdgroup_index_in_threadgroup]]                                         \
    );

// --- Create the concrete specializations ---
INSTANTIATE_PAGED_ATTN_FUSED(half,        float16);
INSTANTIATE_PAGED_ATTN_FUSED(bfloat16_t,  bfloat16);
