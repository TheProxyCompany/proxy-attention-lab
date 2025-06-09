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
#include "paged_attention.h.metal"
#include "paged_reduce.h.metal"

// --- Instantiation Macro for fill_kv_pages ---
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

// --- Instantiation Macro for pal_paged_attention ---
#define INSTANTIATE_PAL_PAGED_ATTENTION(TYPE, HEAD_DIM, CHUNK_SIZE, SUFFIX)                                                     \
    template [[host_name("pal_paged_attention_" #SUFFIX "_" #HEAD_DIM)]] [[kernel]] void                                        \
    pal_paged_attention<TYPE, HEAD_DIM, CHUNK_SIZE>(                                                                            \
        device const TYPE*  queries_in               [[buffer(0)]],                                                             \
        device const TYPE*  k_cache_pool_in          [[buffer(1)]],                                                             \
        device const TYPE*  v_cache_pool_in          [[buffer(2)]],                                                             \
        device const uint*  page_table_in            [[buffer(3)]],                                                             \
        device const int*   context_lens_in          [[buffer(4)]],                                                             \
        device TYPE*        output_buffer            [[buffer(5)]],                                                             \
        device float*       max_logits_out           [[buffer(6), function_constant(USE_TWO_PASS)]],                            \
        device float*       exp_sums_out             [[buffer(7), function_constant(USE_TWO_PASS)]],                            \
        device TYPE*        tmp_out                  [[buffer(8), function_constant(USE_TWO_PASS)]],                            \
        constant const PagedAttentionParams& params  [[buffer(9)]],                                                             \
        threadgroup uchar*  tg_mem                   [[threadgroup(0)]],                                                        \
        uint3               tg_pos_in_grid           [[threadgroup_position_in_grid]],                                          \
        uint                local_idx_in_tg          [[thread_index_in_threadgroup]]                                            \
    );

// --- Instantiation Macro for pal_paged_reduce ---
#define INSTANTIATE_PAL_PAGED_REDUCE(TYPE, HEAD_DIM, CHUNK_SIZE, SUFFIX)                                                        \
    template [[host_name("pal_paged_reduce_" #SUFFIX "_" #HEAD_DIM)]] [[kernel]] void                                           \
    pal_paged_reduce<TYPE, HEAD_DIM, CHUNK_SIZE>(                                                                               \
        device TYPE*        output_buffer            [[buffer(0)]],                                                             \
        device const float* max_logits_in            [[buffer(1)]],                                                             \
        device const float* exp_sums_in              [[buffer(2)]],                                                             \
        device const TYPE*  tmp_in                   [[buffer(3)]],                                                             \
        device const int*   context_lens_in          [[buffer(4)]],                                                             \
        constant const PagedAttentionParams& params  [[buffer(5)]],                                                             \
        threadgroup uchar*  tg_mem                   [[threadgroup(0)]],                                                        \
        uint3               tg_pos_in_grid           [[threadgroup_position_in_grid]],                                          \
        uint                local_idx_in_tg          [[thread_index_in_threadgroup]],                                           \
        uint                simdgroup_idx            [[simdgroup_index_in_threadgroup]],                                         \
        uint                lane_idx                 [[thread_index_in_simdgroup]]                                              \
    );

// --- Create concrete specializations for common head dimensions ---

// Head dimension 32
INSTANTIATE_PAL_PAGED_ATTENTION(half,        32, 512, float16)
INSTANTIATE_PAL_PAGED_ATTENTION(bfloat16_t,  32, 512, bfloat16)
INSTANTIATE_PAL_PAGED_REDUCE(half,           32, 512, float16)
INSTANTIATE_PAL_PAGED_REDUCE(bfloat16_t,     32, 512, bfloat16)

// Head dimension 64
INSTANTIATE_PAL_PAGED_ATTENTION(half,        64, 512, float16)
INSTANTIATE_PAL_PAGED_ATTENTION(bfloat16_t,  64, 512, bfloat16)
INSTANTIATE_PAL_PAGED_REDUCE(half,           64, 512, float16)
INSTANTIATE_PAL_PAGED_REDUCE(bfloat16_t,     64, 512, bfloat16)

// Head dimension 80
INSTANTIATE_PAL_PAGED_ATTENTION(half,        80, 512, float16)
INSTANTIATE_PAL_PAGED_ATTENTION(bfloat16_t,  80, 512, bfloat16)
INSTANTIATE_PAL_PAGED_REDUCE(half,           80, 512, float16)
INSTANTIATE_PAL_PAGED_REDUCE(bfloat16_t,     80, 512, bfloat16)

// Head dimension 96
INSTANTIATE_PAL_PAGED_ATTENTION(half,        96, 512, float16)
INSTANTIATE_PAL_PAGED_ATTENTION(bfloat16_t,  96, 512, bfloat16)
INSTANTIATE_PAL_PAGED_REDUCE(half,           96, 512, float16)
INSTANTIATE_PAL_PAGED_REDUCE(bfloat16_t,     96, 512, bfloat16)

// Head dimension 128
INSTANTIATE_PAL_PAGED_ATTENTION(half,        128, 512, float16)
INSTANTIATE_PAL_PAGED_ATTENTION(bfloat16_t,  128, 512, bfloat16)
INSTANTIATE_PAL_PAGED_REDUCE(half,           128, 512, float16)
INSTANTIATE_PAL_PAGED_REDUCE(bfloat16_t,     128, 512, bfloat16)

// Head dimension 256
INSTANTIATE_PAL_PAGED_ATTENTION(half,        256, 512, float16)
INSTANTIATE_PAL_PAGED_ATTENTION(bfloat16_t,  256, 512, bfloat16)
INSTANTIATE_PAL_PAGED_REDUCE(half,           256, 512, float16)
INSTANTIATE_PAL_PAGED_REDUCE(bfloat16_t,     256, 512, bfloat16)
