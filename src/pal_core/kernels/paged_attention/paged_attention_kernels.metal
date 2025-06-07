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
// We will add includes for the other templated kernels here as we create them.
// #include "paged_attention_fused.h.metal"
// #include "paged_attention_2pass.h.metal"

[[kernel]] void get_device_info() {
    // used for fetching a metal compute pipeline state
    // for the current device to get the max threads per group
    // and simd group size
}

// --- Instantiation Macro for fill_kv_pages ---
// This macro creates a specialized version of the kernel with a unique host name
// that the C++ code can look up at runtime.
#define INSTANTIATE_FILL_KV_PAGES(TYPE, SUFFIX) \
    template [[host_name("fill_kv_pages_kernel_" #SUFFIX)]] [[kernel]] void \
    fill_kv_pages_kernel<TYPE>( \
        device const    TYPE*       new_keys [[buffer(0)]], \
        device const    TYPE*       new_values [[buffer(1)]], \
        device const    uint*       page_table_data [[buffer(2)]], \
        device const    int*        current_token_write_positions [[buffer(3)]], \
        device const    uint*       query_to_seq_map_data [[buffer(4)]], \
        constant const  FillKVPagesParams& params [[buffer(5)]], \
        device          TYPE*       global_k_pool_out [[buffer(6)]], \
        device          TYPE*       global_v_pool_out [[buffer(7)]], \
        uint3           tg_id [[threadgroup_position_in_grid]], \
        uint3           thread_idx_in_group_3d [[thread_position_in_threadgroup]], \
        uint3           threads_per_tg_3d [[threads_per_threadgroup]]);

// --- Create the concrete specializations ---
INSTANTIATE_FILL_KV_PAGES(half, float16);
INSTANTIATE_FILL_KV_PAGES(bfloat16_t, bfloat16);
