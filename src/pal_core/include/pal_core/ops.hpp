#pragma once
// ops.hpp
// Declarations of PAL core operations for MLX integration.
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

#include <mlx/array.h>
#include <mlx/utils.h>

namespace mx = mlx::core;

namespace pal::cpp {

/**
 * @brief Performs paged attention operation using cached key-value pairs.
 *
 * This function implements paged attention, allowing transformer models to
 * efficiently access key-value pairs stored in a memory pool organized as pages.
 * It supports Multi-head Attention (MHA), Grouped Query Attention (GQA), and
 * Multi-query Attention (MQA) patterns.
 *
 * @param queries Query vectors to compute attention against cached keys.
 *                Shape can be 1D, 2D [tokens, head_dim], or 3D [tokens, heads, head_dim]
 * @param k_cache_pool Global key cache pool with shape
 *                    [num_pages, tokens_per_page, kv_heads, head_dim]
 * @param v_cache_pool Global value cache pool with shape
 *                    [num_pages, tokens_per_page, kv_heads, head_dim]
 * @param page_table Page table mapping logical blocks to physical page IDs.
 *                  Shape [num_sequences, max_blocks_per_seq]
 * @param sequence_lengths Actual length of each sequence in the batch
 * @param query_to_seq_map Maps each query token to its sequence index in the batch
 * @param query_token_offset Logical position of each query token within its sequence
 * @param use_fused_kernel Whether to use the fused kernel
 * @param stream MLX stream or device for the operation
 * @return mx::array Output of shape [num_queries, head_dim] containing the attention results
 */
mx::array paged_attention(
    const mx::array& queries,
    const mx::array& k_cache_pool,
    const mx::array& v_cache_pool,
    const mx::array& page_table,
    const mx::array& sequence_lengths,
    const mx::array& query_to_seq_map,
    const mx::array& query_token_offset,
    bool use_fused_kernel,
    mx::StreamOrDevice stream = {}
);

}  // namespace pal::cpp
