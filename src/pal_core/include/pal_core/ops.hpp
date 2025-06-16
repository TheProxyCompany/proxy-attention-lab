#pragma once
// ops.hpp
// Declarations of PAL core operations for MLX integration.
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

#include <mlx/array.h>
#include <mlx/utils.h>

namespace mx = mlx::core;

namespace pal::cpp {

/**
 * @brief Performs paged attention decode operation using cached key-value pairs.
 *
 * This function implements paged attention decode, allowing transformer models to
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
 * @param stream MLX stream or device for the operation
 * @return mx::array Output of shape [num_queries, head_dim] containing the attention results
 */
mx::array paged_attention_decode(
    const mx::array& queries,
    const mx::array& k_cache_pool,
    const mx::array& v_cache_pool,
    const mx::array& page_table,
    const mx::array& sequence_lengths,
    mx::StreamOrDevice stream = {}
);

/**
 * @brief Performs paged attention prefill operation using cached key-value pairs.
 *
 * This function implements paged attention prefill, allowing transformer models to
 * efficiently access key-value pairs stored in a memory pool organized as pages.
 * It supports Multi-head Attention (MHA), Grouped Query Attention (GQA), and
 * Multi-query Attention (MQA) patterns.
 *
 * @param q_prompt Prompt query vectors to compute attention against cached keys.
 *                Shape can be 1D, 2D [tokens, head_dim], or 3D [tokens, heads, head_dim]
 * @param k_prompt Prompt key vectors to compute attention against cached keys.
 *                Shape can be 1D, 2D [tokens, head_dim], or 3D [tokens, heads, head_dim]
 * @param v_prompt Prompt value vectors to compute attention against cached keys.
 *                Shape can be 1D, 2D [tokens, head_dim], or 3D [tokens, heads, head_dim]
 * @param k_cache_paged Global key cache pool with shape
 *                    [num_pages, tokens_per_page, kv_heads, head_dim]
 * @param v_cache_paged Global value cache pool with shape
 *                    [num_pages, tokens_per_page, kv_heads, head_dim]
 * @param page_table Page table mapping logical blocks to physical page IDs.
 *                  Shape [num_sequences, max_blocks_per_seq]
 * @param context_len_arr Actual length of each sequence in the batch
 * @param stream Optional stream or device for the operation.
 * @return mx::array Output of shape [num_queries, head_dim] containing the attention results
 */
mx::array paged_attention_prefill(
    const mx::array& q_prompt,
    const mx::array& k_prompt,
    const mx::array& v_prompt,
    const mx::array& k_cache_paged,
    const mx::array& v_cache_paged,
    const mx::array& page_table,
    const mx::array& context_len_arr,
    mx::StreamOrDevice stream = {}
);

/**
 * @brief Fills the KV cache pages with the new keys and values.
 *
 * This function fills the KV cache pages with the new keys and values.
 *
 * @param new_keys The new keys to fill the KV cache with.
 * @param new_values The new values to fill the KV cache with.
 * @param global_key_pool The global key pool to fill the KV cache with.
 * @param global_value_pool The global value pool to fill the KV cache with.
 * @param page_table Page table mapping logical blocks for each sequence
 *                  to physical page IDs in the k_cache_pool/v_cache_pool.
 *                  Shape: [NumSequencesInBatch, MaxLogicalBlocksPerSequence]
 * @param current_token_write_positions Logical token index within its sequence where the new K/V should be written
 *                              Shape: [TotalCurrentTokensInBatch]
 * @param query_to_seq_map Maps each query token to its sequence index.
 * @param stream Optional stream or device for the operation.
 * @return Tuple containing the updated global key pool and global value pool.
 */
std::tuple<mx::array, mx::array> fill_kv_pages(
    const mx::array& new_keys,
    const mx::array& new_values,
    const mx::array& global_key_pool,
    const mx::array& global_value_pool,
    const mx::array& page_table,
    const mx::array& current_token_write_positions,
    const mx::array& query_to_seq_map,
    mx::StreamOrDevice stream = {}
);

}  // namespace pal::cpp
