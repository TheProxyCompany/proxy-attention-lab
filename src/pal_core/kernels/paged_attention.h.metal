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

// --- Kernel Configuration Constants ---
constant static const uint kMaxHeadDimMetal = 256;

constant static const uint kSimdLanesPerGroup = 32;

constant static const uint kAlignmentBytes = 64;
constant static const uint kAlignmentMask = kAlignmentBytes - 1;

constant static const float kEpsilonForZeroGuard = 1e-9f;
constant static const float kSmallDenominatorThreshold = 1e-6f;


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
 * @param tg_mem Threadgroup memory for scratch space
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
    device      const int*  sequence_lengths_in,
    device      const int*  query_to_seq_map_in,
    device      const int*  query_token_offset_in,
    constant    const PagedAttentionParams& params,
    device      half* output_buffer,
    threadgroup float* tg_mem,
    uint3       tg_pos_in_grid,
    uint3       tg_dim,
    uint        local_idx_in_tg,
    uint        simd_lane_id,
    uint        simd_group_id
);


/**
 * Maps a query head index to its corresponding key-value head index.
 * Handles all attention types: MHA (num_q_heads == num_kv_heads),
 * GQA (num_q_heads > num_kv_heads), and MQA (num_kv_heads = 1).
 *
 * @param global_item_q_head_idx Effective Q head index for the item
 * @param num_q_heads_param Number of query heads
 * @param num_kv_heads_param Number of key-value heads
 * @return The mapped KV head index
 */
static inline uint map_q_to_kv_head(
    uint global_item_q_head_idx,
    uint num_q_heads_param,
    uint num_kv_heads_param
) {
    if (num_kv_heads_param == 0) return 0; // Safety for no KV heads
    uint target_kv_head_idx = 0;
    if (num_q_heads_param > num_kv_heads_param) { // GQA
        target_kv_head_idx = global_item_q_head_idx / (num_q_heads_param / num_kv_heads_param);
    } else { // MHA or MQA (num_q_heads <= num_kv_heads)
        target_kv_head_idx = global_item_q_head_idx; // For MHA, direct map. For MQA (q_heads=1), this is 0.
    }
    return target_kv_head_idx; // Removed redundant modulo
}

/**
 * Fetches a pointer to a key or value vector from the paged cache.
 * Handles page table lookup, validity checks, and final pointer calculation.
 *
 * @param is_k_vector True for K pointer, false for V pointer
 * @param actual_hist_token_pos History token position to fetch
 * @param target_kv_head_idx Already mapped KV head index
 * @param k_cache_pool_in_param Key cache base pointer
 * @param v_cache_pool_in_param Value-cache base pointer
 * @param page_table_slice Prefetched page-table slice for current sequence
 * @param kernel_params Kernel parameters struct
 * @return Pointer to the K/V vector, or nullptr if invalid
 */
static inline device const half* fetch_kv_pointer(
    bool is_k_vector, // true for K, false for V
    uint actual_hist_token_pos,
    uint target_kv_head_idx, // Already mapped
    device const half* k_cache_pool_in_param,
    device const half* v_cache_pool_in_param,
    threadgroup const uint* page_table_slice,
    constant const PagedAttentionParams& kernel_params
) {
    if ((is_k_vector && k_cache_pool_in_param == nullptr) ||
        (!is_k_vector && v_cache_pool_in_param == nullptr) ||
        page_table_slice == nullptr) {
        return nullptr;
    }

    // Calculate indices for page table lookup
    uint logical_block_idx = actual_hist_token_pos / kernel_params.tokens_per_page;
    if (logical_block_idx >= kernel_params.max_logical_blocks_per_seq) {
        return nullptr;  // Invalid block index
    }

    uint tokens_per_page = kernel_params.tokens_per_page;
    uint token_slot_in_page = actual_hist_token_pos % tokens_per_page;
    uint physical_page_id = page_table_slice[logical_block_idx];
    if (physical_page_id >= kernel_params.num_physical_pages_in_pool) {
        return nullptr;  // Invalid page
    }

    // Calculate the offset with careful type casting and multiplication order
    uint head_dim = kernel_params.head_dim;
    uint num_kv_heads = kernel_params.num_kv_heads;

    // Use 32-bit math where possible
    uint per_token_stride = num_kv_heads * head_dim;
    uint per_page_stride = tokens_per_page * per_token_stride;

    ulong total_offset = (ulong)physical_page_id * (ulong)per_page_stride +
                         (ulong)token_slot_in_page * (ulong)per_token_stride +
                         (ulong)target_kv_head_idx * (ulong)head_dim;

    // Return the appropriate pointer
    return is_k_vector ? (k_cache_pool_in_param + total_offset) : (v_cache_pool_in_param + total_offset);
}

/**
 * Updates the shared softmax statistics using Kahan summation for numerical stability.
 * This helper encapsulates the update of shared global statistics (m_global, s_global) and
 * Kahan compensation term, and broadcasts the scale factor for rescaling previous accumulations.
 *
 * @param current_global_stats_ptr Pointer to shared {m_global, s_global} stats
 * @param current_s_comp_ptr Pointer to shared Kahan compensation term
 * @param m_local_tile_from_reduction Maximum score from current tile reduction
 * @param d_local_tile_from_reduction Sum of fast::exp(score - m_local) from current tile
 * @param broadcast_scale_scratch_ptr Pointer to scratch space for broadcasting scale factor
 * @param kernel_params Kernel parameters struct with log_exp_min_clamp
 */
static inline void update_softmax_stats_kahan(
    threadgroup float2* current_global_stats_ptr,
    threadgroup float* current_s_comp_ptr,
    float m_local_tile_from_reduction,
    float d_local_tile_from_reduction,
    threadgroup float* broadcast_scale_scratch_ptr,
    constant const PagedAttentionParams& params
) {
    // This helper is CALLED ONLY BY local_thread_idx == 0
    float m_prev = (*current_global_stats_ptr).x;
    float s_prev = (*current_global_stats_ptr).y;
    float c_s_prev = *current_s_comp_ptr;

    float m_new = m_prev;
    float s_new_uncompensated = s_prev; // s_global before Kahan for current addition
    float c_s_new = c_s_prev;
    float scale_f = 1.0f;

    if (m_local_tile_from_reduction > m_prev) {
        m_new = m_local_tile_from_reduction;
        scale_f = fast::exp(max(m_prev - m_new, params.log_exp_min_clamp));
        s_new_uncompensated = s_prev * scale_f; // Rescale s
        c_s_new = c_s_prev * scale_f;         // Rescale its compensation term
    }

    // For improved precision when d_local_tile_from_reduction is very large or small,
    // use precise::exp for the second exponentiation in critical cases
    float exp_arg = max(m_local_tile_from_reduction - m_new, params.log_exp_min_clamp);
    float term_to_add;

    // Use precise::exp when the argument is large in magnitude or d_local is extreme
    if (abs(exp_arg) > 10.0f || d_local_tile_from_reduction > 1e6f || d_local_tile_from_reduction < 1e-6f) {
        term_to_add = d_local_tile_from_reduction * precise::exp(exp_arg);
    } else {
        term_to_add = d_local_tile_from_reduction * fast::exp(exp_arg);
    }

    float y_kahan = term_to_add - c_s_new; // c_s_new is compensation from *previous* Kahan steps on s_new_uncompensated
    float t_kahan = s_new_uncompensated + y_kahan;
    c_s_new = (t_kahan - s_new_uncompensated) - y_kahan; // New compensation for *next* addition
    float s_new_final = t_kahan;                         // Final s_global after this tile

    *current_global_stats_ptr = float2(m_new, s_new_final);
    *current_s_comp_ptr = c_s_new;
    broadcast_scale_scratch_ptr[0] = scale_f;
}

/**
 * Zeros out the output vector for a given item when an early exit condition is met.
 * This helper centralizes the logic for output zeroing in early exit conditions.
 *
 * Note: This function is now deprecated in favor of direct zeroing in the kernel
 * which handles all query heads for a token. This is kept for backwards compatibility.
 *
 * @param item_global_idx The global item index for which to zero the output
 * @param output_buffer_param Output buffer to write zeros to
 * @param kernel_params Kernel parameters struct with head_dim
 */
static inline void zero_output_vector_for_item(
    uint item_global_idx,
    device half* output_buffer_param,
    constant const PagedAttentionParams& kernel_params
) {
    // This helper should only be CALLED by local_thread_idx == 0
    for (uint i = 0; i < kernel_params.head_dim; ++i) {
        output_buffer_param[item_global_idx * kernel_params.head_dim + i] = 0.0h;
    }
}


/**
 * Computes the dot product between a query vector and a key vector in threadgroup memory.
 * Efficient vectorized implementation that assumes head_dim is a multiple of 4.
 * This version accepts K-vector in half precision and converts to float on-the-fly.
 *
 * @param q_vec_shmem_param Pointer to query vector in shared memory (float)
 * @param k_vec_tile_entry_param Pointer to key vector in threadgroup memory (K_tile, half precision)
 * @param kernel_params Kernel parameters struct with head_dim
 * @return The dot product result as a float
 */
static inline float dot_product_qk(
    threadgroup const float* q_vec_shmem_param,
    threadgroup const half* k_vec_tile_entry_param,
    constant const PagedAttentionParams& kernel_params
) {
    float score = 0.0f;
    // The helper assumes it's always called for a full head_dim that's a multiple of 4.
    for(uint d = 0; d < kernel_params.head_dim; d += 4) {
        float4 qv = *((threadgroup const float4*)(q_vec_shmem_param + d));
        // Convert from half4 to float4 on-the-fly
        float4 kv = float4(*((threadgroup const half4*)(k_vec_tile_entry_param + d)));
        score += dot(qv, kv);
    }
    return score;
}
