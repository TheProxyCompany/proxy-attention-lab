// paged_attention_primitive.cpp
// Implementation of the PagedAttentionPrimitive class that provides GPU-accelerated
// paged attention operations for transformer models.
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

#include "pal_core/paged_attention_primitive.hpp"

#include <cmath>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <limits>
#include <unordered_set>
#include <algorithm>
#include <spdlog/spdlog.h>

// Define half type for memory calculations (matching Metal's half type)
using half = short;

// MLX and Metal includes
#include <mlx/allocator.h>
#include <mlx/array.h>
#include <mlx/backend/cpu/encoder.h>
#include <mlx/backend/metal/device.h>
#include <mlx/backend/metal/metal.h>
#include <mlx/utils.h>

#include "shaders/paged_attention_types.h"

namespace mx = mlx::core;

namespace pal::cpp {

constexpr static float kLogFp16DenormMinVal = -88.0f;

constexpr static uint32_t PREFILL_TOKENS_PER_TG_KNOB = 1;
constexpr static uint32_t PREFILL_HEADS_PER_TG_KNOB = 32; // Process all heads for one token
constexpr static uint32_t DEFAULT_THREADS_PER_GROUP = 64; // 64 is the default

// New Pass 1 parameters for page-centric prefill architecture
constexpr static uint32_t PREFILL_PASS1_Q_HEAD_BLOCK_SIZE = 8;

struct CoreDims {
    uint32_t head_dim{0};
    uint32_t num_q_heads{0};
    uint32_t tokens_per_page{0};
    uint32_t num_kv_heads{0};
    size_t num_items_to_process{0};
    size_t query_token_count{0};
};

struct ThreadgroupMemoryLayout {
    size_t q_shmem_bytes{0};
    size_t partial_reduce_scratch_bytes{0};
    size_t simd_reduced_maxes_bytes{0};
    size_t simd_reduced_adjusted_sum_exps_bytes{0};
    size_t global_stats_bytes{0};
    size_t s_global_compensation_bytes{0};
    size_t simd_v_chunk_sums_bytes{0};
    size_t k_tile_bytes{0}; // For caching K-vectors in threadgroup memory
    size_t v_tile_bytes{0}; // For caching V-vectors in threadgroup memory
    size_t page_table_slice_bytes{0};
    size_t final_guard_bytes{0};
    size_t total_bytes{0};
};

// Struct to represent a single query token's relevance to a specific page
struct QueryTokenPageRelevance {
    uint32_t query_token_global_idx;      // Index in the flat queries array
    uint32_t history_start_offset_on_page; // Where history starts on this page
    uint32_t num_history_tokens_on_page;   // Number of history tokens on this page
};

// Struct to hold the complete relevant query mapping for all active pages
struct RelevantQueryMap {
    // Primary data: flat array of query-page relevance entries
    std::vector<QueryTokenPageRelevance> relevance_entries;

    // Secondary data: for each active page, where its entries start/end in relevance_entries
    std::vector<uint32_t> page_start_offsets; // Size: num_active_pages + 1

    // List of active page IDs in the order they'll be processed
    std::vector<uint32_t> active_page_ids;

    // Helper method to get relevance entries for a specific page index
    std::pair<uint32_t, uint32_t> get_page_entry_range(size_t page_index) const {
        if (page_index >= active_page_ids.size()) {
            return {0, 0};
        }
        return {page_start_offsets[page_index], page_start_offsets[page_index + 1]};
    }
};


static ThreadgroupMemoryLayout calculate_threadgroup_memory_breakdown_and_total(
    const PagedAttentionParams& params,
    size_t threads_per_group, // This is the target_threads_per_group for sizing fixed components
    size_t actual_simd_lanes_per_group // Added parameter for actual SIMD width
) {
    ThreadgroupMemoryLayout layout;
    uintptr_t tg_mem_current_offset_bytes = 0;

    // Calculate the number of SIMD groups based on the actual threads_per_group being dispatched
    // constexpr size_t kSimdLanesPerGroup = 32; // Should ideally match queried threadExecutionWidth
    const uint32_t num_simd_groups = (threads_per_group + actual_simd_lanes_per_group - 1) / actual_simd_lanes_per_group;

    // 1. q_shmem: head_dim floats
    layout.q_shmem_bytes = params.head_dim * sizeof(float);
    tg_mem_current_offset_bytes += layout.q_shmem_bytes;
    // Align subsequent sections to 64-byte boundaries
    constexpr size_t kAlignmentBytes = 64;
    constexpr size_t kAlignmentMask = kAlignmentBytes - 1;
    tg_mem_current_offset_bytes = (tg_mem_current_offset_bytes + kAlignmentMask) & ~kAlignmentMask;

    // 2. G_partial_max_scores: threads_per_group floats
    layout.partial_reduce_scratch_bytes = threads_per_group * sizeof(float);
    tg_mem_current_offset_bytes += layout.partial_reduce_scratch_bytes;

    // 3. G_simd_reduced_maxes: num_simd_groups floats
    tg_mem_current_offset_bytes = (tg_mem_current_offset_bytes + kAlignmentMask) & ~kAlignmentMask;
    layout.simd_reduced_maxes_bytes = num_simd_groups * sizeof(float);
    tg_mem_current_offset_bytes += layout.simd_reduced_maxes_bytes;

    // 4. G_simd_reduced_adjusted_sum_exps: num_simd_groups floats
    tg_mem_current_offset_bytes = (tg_mem_current_offset_bytes + kAlignmentMask) & ~kAlignmentMask;
    layout.simd_reduced_adjusted_sum_exps_bytes = num_simd_groups * sizeof(float);
    tg_mem_current_offset_bytes += layout.simd_reduced_adjusted_sum_exps_bytes;

    // 5. g_global_stats (float2 for m_global, s_global)
    tg_mem_current_offset_bytes = (tg_mem_current_offset_bytes + kAlignmentMask) & ~kAlignmentMask;
    layout.global_stats_bytes = 2 * sizeof(float);
    tg_mem_current_offset_bytes += layout.global_stats_bytes;

    // 6. g_s_global_compensation (for Kahan summation)
    tg_mem_current_offset_bytes = (tg_mem_current_offset_bytes + kAlignmentMask) & ~kAlignmentMask;
    layout.s_global_compensation_bytes = 1 * sizeof(float);
    tg_mem_current_offset_bytes += layout.s_global_compensation_bytes;

    // 7. G_simd_group_v_sums (float4 per SIMD group)
    tg_mem_current_offset_bytes = (tg_mem_current_offset_bytes + kAlignmentMask) & ~kAlignmentMask;
    layout.simd_v_chunk_sums_bytes = num_simd_groups * sizeof(float) * 4;
    tg_mem_current_offset_bytes += layout.simd_v_chunk_sums_bytes;

    // 8. K Tile memory - for caching K vectors with padding for bank conflict avoidance
    tg_mem_current_offset_bytes = (tg_mem_current_offset_bytes + kAlignmentMask) & ~kAlignmentMask;

    // K_tile requires tile_size_T_runtime * padded_head_dim_for_tile * sizeof(half) bytes
    layout.k_tile_bytes = params.tile_size_T_runtime * params.head_dim * sizeof(half);
    tg_mem_current_offset_bytes += layout.k_tile_bytes;

    // 9. V Tile memory - for caching V vectors with padding for bank conflict avoidance
    tg_mem_current_offset_bytes = (tg_mem_current_offset_bytes + kAlignmentMask) & ~kAlignmentMask;

    // V_tile requires tile_size_T_runtime * padded_head_dim_for_tile * sizeof(half) bytes
    layout.v_tile_bytes = params.tile_size_T_runtime * params.head_dim * sizeof(half);
    tg_mem_current_offset_bytes += layout.v_tile_bytes;

    // 10. Per-sequence page-table slice
    tg_mem_current_offset_bytes = (tg_mem_current_offset_bytes + kAlignmentMask) & ~kAlignmentMask;
    layout.page_table_slice_bytes = params.max_logical_blocks_per_seq * sizeof(uint32_t);
    tg_mem_current_offset_bytes += layout.page_table_slice_bytes;

    // Final padding guard
    constexpr size_t kFinalTgMemoryPaddingGuardBytes = 32;
    layout.final_guard_bytes = kFinalTgMemoryPaddingGuardBytes;
    tg_mem_current_offset_bytes += layout.final_guard_bytes;

    // Ensure final size is aligned to 64-byte boundary
    layout.total_bytes = (tg_mem_current_offset_bytes + kAlignmentMask) & ~kAlignmentMask;
    spdlog::debug("[PAL TGMemCalc] Total calculated tg_memory_bytes: {} bytes", layout.total_bytes);

    return layout;
}

static ThreadgroupMemoryLayout calculate_final_params_and_memory_layout(
    PagedAttentionParams& params,
    const CoreDims& extracted_core_dims,
    const mx::array& k_pool_arr,
    const mx::array& page_table_arr,
    MTL::Device* mtl_device_ptr,
    size_t target_threads_per_group,
    size_t actual_simd_lanes_per_group, // Added parameter for actual SIMD width
    bool is_prefill
) {
    params.num_q_heads              = extracted_core_dims.num_q_heads;
    params.num_kv_heads             = extracted_core_dims.num_kv_heads;
    params.head_dim                 = extracted_core_dims.head_dim;
    params.tokens_per_page          = extracted_core_dims.tokens_per_page;

    // Kernel constants
    params.log_exp_min_clamp        = kLogFp16DenormMinVal;

    // Derived from input array shapes
    params.max_logical_blocks_per_seq   = page_table_arr.shape(1);
    params.num_sequences_in_batch       = page_table_arr.shape(0);
    params.num_physical_pages_in_pool   = k_pool_arr.shape(0);

    // Calculate inv_sqrt_head_dim
    if (params.head_dim > 0) {
        params.inv_sqrt_head_dim = 1.0f / std::sqrt(static_cast<float>(params.head_dim));
    } else {
        params.inv_sqrt_head_dim = 1.0f;  // Default fallback
    }

    // Calculate the memory layout with a zero tile size
    // So we know how much memory we have to tile with
    constexpr size_t kFinalTgMemoryPaddingGuardBytes = 32;
    constexpr size_t kAlignmentBytes = 64;
    constexpr size_t kAlignmentMask = kAlignmentBytes - 1;

    // Calculate the memory layout with a zero tile size
    PagedAttentionParams params_for_fixed_sizing = params;
    params_for_fixed_sizing.tile_size_T_runtime = 0; // zero tile size

    size_t precise_fixed_tg_mem_bytes =
    calculate_threadgroup_memory_breakdown_and_total(
        params_for_fixed_sizing,
        target_threads_per_group, // Use target_threads_per_group here
        actual_simd_lanes_per_group // Pass actual_simd_lanes_per_group
    ).total_bytes;

    // 1. Gather constants once
    const size_t tg_limit = mtl_device_ptr->maxThreadgroupMemoryLength();
    const size_t guard = kFinalTgMemoryPaddingGuardBytes;
    const uint32_t practical_max_T = 256;
    const uint32_t min_T_soft = 8;
    const uint32_t align_val = 4;

    // 2. Compute the fixed part
    size_t fixed_mem = precise_fixed_tg_mem_bytes;
    if (fixed_mem + guard >= tg_limit) {
        throw std::runtime_error("[PAL Primitive] Fixed scratch memory alone exceeds threadgroup memory limit.");
    }

    // 3. Compute "bytes per history token" for dynamic tiles (K_tile + V_tile)
    uint32_t padded_head_dim_for_tile_calc = params.head_dim;
    
    // For Pass 1 with spatial KV tiling, we need memory for PAGE_SUB_TILE_TOKEN_COUNT tokens
    // across all unique KV heads required by a Q-head block
    size_t bytes_per_token_in_tile;
    if (is_prefill) {
        // For prefill Pass 1: Calculate unique KV heads for a Q-head block
        // With GQA/MQA, PREFILL_PASS1_Q_HEAD_BLOCK_SIZE Q-heads may map to fewer unique KV heads
        uint32_t q_heads_per_kv_head = params.num_q_heads / params.num_kv_heads;
        uint32_t max_unique_kv_heads_per_block = std::min(
            PREFILL_PASS1_Q_HEAD_BLOCK_SIZE,
            (PREFILL_PASS1_Q_HEAD_BLOCK_SIZE + q_heads_per_kv_head - 1) / q_heads_per_kv_head
        );
        
        // PAGE_SUB_TILE_TOKEN_COUNT is 12 (from the kernel)
        constexpr uint32_t PAGE_SUB_TILE_TOKEN_COUNT = 12;
        
        // Total bytes for spatial KV tiles
        size_t bytes_for_kv_tiles = PAGE_SUB_TILE_TOKEN_COUNT * max_unique_kv_heads_per_block * 
                                   padded_head_dim_for_tile_calc * sizeof(half) * 2; // K and V
        
        // Calculate effective bytes per token to fit the spatial tiling
        bytes_per_token_in_tile = bytes_for_kv_tiles / PAGE_SUB_TILE_TOKEN_COUNT;
    } else {
        // For decode: Original calculation (one K/V pair per token)
        bytes_per_token_in_tile = (padded_head_dim_for_tile_calc * sizeof(half)) /*K_tile_half_padded*/ +
                                 (padded_head_dim_for_tile_calc * sizeof(half)) /*V_tile_half_padded*/;
    }
    // 4. Raw ceiling that actually fits
    uint32_t max_T_fit = 0;
    if (tg_limit > fixed_mem + guard) { // Ensure there's some memory left for dynamic parts
        max_T_fit = static_cast<uint32_t>(
            (tg_limit - fixed_mem - guard) / bytes_per_token_in_tile);
    }

    max_T_fit = std::min({max_T_fit,
                          practical_max_T,
                          static_cast<uint32_t>(params.tokens_per_page)});

    // 5c. Align down to multiple of 'align_val' (e.g., 4)
    max_T_fit = (max_T_fit / align_val) * align_val;
    // If alignment made it zero, but it was possible to fit something, force to smallest alignment.
    if (max_T_fit == 0 && ((tg_limit - fixed_mem - guard) / bytes_per_token_in_tile) > 0) {
        max_T_fit = align_val;
    }

    // 6. heuristic bump (but never past the limit of max_T_fit)
    uint32_t desired_T = (params.head_dim <= 256) ? 64u : min_T_soft;
    desired_T = std::min(desired_T, practical_max_T);

    if (max_T_fit >= desired_T) {
        params.tile_size_T_runtime = desired_T;
    } else {
        params.tile_size_T_runtime = max_T_fit;
    }

    // Ensure tile_size_T_runtime is at least min_T_soft, but only if max_T_fit was also at least min_T_soft.
    // This prevents forcing a tile size that doesn't fit.
    if (max_T_fit >= min_T_soft) {
        params.tile_size_T_runtime = std::max(params.tile_size_T_runtime, min_T_soft);
    }

    if (params.tile_size_T_runtime > 0) {
       params.tile_size_T_runtime = (params.tile_size_T_runtime / align_val) * align_val;
       if (params.tile_size_T_runtime == 0) params.tile_size_T_runtime = align_val; // Ensure not zero if it was viable
    }

    if (params.tile_size_T_runtime == 0) {
        throw std::runtime_error(
            "[PAL Primitive] head_dim/tokens_per_page combination leaves no "
            "scratch space for a single KV token.");
    }

    // Recalculate final total memory with the chosen tile_size_T_runtime
    return calculate_threadgroup_memory_breakdown_and_total(params, target_threads_per_group, actual_simd_lanes_per_group);
}

static CoreDims validate_inputs_and_populate_initial_params(const std::vector<mx::array>& inputs) {
    if (inputs.size() != 7) {
        throw std::invalid_argument(
            "[PAL Primitive Validate] Expected 7 inputs, received " + std::to_string(inputs.size()));
    }
    const auto& q = inputs[0];
    const auto& k_pool = inputs[1];
    const auto& page_table = inputs[3];
    const auto& sequence_lengths = inputs[4];
    const auto& query_to_seq_map = inputs[5];
    const auto& query_token_offset = inputs[6];

    // --- Dtype Checks ---
    if (
        page_table.dtype() != mx::uint32 ||
        sequence_lengths.dtype() != mx::int32 ||
        query_to_seq_map.dtype() != mx::int32 ||
        query_token_offset.dtype() != mx::int32
    ) {
         throw std::invalid_argument("[PAL Primitive Validate] sequence_lengths, query_to_seq_map, and query_token_offset must be int32.");
    }

    CoreDims dims; // Initialize struct to be returned

    // --- K/V Pool Geometry & Initial Params ---
    if (k_pool.ndim() != 4) {
        throw std::invalid_argument("[PAL Primitive Validate] k_pool must be 4D [NumPages, TokensPerPage, NumKVHeads, HeadDim].");
    }
    dims.tokens_per_page = k_pool.shape(1);
    dims.num_kv_heads = k_pool.shape(2);
    dims.head_dim = k_pool.shape(3);

    if (dims.head_dim % 4 != 0) {
        throw std::invalid_argument("[PAL Primitive Validate] head_dim (" + std::to_string(dims.head_dim) +
                                   ") must be a multiple of 4 for vectorized kernel execution.");
    }

    // --- Query Format & num_q_heads ---
    if (q.ndim() == 3) { // [NumTokens, NumQHeads, HeadDim]
        if (q.shape(2) != static_cast<int>(dims.head_dim)) { // Cast dims.head_dim for comparison
            throw std::invalid_argument("[PAL Primitive Validate] 3D Query HeadDim mismatch with K-pool head_dim.");
        }
        dims.num_q_heads = q.shape(1);
        dims.num_items_to_process = q.shape(0) * q.shape(1);
        dims.query_token_count = q.shape(0);
    } else if (q.ndim() == 2) { // [NumItems, HeadDim]
        if (q.shape(1) != static_cast<int>(dims.head_dim)) {
            throw std::invalid_argument("[PAL Primitive Validate] 2D Query HeadDim mismatch with K-pool head_dim.");
        }
        dims.num_q_heads = 1;
        dims.num_items_to_process = q.shape(0);
        dims.query_token_count = q.shape(0);
    } else if (q.ndim() == 1) { // [NumItems], implies head_dim=1
        if (dims.head_dim != 1) {
            throw std::invalid_argument("[PAL Primitive Validate] 1D Query requires K-pool head_dim to be 1.");
        }
        dims.num_q_heads = 1;
        dims.num_items_to_process = q.shape(0);
        dims.query_token_count = q.shape(0);
    } else {
        throw std::invalid_argument("[PAL Primitive Validate] Query 'q' ndim (" + std::to_string(q.ndim()) + ") not supported.");
    }

    if (dims.num_items_to_process == 0 && q.size() > 0) {
         throw std::invalid_argument("[PAL Primitive Validate] num_items_to_process is 0 but query array is not empty.");
    }

    // --- Page Table Validation ---
    if (page_table.ndim() != 2) {
        throw std::invalid_argument("[PAL Primitive Validate] page_table must be 2D [NumBatchSeq, MaxLogBlocksPerSeq].");
    }

    // --- Sideband Array Size Validation ---
    if (query_to_seq_map.size() != dims.query_token_count) {
        throw std::invalid_argument("[PAL Primitive Validate] query_to_seq_map size (" + std::to_string(query_to_seq_map.size()) +
                                   ") must match query_token_count (" + std::to_string(dims.query_token_count) + ").");
    }
    if (query_token_offset.size() != dims.query_token_count) {
        throw std::invalid_argument("[PAL Primitive Validate] query_token_offset size (" + std::to_string(query_token_offset.size()) +
                                   ") must match query_token_count (" + std::to_string(dims.query_token_count) + ").");
    }

    spdlog::debug("[PAL Primitive Validate] Input validation and core dimension extraction complete.");
    return dims;
}

PagedAttentionPrimitive::PagedAttentionPrimitive(
    mx::StreamOrDevice stream_or_device,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int tokens_per_page,
    bool is_prefill
) : mx::UnaryPrimitive(mx::to_stream(stream_or_device)),
      num_q_heads_(num_q_heads),
      num_kv_heads_(num_kv_heads),
      head_dim_(head_dim),
      tokens_per_page_(tokens_per_page),
      is_prefill_(is_prefill) { }

void PagedAttentionPrimitive::eval_cpu(const std::vector<mx::array>& inputs, mx::array& out) {
    std::ostringstream oss;
    this->print(oss);
    spdlog::error("[PAL Primitive] CPU evaluation is not supported for {}.", oss.str());
    throw std::runtime_error(
        "[PagedAttentionPrimitive] CPU evaluation is not supported for " + oss.str());
}

void PagedAttentionPrimitive::eval_gpu(const std::vector<mx::array>& inputs,
                                       mx::array& out) {
  // Prepare Metal kernel and command encoder
  auto& s = stream();
  auto& d = mlx::core::metal::device(s.device);

  const std::string library_name_for_mlx = "pal";
  const std::string kernel_name = is_prefill_ ? "paged_attn_prefill_kernel" : "paged_attn_decode_kernel";
  auto kernel_state_ = d.get_kernel(kernel_name, library_name_for_mlx);
  if (!kernel_state_) {
    throw std::runtime_error("[PAL Primitive] Failed to load kernel: " + kernel_name);
  }

  size_t bytes = out.nbytes();
  out.set_data(mx::allocator::malloc(bytes));
  auto& compute_encoder = d.get_command_encoder(s.index);

  // Create and populate PagedAttentionParams struct
  PagedAttentionParams params_struct;
  CoreDims core_dims = validate_inputs_and_populate_initial_params(inputs);

  // Populate initial parts directly for head_dim check
  params_struct.head_dim = core_dims.head_dim;

  // --- Determine threads per group for kernel execution ---
  auto* device_ptr = d.mtl_device(); // Get device_ptr earlier
  size_t execution_width = kernel_state_->threadExecutionWidth();
  size_t max_threads_device = kernel_state_->maxTotalThreadsPerThreadgroup();

  size_t threads_to_launch = DEFAULT_THREADS_PER_GROUP;
  threads_to_launch = ((threads_to_launch + execution_width - 1) / execution_width) * execution_width; // Align to exec width
  threads_to_launch = std::min(threads_to_launch, max_threads_device); // Cap by device max

  // Call helper to populate the remaining attention parameters and calculate final memory layout
  ThreadgroupMemoryLayout memory_layout = calculate_final_params_and_memory_layout(
      params_struct,
      core_dims,
      inputs[1], // k_pool
      inputs[3], // page_table
      device_ptr,
      threads_to_launch, // Use the same threads_to_launch for memory layout calculation
      execution_width,   // Pass the actual execution_width here
      is_prefill_       // Pass whether this is prefill mode
  );

  spdlog::debug(
      "[PAL Primitive Dispatch] Using fixed threads_to_launch: {}, tile_size_T_runtime: {}, "
      "execution_width: {}, max_threads_device: {}",
      threads_to_launch,
      params_struct.tile_size_T_runtime,
      execution_width,
      execution_width
  );

  // Verify all pointers are valid before passing to Metal
  if (!inputs[0].data<void>() || !inputs[1].data<void>() || !inputs[2].data<void>() ||
      !inputs[3].data<void>() || !inputs[4].data<void>() || !inputs[5].data<void>() ||
      !inputs[6].data<void>()) {
    spdlog::error("[PAL Primitive] One or more input arrays have null data pointers");
    throw std::runtime_error("Null input data pointers detected in paged attention primitive");
  }

  // Determine the dispatch grid based on prefill or decode mode
  size_t dispatch_grid_width;
  size_t dispatch_grid_height = 1;

  // Arrays for Pass 1 relevant query mapping (only used in prefill mode)
  // Initialize with dummy data - will be properly populated in prefill branch
  uint32_t dummy_uint = 0;
  mx::array relevant_query_indices(&dummy_uint, {0}, mx::uint32);
  mx::array relevant_history_starts(&dummy_uint, {0}, mx::uint32);
  mx::array relevant_history_counts(&dummy_uint, {0}, mx::uint32);
  mx::array page_offsets(&dummy_uint, {0}, mx::uint32);
  mx::array active_pages(&dummy_uint, {0}, mx::uint32);

  if (is_prefill_) {
      // Step 2: CPU-side pre-computation for Pass 1
      spdlog::debug("[PAL Primitive Dispatch] PREFILL MODE: Starting Pass 1 pre-computation");

      // Extract input arrays
      const auto& page_table = inputs[3];
      const auto& sequence_lengths = inputs[4];
      const auto& query_to_seq_map = inputs[5];
      const auto& query_token_offset = inputs[6];

      // Get raw pointers for efficient access
      const uint32_t* page_table_ptr = page_table.data<uint32_t>();
      const int32_t* sequence_lengths_ptr = sequence_lengths.data<int32_t>();
      const int32_t* query_to_seq_map_ptr = query_to_seq_map.data<int32_t>();
      const int32_t* query_token_offset_ptr = query_token_offset.data<int32_t>();

      // Step 2.1: Determine active KV pages
      std::unordered_set<uint32_t> active_pages_set;
      RelevantQueryMap query_map;

      // Iterate through all query tokens to find active pages
      for (size_t query_idx = 0; query_idx < core_dims.query_token_count; ++query_idx) {
          int32_t seq_idx = query_to_seq_map_ptr[query_idx];
          if (seq_idx < 0 || seq_idx >= static_cast<int32_t>(params_struct.num_sequences_in_batch)) {
              continue; // Skip invalid sequence indices
          }

          int32_t query_offset = query_token_offset_ptr[query_idx];
          int32_t seq_length = sequence_lengths_ptr[seq_idx];

          // Calculate the range of history tokens for this query
          int32_t history_start = 0;
          int32_t history_end = query_offset; // Exclusive end

          if (history_end <= history_start) {
              continue; // No history for this query token
          }

          // Find which pages contain the history tokens
          int32_t first_history_page = history_start / params_struct.tokens_per_page;
          int32_t last_history_page = (history_end - 1) / params_struct.tokens_per_page;

          // Add all pages in the range to active set
          for (int32_t page_idx = first_history_page; page_idx <= last_history_page; ++page_idx) {
              if (page_idx < static_cast<int32_t>(params_struct.max_logical_blocks_per_seq)) {
                  uint32_t physical_page_id = page_table_ptr[seq_idx * params_struct.max_logical_blocks_per_seq + page_idx];
                  if (physical_page_id < params_struct.num_physical_pages_in_pool) {
                      active_pages_set.insert(physical_page_id);
                  }
              }
          }
      }

      // Convert set to vector for deterministic ordering
      query_map.active_page_ids.assign(active_pages_set.begin(), active_pages_set.end());
      std::sort(query_map.active_page_ids.begin(), query_map.active_page_ids.end());

      spdlog::debug("[PAL Primitive Dispatch] Found {} active KV pages", query_map.active_page_ids.size());

      // Step 2.2: Construct the Relevant Query Map
      query_map.page_start_offsets.resize(query_map.active_page_ids.size() + 1, 0);

      // For each active page, find relevant query tokens
      for (size_t page_idx = 0; page_idx < query_map.active_page_ids.size(); ++page_idx) {
          uint32_t physical_page_id = query_map.active_page_ids[page_idx];

          // Iterate through all query tokens to find which ones have history on this page
          for (size_t query_idx = 0; query_idx < core_dims.query_token_count; ++query_idx) {
              int32_t seq_idx = query_to_seq_map_ptr[query_idx];
              if (seq_idx < 0 || seq_idx >= static_cast<int32_t>(params_struct.num_sequences_in_batch)) {
                  continue;
              }

              int32_t query_offset = query_token_offset_ptr[query_idx];
              int32_t history_start = 0;
              int32_t history_end = query_offset;

              if (history_end <= history_start) {
                  continue;
              }

              // Check each logical page in the sequence
              for (int32_t logical_page_idx = 0; logical_page_idx < static_cast<int32_t>(params_struct.max_logical_blocks_per_seq); ++logical_page_idx) {
                  uint32_t page_physical_id = page_table_ptr[seq_idx * params_struct.max_logical_blocks_per_seq + logical_page_idx];

                  if (page_physical_id == physical_page_id) {
                      // Calculate the token range on this logical page
                      int32_t page_token_start = logical_page_idx * params_struct.tokens_per_page;
                      int32_t page_token_end = page_token_start + params_struct.tokens_per_page;

                      // Find intersection with history range
                      int32_t overlap_start = std::max(history_start, page_token_start);
                      int32_t overlap_end = std::min(history_end, page_token_end);

                      if (overlap_start < overlap_end) {
                          // This query has history on this page
                          QueryTokenPageRelevance relevance;
                          relevance.query_token_global_idx = static_cast<uint32_t>(query_idx);
                          relevance.history_start_offset_on_page = overlap_start - page_token_start;
                          relevance.num_history_tokens_on_page = overlap_end - overlap_start;

                          query_map.relevance_entries.push_back(relevance);
                      }
                  }
              }
          }

          // Update page start offset for next page
          query_map.page_start_offsets[page_idx + 1] = query_map.relevance_entries.size();
      }

      spdlog::debug("[PAL Primitive Dispatch] Built relevant query map with {} total entries",
                    query_map.relevance_entries.size());

      // Step 2.3: Prepare data for kernel - flatten into mx::arrays
      // Create arrays for the flattened data
      std::vector<uint32_t> flat_query_indices;
      std::vector<uint32_t> flat_history_starts;
      std::vector<uint32_t> flat_history_counts;

      for (const auto& entry : query_map.relevance_entries) {
          flat_query_indices.push_back(entry.query_token_global_idx);
          flat_history_starts.push_back(entry.history_start_offset_on_page);
          flat_history_counts.push_back(entry.num_history_tokens_on_page);
      }

      // Create mx::arrays from the vectors
      if (!flat_query_indices.empty()) {
          relevant_query_indices = mx::array(
              flat_query_indices.data(),
              {static_cast<int>(flat_query_indices.size())},
              mx::uint32);
          relevant_history_starts = mx::array(
              flat_history_starts.data(),
              {static_cast<int>(flat_history_starts.size())},
              mx::uint32);
          relevant_history_counts = mx::array(
              flat_history_counts.data(),
              {static_cast<int>(flat_history_counts.size())},
              mx::uint32);
      } else {
          // Create empty arrays if no relevant queries
          uint32_t dummy = 0;
          relevant_query_indices = mx::array(&dummy, {0}, mx::uint32);
          relevant_history_starts = mx::array(&dummy, {0}, mx::uint32);
          relevant_history_counts = mx::array(&dummy, {0}, mx::uint32);
      }

      if (!query_map.page_start_offsets.empty()) {
          page_offsets = mx::array(
              query_map.page_start_offsets.data(),
              {static_cast<int>(query_map.page_start_offsets.size())},
              mx::uint32);
      } else {
          uint32_t dummy = 0;
          page_offsets = mx::array(&dummy, {0}, mx::uint32);
      }

      if (!query_map.active_page_ids.empty()) {
          active_pages = mx::array(
              query_map.active_page_ids.data(),
              {static_cast<int>(query_map.active_page_ids.size())},
              mx::uint32);
      } else {
          uint32_t dummy = 0;
          active_pages = mx::array(&dummy, {0}, mx::uint32);
      }

      // Step 3: Modify dispatch grid calculation for Pass 1
      // Grid dimensions for page-centric Pass 1
      dispatch_grid_width = query_map.active_page_ids.size();  // One TG per active page
      dispatch_grid_height = (core_dims.num_q_heads + PREFILL_PASS1_Q_HEAD_BLOCK_SIZE - 1) / PREFILL_PASS1_Q_HEAD_BLOCK_SIZE;

      spdlog::debug("[PAL Primitive Dispatch] PREFILL MODE: Using Pass 1 page-centric grid dispatch");
      spdlog::debug("[PAL Primitive Dispatch] Grid dimensions: {} x {} (active_pages x q_head_blocks)",
                    dispatch_grid_width, dispatch_grid_height);
      spdlog::debug("[PAL Primitive Dispatch] Active pages: {}, Q-head blocks: {} (block size: {})",
                    dispatch_grid_width, dispatch_grid_height, PREFILL_PASS1_Q_HEAD_BLOCK_SIZE);
      spdlog::debug("[PAL Primitive Dispatch] Total threadgroups: {}",
                    dispatch_grid_width * dispatch_grid_height);
  } else {
      dispatch_grid_width = core_dims.num_items_to_process;
      spdlog::debug("[PAL Primitive Dispatch] DECODE MODE: Using item-based dispatch grid width: {}", dispatch_grid_width);
  }

  spdlog::debug("[PAL Primitive Dispatch] Dispatching kernel with tile_size_T_runtime: {}", params_struct.tile_size_T_runtime);

  compute_encoder.set_compute_pipeline_state(kernel_state_); // Set PSO here

  // Set input arrays for the compute encoder
  compute_encoder.set_input_array(inputs[0], 0); // q
  compute_encoder.set_input_array(inputs[1], 1); // k_pool
  compute_encoder.set_input_array(inputs[2], 2); // v_pool
  compute_encoder.set_input_array(inputs[3], 3); // page_table
  compute_encoder.set_input_array(inputs[4], 4); // sequence_lengths
  compute_encoder.set_input_array(inputs[5], 5); // query_to_seq_map
  compute_encoder.set_input_array(inputs[6], 6); // query_token_offset
  // Upload the parameter struct to the GPU
  compute_encoder.set_bytes(&params_struct, sizeof(PagedAttentionParams), 7);

  // Set the output array
  compute_encoder.set_output_array(out, 8);

  // Declare Pass 1 output arrays outside the if block for later use
  // Initialize with empty arrays
  float dummy_float = 0.0f;
  mx::float16_t dummy_half = mx::float16_t(0.0f);
  mx::array m_locals_pass1(&dummy_float, {0}, mx::float32);
  mx::array s_locals_pass1(&dummy_float, {0}, mx::float32);
  mx::array o_partials_pass1(&dummy_half, {0}, mx::float16);
  
  // Set Pass 1 relevant query map arrays (only for prefill)
  if (is_prefill_) {
      compute_encoder.set_input_array(relevant_query_indices, 9);  // Flat query indices
      compute_encoder.set_input_array(relevant_history_starts, 10); // History start offsets on page
      compute_encoder.set_input_array(relevant_history_counts, 11); // Number of history tokens on page
      compute_encoder.set_input_array(page_offsets, 12);            // Page start offsets in flat arrays
      compute_encoder.set_input_array(active_pages, 13);            // Active page IDs

      // Step 2: Correct allocation of Pass 1 intermediate output buffers
      // Structure of Arrays layout for better Pass 2 access:
      // - m_locals_pass1: [TotalQueryTokens][NumTotalQHeads][NumActivePages]
      // - s_locals_pass1: [TotalQueryTokens][NumTotalQHeads][NumActivePages]
      // - o_partials_pass1: [TotalQueryTokens][NumTotalQHeads][NumActivePages][HeadDim]
      
      size_t num_active_pages = active_pages.size();
      size_t total_query_tokens = core_dims.query_token_count;
      size_t total_q_heads = core_dims.num_q_heads;
      
      // Store num_active_pages in params for Pass 2
      params_struct.num_active_pages_in_batch = static_cast<uint32_t>(num_active_pages);

      // Allocate m_locals_pass1: Shape [TotalQueryTokens][NumTotalQHeads][NumActivePages]
      // Initialize with zeros using vector
      size_t m_locals_size = total_query_tokens * total_q_heads * num_active_pages;
      std::vector<float> m_locals_data(m_locals_size, -std::numeric_limits<float>::infinity());  // Initialize with -inf
      m_locals_pass1 = mx::array(m_locals_data.data(),
                                {static_cast<int>(total_query_tokens),
                                 static_cast<int>(total_q_heads),
                                 static_cast<int>(num_active_pages)},
                                mx::float32);

      // Allocate s_locals_pass1: Shape [TotalQueryTokens][NumTotalQHeads][NumActivePages]
      size_t s_locals_size = total_query_tokens * total_q_heads * num_active_pages;
      std::vector<float> s_locals_data(s_locals_size, 0.0f);
      s_locals_pass1 = mx::array(s_locals_data.data(),
                                {static_cast<int>(total_query_tokens),
                                 static_cast<int>(total_q_heads),
                                 static_cast<int>(num_active_pages)},
                                mx::float32);

      // Allocate o_partials_pass1: Shape [TotalQueryTokens][NumTotalQHeads][NumActivePages][HeadDim]
      size_t o_partials_size = total_query_tokens * total_q_heads * num_active_pages * core_dims.head_dim;
      std::vector<mx::float16_t> o_partials_data(o_partials_size, mx::float16_t(0.0f));
      o_partials_pass1 = mx::array(o_partials_data.data(),
                                  {static_cast<int>(total_query_tokens),
                                   static_cast<int>(total_q_heads),
                                   static_cast<int>(num_active_pages),
                                   static_cast<int>(core_dims.head_dim)},
                                  mx::float16);

      // Set these as additional output buffers
      compute_encoder.set_output_array(m_locals_pass1, 14);  // Max scores from Pass 1
      compute_encoder.set_output_array(s_locals_pass1, 15);  // Sum exponentials from Pass 1
      compute_encoder.set_output_array(o_partials_pass1, 16); // Partial outputs from Pass 1

      spdlog::debug("[PAL Primitive Dispatch] Allocated Pass 1 output buffers:");
      spdlog::debug("  m_locals_pass1 shape: [{}, {}, {}]",
                    total_query_tokens, total_q_heads, num_active_pages);
      spdlog::debug("  s_locals_pass1 shape: [{}, {}, {}]",
                    total_query_tokens, total_q_heads, num_active_pages);
      spdlog::debug("  o_partials_pass1 shape: [{}, {}, {}, {}]",
                    total_query_tokens, total_q_heads, num_active_pages, core_dims.head_dim);
  }

  // Configure dispatch grid sizes
  MTL::Size threadgroups_per_grid = MTL::Size(dispatch_grid_width, dispatch_grid_height, 1);
  MTL::Size threads_per_threadgroup = MTL::Size(threads_to_launch, 1, 1);

  spdlog::debug("[PAL Primitive Dispatch] Dispatching kernel with grid: {} x {} x 1",
                dispatch_grid_width, dispatch_grid_height);
  spdlog::debug("[PAL Primitive Dispatch] Dispatching kernel with threads_to_launch: {}", threads_to_launch);

  // Set the threadgroup memory length and dispatch the kernel
  compute_encoder.set_threadgroup_memory_length(memory_layout.total_bytes, 0);
  compute_encoder.dispatch_threadgroups(threadgroups_per_grid, threads_per_threadgroup);
  
  // Step 4: Launch Pass 2 kernel for prefill
  if (is_prefill_) {
      spdlog::debug("[PAL Primitive Dispatch] Launching Pass 2 finalization kernel");
      
      // Get Pass 2 kernel
      const std::string pass2_kernel_name = "paged_attn_prefill_pass2_kernel";
      auto pass2_kernel_state = d.get_kernel(pass2_kernel_name, library_name_for_mlx);
      if (!pass2_kernel_state) {
          throw std::runtime_error("[PAL Primitive] Failed to load Pass 2 kernel: " + pass2_kernel_name);
      }
      
      // Create new encoder for Pass 2
      auto& pass2_compute_encoder = d.get_command_encoder(s.index);
      pass2_compute_encoder.set_compute_pipeline_state(pass2_kernel_state);
      
      // Calculate Pass 2 dispatch grid
      size_t dispatch_grid_width_pass2 = core_dims.query_token_count;   // 1 TG per query token
      size_t dispatch_grid_height_pass2 = core_dims.num_q_heads;        // 1 TG per Q-head
      
      spdlog::debug("[PAL Primitive Dispatch] Pass 2 grid dimensions: {} x {} (query_tokens x q_heads)",
                    dispatch_grid_width_pass2, dispatch_grid_height_pass2);
      
      // Set Pass 1 outputs as Pass 2 inputs
      pass2_compute_encoder.set_input_array(m_locals_pass1, 17);    // Pass 1 max scores
      pass2_compute_encoder.set_input_array(s_locals_pass1, 18);    // Pass 1 sum exponentials
      pass2_compute_encoder.set_input_array(o_partials_pass1, 19);  // Pass 1 partial outputs
      
      // Set parameters for Pass 2
      pass2_compute_encoder.set_bytes(&params_struct, sizeof(PagedAttentionParams), 7);
      
      // Set final output buffer
      pass2_compute_encoder.set_output_array(out, 8);
      
      // Calculate Pass 2 threadgroup memory requirements
      // Pass 2 needs much less memory - just for reductions
      size_t pass2_threads_per_group = std::min(static_cast<size_t>(256), max_threads_device);
      pass2_threads_per_group = ((pass2_threads_per_group + execution_width - 1) / execution_width) * execution_width;
      
      // Calculate memory for Pass 2 (simplified - just reduction scratch space)
      size_t pass2_tg_mem_bytes = 0;
      pass2_tg_mem_bytes += pass2_threads_per_group * sizeof(float);        // Thread max scratch
      pass2_tg_mem_bytes += pass2_threads_per_group * sizeof(float);        // Thread sum scratch  
      pass2_tg_mem_bytes += 2 * sizeof(float);                              // Global stats (M, S)
      pass2_tg_mem_bytes += pass2_threads_per_group * core_dims.head_dim * sizeof(float); // Thread O accumulators
      pass2_tg_mem_bytes += core_dims.head_dim * sizeof(float);             // Final O accumulator
      pass2_tg_mem_bytes += 64;                                              // Alignment padding
      pass2_tg_mem_bytes = (pass2_tg_mem_bytes + 63) & ~63;                 // Align to 64 bytes
      
      // Configure Pass 2 dispatch
      MTL::Size pass2_threadgroups_per_grid = MTL::Size(dispatch_grid_width_pass2, dispatch_grid_height_pass2, 1);
      MTL::Size pass2_threads_per_threadgroup = MTL::Size(pass2_threads_per_group, 1, 1);
      
      spdlog::debug("[PAL Primitive Dispatch] Pass 2 using {} threads per group, {} bytes TG memory",
                    pass2_threads_per_group, pass2_tg_mem_bytes);
      
      // Dispatch Pass 2
      pass2_compute_encoder.set_threadgroup_memory_length(pass2_tg_mem_bytes, 0);
      pass2_compute_encoder.dispatch_threadgroups(pass2_threadgroups_per_grid, pass2_threads_per_threadgroup);
  }
}

bool PagedAttentionPrimitive::is_equivalent(const mx::Primitive& other) const {
  // Check if the other primitive is the same type
  if (typeid(*this) != typeid(other)) {
    return false;
  }

  // Cast and compare stored parameters
  const PagedAttentionPrimitive& other_pa =
      static_cast<const PagedAttentionPrimitive&>(other);
  return (this->num_q_heads_ == other_pa.num_q_heads_ &&
          this->num_kv_heads_ == other_pa.num_kv_heads_ &&
          this->head_dim_ == other_pa.head_dim_ &&
          this->tokens_per_page_ == other_pa.tokens_per_page_ &&
          this->is_prefill_ == other_pa.is_prefill_);
}

std::vector<mx::array> PagedAttentionPrimitive::vjp(
    const std::vector<mx::array>& primals,
    const std::vector<mx::array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<mx::array>& outputs) {
  // VJP for attention is complex. Stub for now.
  throw std::runtime_error("[PagedAttentionPrimitive] VJP not implemented.");
}

std::vector<mx::array> PagedAttentionPrimitive::jvp(
    const std::vector<mx::array>& primals,
    const std::vector<mx::array>& tangents,
    const std::vector<int>& argnums) {
  // JVP for attention is complex. Stub for now.
  throw std::runtime_error("[PagedAttentionPrimitive] JVP not implemented.");
}

std::pair<std::vector<mx::array>, std::vector<int>>
PagedAttentionPrimitive::vmap(const std::vector<mx::array>& inputs,
                              const std::vector<int>& axes) {
  // Vmap for attention might require careful handling of batch/head dimensions.
  throw std::runtime_error("[PagedAttentionPrimitive] Vmap not implemented.");
}

std::vector<mx::Shape> PagedAttentionPrimitive::output_shapes(
    const std::vector<mx::array>& inputs) {
  if (inputs.empty()) {
    throw std::invalid_argument(
        "[PagedAttentionPrimitive::output_shapes] Requires at least one input "
        "(query).");
  }
  if (inputs.size() < 2 || inputs[1].ndim() != 4) {
    throw std::invalid_argument(
        "[PagedAttentionPrimitive::output_shapes] K-pool (inputs[1]) is needed "
        "and must be 4D to determine head_dim for output shape.");
  }

  const auto& q = inputs[0];
  uint32_t current_head_dim = inputs[1].shape(3);  // Get head_dim from K-pool

  if (q.ndim() == 3) {  // Q is [NumTokens, NumQHeads, QueryHeadDim]
    // Output will be [NumTokens * NumQHeads, AttentionOutputHeadDim]
    return {{q.shape(0) * q.shape(1), static_cast<int>(current_head_dim)}};
  } else if (q.ndim() == 2) {  // Q is [NumItems, QueryHeadDim]
    // Output will be [NumItems, AttentionOutputHeadDim]
    return {{q.shape(0), static_cast<int>(current_head_dim)}};
  } else if (q.ndim() == 1) {  // Q is [NumItems] with QueryHeadDim=1
    // The C++ validation for 1D queries already ensures head_dim is 1
    return {{q.shape(0), static_cast<int>(current_head_dim)}};
  } else {
    throw std::invalid_argument(
        "[PagedAttentionPrimitive::output_shapes] Query input 'q' must be 1D, "
        "2D, or 3D.");
  }
}

void PagedAttentionPrimitive::print(std::ostream& os) {
  os << "PagedAttentionPrimitive(num_q_heads=" << num_q_heads_ << ", num_kv_heads=" << num_kv_heads_
     << ", head_dim=" << head_dim_ << ", tokens_per_page=" << tokens_per_page_
     << ", is_prefill=" << (is_prefill_ ? "true" : "false") << ")";
}

}  // namespace pal::cpp
