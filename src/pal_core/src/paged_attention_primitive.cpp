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

#if __has_include(<spdlog/spdlog.h>)
    #include <spdlog/spdlog.h>
    #define PAL_HAS_SPDLOG 1
#else
    #define PAL_HAS_SPDLOG 0
    // Minimal fallback if spdlog is not available (e.g., for standalone builds without FetchContent run)
    // This could be a no-op or a simple std::cerr wrapper. For now, let's make it a no-op.
    namespace spdlog {
        template<typename... Args> void trace(Args... args) {}
        template<typename... Args> void debug(Args... args) {}
        template<typename... Args> void info(Args... args) {}
        template<typename... Args> void warn(Args... args) {}
        template<typename... Args> void error(Args... args) {}
        template<typename... Args> void critical(Args... args) {}
    } // namespace spdlog
#endif

#include <mlx/allocator.h>
#include <mlx/array.h>
#include <mlx/backend/common/utils.h>
#include <mlx/backend/cpu/encoder.h>
#include <mlx/backend/metal/device.h>
#include <mlx/backend/metal/metal.h>
#include <mlx/backend/metal/utils.h>
#include <mlx/utils.h>

#include "shaders/paged_attention_types.h"

namespace mx = mlx::core;

namespace pal::cpp {

// Expected size for PagedAttentionParams: 8 uint32_t (32 bytes) + 1 float (4 bytes) = 36 bytes.
// alignas(16) means total size is 48, as it's padded to multiple of 16.
// Note: We use 64-byte alignment for threadgroup memory, but the struct itself remains 16-byte aligned.
constexpr size_t kExpectedPagedAttentionParamsSize = 48;
static_assert(
    sizeof(PagedAttentionParams) == kExpectedPagedAttentionParamsSize,
    "sizeof(PagedAttentionParams) mismatch between C++ and expected size (48 bytes). "
    "Check paged_attention_types.h, members, and padding.");

PagedAttentionPrimitive::PagedAttentionPrimitive(
    mx::StreamOrDevice stream_or_device,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int tokens_per_page)
    : mx::UnaryPrimitive(mx::to_stream(stream_or_device)),
      num_q_heads_(num_q_heads),
      num_kv_heads_(num_kv_heads),
      head_dim_(head_dim),
      tokens_per_page_(tokens_per_page) {
  spdlog::debug(
      "[PAL Primitive] Constructed with params: num_q_heads={}, "
      "num_kv_heads={}, head_dim={}, tokens_per_page={}",
      num_q_heads_, num_kv_heads_, head_dim_, tokens_per_page_);
}

void PagedAttentionPrimitive::eval_cpu(const std::vector<mx::array>& inputs,
                                       mx::array& out) {
  spdlog::debug(
      "[PAL Primitive] PagedAttentionPrimitive::eval_cpu called (not "
      "supported).");
  throw std::runtime_error(
      "[PagedAttentionPrimitive] CPU evaluation is not supported for paged "
      "attention.");
}

void PagedAttentionPrimitive::eval_gpu(const std::vector<mx::array>& inputs,
                                       mx::array& out) {
  // Validate input count
  if (inputs.size() != 7) {
    throw std::invalid_argument(
        "[PagedAttentionPrimitive::eval_gpu] Expected 7 inputs (Q, K_pool, "
        "V_pool, Page_table, SeqLens, QToSeqMap, QOffsets), received " +
        std::to_string(inputs.size()));
  }

  // Extract inputs
  const auto& q = inputs[0];
  const auto& k_pool = inputs[1];
  const auto& v_pool = inputs[2];
  const auto& page_table = inputs[3];
  const auto& sequence_lengths = inputs[4];
  const auto& query_to_seq_map = inputs[5];
  const auto& query_token_offset = inputs[6];

  // Type checks for inputs (intentionally abbreviated as commented in original)
  if (q.dtype() != mx::float16 || k_pool.dtype() != mx::float16 ||
      v_pool.dtype() != mx::float16) {
    /* Type checks preserved from original */
  }
  if (page_table.dtype() != mx::uint32) {
    /* Type checks preserved from original */
  }
  if (sequence_lengths.dtype() != mx::int32 ||
      query_to_seq_map.dtype() != mx::int32 ||
      query_token_offset.dtype() != mx::int32) {
    /* Type checks preserved from original */
  }

  // Prepare Metal kernel and command encoder
  auto& s = stream();
  auto& d = mlx::core::metal::device(s.device);

  size_t bytes = out.nbytes();
  out.set_data(mx::allocator::malloc(bytes));

  const std::string library_name_for_mlx = "pal";
  const std::string kernel_name = "paged_attn_kernel";
  MTL::ComputePipelineState* kernel_pipeline_state =
      d.get_kernel(kernel_name, library_name_for_mlx);
  auto& compute_encoder = d.get_command_encoder(s.index);
  compute_encoder.set_compute_pipeline_state(kernel_pipeline_state);

  // Create and populate PagedAttentionParams struct
  PagedAttentionParams params_struct;

  // Debug logging of parameter structure memory layout
  spdlog::trace("[PAL DEBUG PARAMS] C++ sizeof(PagedAttentionParams): {} bytes.",
                sizeof(PagedAttentionParams));
  spdlog::trace("[PAL DEBUG PARAMS] params_struct ADDRESS: {}",
                fmt::ptr(&params_struct));
  spdlog::trace("[PAL DEBUG PARAMS] params_struct.head_dim ADDRESS: {}",
                fmt::ptr(&params_struct.head_dim));
  spdlog::trace("[PAL DEBUG PARAMS] offsetof(PagedAttentionParams, num_q_heads): {}",
                offsetof(PagedAttentionParams, num_q_heads));
  spdlog::trace("[PAL DEBUG PARAMS] offsetof(PagedAttentionParams, num_kv_heads): {}",
                offsetof(PagedAttentionParams, num_kv_heads));
  spdlog::trace("[PAL DEBUG PARAMS] offsetof(PagedAttentionParams, head_dim): {}",
                offsetof(PagedAttentionParams, head_dim));
  spdlog::trace("[PAL DEBUG PARAMS] offsetof(PagedAttentionParams, tokens_per_page): {}",
                offsetof(PagedAttentionParams, tokens_per_page));
  // Parameter scale has been removed
  // Parameter total_items_in_dispatch has been removed

  // Validate K/V pool geometry
  if (k_pool.ndim() != 4) {  // Expecting [NumPhysPages, TokensPerPage, NumKVHeads, HeadDim]
    throw std::invalid_argument("[PagedAttentionPrimitive] k_pool must be 4D.");
  }

  // Extract and validate tokens_per_page from KV pool
  int tokens_per_page_from_k_pool = k_pool.shape(1);
  if (this->tokens_per_page_ > 0 &&
      this->tokens_per_page_ != tokens_per_page_from_k_pool) {
    std::string error_msg =
        "[PagedAttentionPrimitive] Mismatch: tokens_per_page at construction (" +
        std::to_string(this->tokens_per_page_) +
        ") does not match k_pool.shape(1) (" +
        std::to_string(tokens_per_page_from_k_pool) + ")";
    throw std::invalid_argument(error_msg);
  }

  // Extract parameters from KV pool shape
  params_struct.tokens_per_page = tokens_per_page_from_k_pool;
  params_struct.num_kv_heads = k_pool.shape(2);
  params_struct.head_dim = k_pool.shape(3);

  // Constants for head_dim validation
  constexpr uint32_t kMaxAccumulationTile = 64;
  constexpr uint32_t kMaxHeadDimMetalInKernel = 256; // Match Metal's kMaxHeadDimMetal

  // Validate head_dim
  if (params_struct.head_dim == 0) {
    throw std::invalid_argument(
        "[PagedAttentionPrimitive] head_dim cannot be 0.");
  }
  if (params_struct.head_dim % 4 != 0) {
    throw std::invalid_argument(
        "[PagedAttentionPrimitive] head_dim (" +
        std::to_string(params_struct.head_dim) +
        ") must be a multiple of 4 for vectorized kernel execution.");
  }
  if (params_struct.head_dim > kMaxHeadDimMetalInKernel) {
    throw std::invalid_argument(
        "[PagedAttentionPrimitive] params_struct.head_dim (" +
        std::to_string(params_struct.head_dim) +
        ") exceeds kernel's internal processing limit kMaxHeadDimMetal (" +
        std::to_string(kMaxHeadDimMetalInKernel) + ").");
  }
  if (params_struct.head_dim > kMaxAccumulationTile * 1024) {
    throw std::invalid_argument(
        "[PagedAttentionPrimitive] params_struct.head_dim (" +
        std::to_string(params_struct.head_dim) +
        ") is excessively large for the tiled kernel approach.");
  }

  // Check if head_dim alone would cause threadgroup memory overflow
  // Get threadgroup memory early to check if head_dim is too large by itself
  auto* device_ptr = d.mtl_device();
  size_t max_tg_memory_bytes = device_ptr->maxThreadgroupMemoryLength();
  size_t q_shmem_bytes = params_struct.head_dim * sizeof(float);
  if (q_shmem_bytes > max_tg_memory_bytes) {
    throw std::runtime_error(
        "[PagedAttentionPrimitive] q_shmem size alone (" +
        std::to_string(q_shmem_bytes) + " bytes) exceeds device's maximum threadgroup memory (" +
        std::to_string(max_tg_memory_bytes) + " bytes). Reduce head_dim to a smaller value.");
  }

  // Validate query dimensions
  if (q.ndim() < 1) {
    throw std::invalid_argument(
        "Queries 'q' must have at least 1 dimension.");
  }

  // Check query format and set num_q_heads
  if (q.ndim() == 3) {
    if (q.shape(2) != params_struct.head_dim) {
      throw std::invalid_argument(
          "[PagedAttentionPrimitive] For 3D query input [NumTokens, NumQHeads, "
          "HeadDim], the HeadDim must match K/V head_dim.");
    }
    params_struct.num_q_heads = q.shape(1);
  } else if (q.ndim() == 2) {
    if (q.shape(1) != params_struct.head_dim) {
      throw std::invalid_argument(
          "[PagedAttentionPrimitive] For 2D query input [NumDispatchThreads, "
          "HeadDim], the HeadDim must match K/V head_dim.");
    }
    params_struct.num_q_heads = 1;
  } else if (q.ndim() == 1) {
    if (params_struct.head_dim != 1) {
      throw std::invalid_argument(
          "[PagedAttentionPrimitive] For 1D query input (interpreted as scalar "
          "items), the K/V head_dim (params_struct.head_dim = " +
          std::to_string(params_struct.head_dim) +
          ") must also be 1. The kernel will attempt to read head_dim elements "
          "for Q.");
    }
    params_struct.num_q_heads = 1;  // Each item is effectively its own "Q-head" of size 1.
  } else {
    throw std::invalid_argument(
        "[PagedAttentionPrimitive] Query 'q' ndim not supported.");
  }

  // Scale factor (1.0f / sqrt(head_dim)) is now calculated in the kernel

  // Validate page table and extract logical blocks parameters
  if (page_table.ndim() != 2) {
    throw std::invalid_argument(
        "[PagedAttentionPrimitive] page_table must be 2D [NumBatchSeq, "
        "MaxLogBlocksPerSeq].");
  }
  params_struct.max_logical_blocks_per_seq = page_table.shape(1);

  // Set pool geometry parameters
  params_struct.num_physical_pages_in_pool = k_pool.shape(0);
  params_struct.num_sequences_in_batch = page_table.shape(0);

  // Validate Grouped Query Attention (GQA) parameters
  if (params_struct.num_q_heads > params_struct.num_kv_heads) {  // GQA case
    if (params_struct.num_kv_heads == 0) {                       // Avoid division by zero
      throw std::invalid_argument(
          "[PagedAttentionPrimitive] num_kv_heads cannot be 0 if num_q_heads > "
          "0 for GQA.");
    }
    if (params_struct.num_q_heads % params_struct.num_kv_heads != 0) {
      throw std::invalid_argument(
          "[PagedAttentionPrimitive] For GQA (num_q_heads > num_kv_heads), "
          "num_q_heads must be an integer multiple of num_kv_heads.");
    }
  }

  // Set input arrays for the compute encoder
  compute_encoder.set_input_array(q, 0);
  compute_encoder.set_input_array(k_pool, 1);
  compute_encoder.set_input_array(v_pool, 2);
  compute_encoder.set_input_array(page_table, 3);
  compute_encoder.set_input_array(sequence_lengths, 4);
  compute_encoder.set_input_array(query_to_seq_map, 5);
  compute_encoder.set_input_array(query_token_offset, 6);
  // Will set bytes at index 7 after all params are populated
  compute_encoder.set_output_array(out, 8);

  // Calculate number of items to process
  size_t num_items_to_process = 0;
  if (q.ndim() == 3) {
    num_items_to_process = q.shape(0) * q.shape(1);
  } else {
    num_items_to_process = q.shape(0);
  }

  // Early return if no items to process
  if (num_items_to_process == 0) {
    spdlog::info(
        "[PagedAttentionPrimitive] No items to process "
        "(num_items_to_process=0). Returning empty result.");
    return;
  }

  // Validate sizes of side-band arrays
  // For 3D queries [NumTokens, NumQHeads, HeadDim], handle the special case
  // where query_to_seq_map size might only match the NumTokens dimension
  size_t expected_size = q.ndim() == 3 ? q.shape(0) : num_items_to_process;

  if (query_to_seq_map.size() != expected_size) {
    throw std::invalid_argument(
        "[PagedAttentionPrimitive] query_to_seq_map size must match number of "
        "tokens (or items for 1D/2D queries)");
  }
  if (query_token_offset.size() != expected_size) {
    throw std::invalid_argument(
        "[PagedAttentionPrimitive] query_token_offset size must match number "
        "of tokens (or items for 1D/2D queries)");
  }

  // No longer setting total_items_in_dispatch

  // Debug output for parameter values
  spdlog::trace("[PAL DEBUG PARAMS] C++ sizeof(PagedAttentionParams): {} bytes.",
                sizeof(PagedAttentionParams));
  spdlog::trace("[PAL DEBUG PARAMS] C++ offsetof(num_q_heads): {}",
                offsetof(PagedAttentionParams, num_q_heads));
  spdlog::trace("[PAL DEBUG PARAMS] C++ offsetof(num_kv_heads): {}",
                offsetof(PagedAttentionParams, num_kv_heads));
  spdlog::trace("[PAL DEBUG PARAMS] C++ offsetof(head_dim): {}",
                offsetof(PagedAttentionParams, head_dim));
  spdlog::trace("[PAL DEBUG PARAMS] C++ offsetof(tokens_per_page): {}",
                offsetof(PagedAttentionParams, tokens_per_page));

  // Log parameter values being sent to the kernel
  spdlog::debug("[PAL SENDING PARAMS] num_q_heads: {}",
                params_struct.num_q_heads);
  spdlog::debug("[PAL SENDING PARAMS] num_kv_heads: {}",
                params_struct.num_kv_heads);
  spdlog::debug("[PAL SENDING PARAMS] head_dim: {}", params_struct.head_dim);
  spdlog::debug("[PAL SENDING PARAMS] tokens_per_page: {}",
                params_struct.tokens_per_page);
  // Scale is now calculated in the kernel

  // Log additional parameter values
  spdlog::trace("[PAL DEBUG PARAMS] max_logical_blocks_per_seq = {}",
                params_struct.max_logical_blocks_per_seq);
  spdlog::trace("[PAL DEBUG PARAMS] num_physical_pages_in_pool = {}",
                params_struct.num_physical_pages_in_pool);
  spdlog::trace("[PAL DEBUG PARAMS] num_sequences_in_batch = {}",
                params_struct.num_sequences_in_batch);

  // Log parameter memory addresses for debugging
  spdlog::trace("[PAL DEBUG PARAMS] params_struct ADDRESS: {}",
                fmt::ptr(&params_struct));
  spdlog::trace("[PAL DEBUG PARAMS] params_struct.head_dim ADDRESS: {}",
                fmt::ptr(&params_struct.head_dim));

  // Configure thread and threadgroup dimensions
  const size_t default_threads_per_item_group = 64;
  auto max_threads = kernel_pipeline_state->maxTotalThreadsPerThreadgroup();
  spdlog::debug("[PAL Primitive] Device maxTotalThreadsPerThreadgroup: {}",
                max_threads);

  const size_t threads_per_item_group =
      std::min(default_threads_per_item_group, max_threads);
  if (threads_per_item_group == 0) {
    throw std::runtime_error(
        "[PagedAttentionPrimitive] Calculated threads_per_item_group is 0. "
        "Device maxTotalThreadsPerThreadgroup might be 0 or default is 0. "
        "This is invalid.");
  }
  spdlog::debug("[PAL Primitive] Effective threads_per_item_group for dispatch: {}",
                threads_per_item_group);

  // These parameters are no longer used in the params struct

  // Set runtime V-accumulation tile size
  constexpr uint32_t kMaxKernelAccumTileSize = 64; // Matches kMaxAccumulationTile in Metal
  params_struct.max_accum_tile_runtime = kMaxKernelAccumTileSize;

  // Calculate a principled clamp value based on float16 denormalized minimum
  // Smallest positive half float subnormal: 2^-24 ≈ 5.96046e-08
  // log(5.96046e-08) ≈ -16.6355
  const float fp16_denorm_min_val = 5.9604644775390625e-08f; // 2^-24
  // Add a safety margin by making it more negative (-1.0f extra)
  params_struct.log_exp_min_clamp = logf(fp16_denorm_min_val);

  // Calculate threadgroup memory size needed for kernel execution
  constexpr uint32_t kMaxSimdGroupsPerThreadgroup = 8;
  const uint32_t head_dim = params_struct.head_dim;
  // max_tg_memory_bytes is already set earlier in the code

  // Threadgroup memory layout calculation with proper 64-byte alignment between sections
  // Starting with array sizes first:
  size_t q_shmem_size = head_dim;
  size_t G_partial_max_scores_size = threads_per_item_group;
  size_t G_simd_reduced_maxes_size = kMaxSimdGroupsPerThreadgroup;
  size_t G_simd_reduced_adjusted_sum_exps_size = kMaxSimdGroupsPerThreadgroup;
  size_t G_final_max_for_item_size = 1;
  size_t G_final_sum_exp_for_item_size = 1;
  size_t G_simd_group_v_sums_size = kMaxSimdGroupsPerThreadgroup * 4; // float4 per SIMD group

  // Calculate threadgroup memory with correct alignment of each section
  const uint32_t head_dim_cpp = params_struct.head_dim; // Use the value from params_struct
  const size_t threads_per_tg_cpp = threads_per_item_group; // From earlier calculation
  const uint32_t num_simd_groups_cpp = (threads_per_tg_cpp + 31u) / 32u; // Consistent with kernel

  uintptr_t current_offset_bytes = 0;

  // 1. q_shmem: head_dim_cpp floats
  size_t q_shmem_section_bytes = head_dim_cpp * sizeof(float);
  current_offset_bytes += q_shmem_section_bytes;

  // 2. G_partial_max_scores: threads_per_tg_cpp floats
  current_offset_bytes = (current_offset_bytes + 63u) & ~63u; // Align start of this section to 64 bytes
  size_t G_partial_max_scores_section_bytes = threads_per_tg_cpp * sizeof(float);
  current_offset_bytes += G_partial_max_scores_section_bytes;

  // 3. G_simd_reduced_maxes: num_simd_groups_cpp floats
  current_offset_bytes = (current_offset_bytes + 63u) & ~63u; // Align to 64 bytes
  size_t G_simd_reduced_maxes_section_bytes = num_simd_groups_cpp * sizeof(float);
  current_offset_bytes += G_simd_reduced_maxes_section_bytes;

  // 4. G_simd_reduced_adjusted_sum_exps: num_simd_groups_cpp floats
  current_offset_bytes = (current_offset_bytes + 63u) & ~63u; // Align to 64 bytes
  size_t G_simd_reduced_adjusted_sum_exps_section_bytes = num_simd_groups_cpp * sizeof(float);
  current_offset_bytes += G_simd_reduced_adjusted_sum_exps_section_bytes;

  // 5. G_final_max_for_item: 1 float
  current_offset_bytes = (current_offset_bytes + 63u) & ~63u; // Align to 64 bytes
  size_t G_final_max_for_item_section_bytes = 1 * sizeof(float);
  current_offset_bytes += G_final_max_for_item_section_bytes;

  // 6. G_final_sum_exp_for_item: 1 float
  current_offset_bytes = (current_offset_bytes + 63u) & ~63u; // Align to 64 bytes
  size_t G_final_sum_exp_for_item_section_bytes = 1 * sizeof(float);
  current_offset_bytes += G_final_sum_exp_for_item_section_bytes;

  // 7. G_simd_group_v_sums (for current V-tile reduction)
  current_offset_bytes = (current_offset_bytes + 63u) & ~63u; // Align to 64 bytes
  size_t G_simd_group_v_sums_section_bytes = num_simd_groups_cpp * sizeof(float) * 4; // float4
  current_offset_bytes += G_simd_group_v_sums_section_bytes;

  // The total fixed memory needed is the current_offset_bytes, properly aligned
  size_t fixed_tg_mem_needed_bytes = (current_offset_bytes + 63u) & ~63u; // Total for fixed parts, aligned

  spdlog::debug("[PAL Primitive] Fixed tg_mem needed (Q, softmax_stats, V_reduct_scratch): {} bytes", fixed_tg_mem_needed_bytes);

  if (fixed_tg_mem_needed_bytes >= max_tg_memory_bytes) {
      throw std::runtime_error(
          "[PagedAttentionPrimitive] Fixed threadgroup memory components alone (" +
          std::to_string(fixed_tg_mem_needed_bytes) + " bytes) exceed or meet device limit (" +
          std::to_string(max_tg_memory_bytes) + " bytes). Cannot allocate for K/V/score tiles."
      );
  }

  // --- Calculate tile_size_T_runtime for score stashing ---
  size_t available_mem_for_kv_score_tiles = max_tg_memory_bytes - fixed_tg_mem_needed_bytes;
  spdlog::debug("[PAL Primitive] Max TG mem: {}, Available for K/V/score tiles: {} bytes", max_tg_memory_bytes, available_mem_for_kv_score_tiles);

  // Memory per history item in a tile:
  // One buffer for K or V (D floats, as V reuses K_tile's memory), plus one float for score.
  // This coupling (V reusing K_tile) is important for this memory calculation.
  size_t bytes_per_hist_item_in_tile = (params_struct.head_dim * sizeof(float)) + sizeof(float);
  if (params_struct.head_dim == 0) { // Should be caught by earlier validation
      bytes_per_hist_item_in_tile = sizeof(float); // Avoid division by zero if head_dim is 0
  }

  uint32_t calculated_tile_size_T = 0;
  if (bytes_per_hist_item_in_tile > 0) {
      calculated_tile_size_T = static_cast<uint32_t>(available_mem_for_kv_score_tiles / bytes_per_hist_item_in_tile);
  }

  spdlog::debug("[PAL Primitive] Max possible tile size T (before clamps): {}", calculated_tile_size_T);

  // Apply O3's clamps and warnings
  const uint32_t min_tile_size_T = 16; // Hard floor
  const uint32_t practical_max_tile_size_T = 256; // Practical upper cap, can be tuned

  params_struct.tile_size_T_runtime = std::max(calculated_tile_size_T, min_tile_size_T);
  params_struct.tile_size_T_runtime = std::min(params_struct.tile_size_T_runtime, practical_max_tile_size_T);

  // Ensure it's a multiple of 4 for potential vector processing of scores, or just for tidiness
  if (params_struct.tile_size_T_runtime > 0) {
      params_struct.tile_size_T_runtime = (params_struct.tile_size_T_runtime / 4u) * 4u;
      if (params_struct.tile_size_T_runtime == 0 && calculated_tile_size_T > 0) { // Avoid making it 0 if it was viable
          params_struct.tile_size_T_runtime = min_tile_size_T; // Reset to min if rounding down made it zero
      }
  } else { // If calculated_tile_size_T was 0
       params_struct.tile_size_T_runtime = min_tile_size_T; // Default to min if calculation yielded zero
  }

  // Final check if tile size became 0 due to constraints, force to a minimum if there's any history
  if (params_struct.tile_size_T_runtime == 0) {
      params_struct.tile_size_T_runtime = min_tile_size_T;
       spdlog::warn("[PAL Primitive] Calculated tile_size_T_runtime was 0, forcing to min_tile_size_T={}", min_tile_size_T);
  }

  spdlog::debug("[PAL Primitive] Final calculated params_struct.tile_size_T_runtime: {}", params_struct.tile_size_T_runtime);

  // This is where you would get item_effective_history_length if it were available at primitive creation.
  // Since it's per-item, the warning for >512 tiles is better placed in the kernel or if we had batch-level max history.
  // For now, just log the calculated T.
  // If (item_effective_history_length / params_struct.tile_size_T_runtime > 512 && params_struct.tile_size_T_runtime > 0) {
  //     spdlog::trace("[PAL Primitive] Warning: Number of history tiles exceeds 512, performance may degrade.");
  // }

  spdlog::debug(
      "[PAL Primitive] Setting max_accum_tile_runtime (kernel's tile capacity) to: {}",
      params_struct.max_accum_tile_runtime);
  spdlog::debug(
      "[PAL Primitive] Setting log_exp_min_clamp (based on fp16_denorm_min): {}",
      params_struct.log_exp_min_clamp);
  spdlog::debug(
      "[PAL Primitive] Setting tile_size_T_runtime (for history tiling): {}",
      params_struct.tile_size_T_runtime);

  // Before finalizing parameters, recalculate the memory required using the final tile_size_T_runtime value
  // This ensures we use the actual memory requirements after all clamping and rounding
  uintptr_t final_current_offset_for_tiles = fixed_tg_mem_needed_bytes;
  final_current_offset_for_tiles = (final_current_offset_for_tiles + 63u) & ~63u; // Align start of K/V tile

  // Calculate memory for finalized K/V and score tiles
  size_t mem_for_kv_tile_final = params_struct.tile_size_T_runtime * params_struct.head_dim * sizeof(float);
  final_current_offset_for_tiles += mem_for_kv_tile_final;

  final_current_offset_for_tiles = (final_current_offset_for_tiles + 63u) & ~63u; // Align start of score tile
  size_t mem_for_score_tile_final = params_struct.tile_size_T_runtime * sizeof(float);
  final_current_offset_for_tiles += mem_for_score_tile_final;

  size_t total_mem_before_padding = (final_current_offset_for_tiles + 63u) & ~63u;
  size_t padding_bytes = 32;
  size_t final_tg_memory_bytes = total_mem_before_padding + padding_bytes;

  spdlog::debug("[PAL Primitive] TG Mem - K/V Tile (final) ({}x{} floats): {} bytes",
                params_struct.tile_size_T_runtime, params_struct.head_dim, mem_for_kv_tile_final);
  spdlog::debug("[PAL Primitive] TG Mem - Score Tile (final) ({} floats): {} bytes",
                params_struct.tile_size_T_runtime, mem_for_score_tile_final);
  spdlog::debug("[PAL Primitive] Added {} padding bytes. Final tg_memory_bytes: {}", padding_bytes, final_tg_memory_bytes);

  // Double-check that the final memory requirement still fits within device limits
  if (final_tg_memory_bytes > max_tg_memory_bytes) {
    throw std::runtime_error(
        "[PagedAttentionPrimitive] Final calculated threadgroup memory (" +
        std::to_string(final_tg_memory_bytes) + " bytes) exceeds device limit (" +
        std::to_string(max_tg_memory_bytes) + " bytes) after tile_size_T_runtime adjustments."
    );
  }

  // Upload the complete parameter struct to the GPU
  compute_encoder.set_bytes(&params_struct, sizeof(PagedAttentionParams), 7);

  // Configure dispatch grid sizes
  MTL::Size threadgroups_per_grid = MTL::Size(num_items_to_process, 1, 1);
  MTL::Size threads_per_threadgroup = MTL::Size(threads_per_item_group, 1, 1);

  // Use the final calculated threadgroup memory size
  size_t tg_memory_bytes = final_tg_memory_bytes;

  spdlog::debug("[PAL Primitive] Final calculated tg_memory_bytes (including K/V/Score tiles): {}", tg_memory_bytes);

  // Set the threadgroup memory length and dispatch the kernel
  compute_encoder.set_threadgroup_memory_length(tg_memory_bytes, 0);
  compute_encoder.dispatch_threadgroups(threadgroups_per_grid,
                                        threads_per_threadgroup);
}

void PagedAttentionPrimitive::print(std::ostream& os) {
  os << "PagedAttention";
  // Add stored parameters for better debugging
  os << "(qheads=" << num_q_heads_ << ",kvheads=" << num_kv_heads_
     << ",dim=" << head_dim_ << ",tpp=" << tokens_per_page_ << ")";
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
          this->tokens_per_page_ == other_pa.tokens_per_page_);
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

}  // namespace pal::cpp
