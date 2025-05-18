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
#include <spdlog/spdlog.h>

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

// fp16 - log(2^-24) = -16.6355
constexpr static float kLogFp16DenormMinVal = -16.6355f;
constexpr static uint32_t kDefaultPaddingFloatsPerRow = 8;

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
    size_t final_guard_bytes{0};
    size_t total_bytes{0};
};


static ThreadgroupMemoryLayout calculate_threadgroup_memory_breakdown_and_total(
    const PagedAttentionParams& params,
    size_t threads_per_group
) {
    ThreadgroupMemoryLayout layout;
    uintptr_t tg_mem_current_offset_bytes = 0;

    // Calculate the number of SIMD groups
    constexpr size_t kSimdLanesPerGroup = 32;
    const uint32_t num_simd_groups = (threads_per_group + kSimdLanesPerGroup - 1) / kSimdLanesPerGroup;

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

    // Use pad_floats_per_row from params (should be already set by caller)
    uint32_t padded_head_dim_for_tile = params.head_dim + params.pad_floats_per_row;

    // K_tile requires tile_size_T_runtime * padded_head_dim_for_tile * sizeof(float) bytes
    layout.k_tile_bytes = params.tile_size_T_runtime * padded_head_dim_for_tile * sizeof(float);
    tg_mem_current_offset_bytes += layout.k_tile_bytes;

    // 9. V Tile memory - for caching V vectors with padding for bank conflict avoidance
    tg_mem_current_offset_bytes = (tg_mem_current_offset_bytes + kAlignmentMask) & ~kAlignmentMask;

    // V_tile requires tile_size_T_runtime * padded_head_dim_for_tile * sizeof(float) bytes
    layout.v_tile_bytes = params.tile_size_T_runtime * padded_head_dim_for_tile * sizeof(float);
    tg_mem_current_offset_bytes += layout.v_tile_bytes;

    // Final padding guard
    constexpr size_t kFinalTgMemoryPaddingGuardBytes = 32;
    layout.final_guard_bytes = kFinalTgMemoryPaddingGuardBytes;
    tg_mem_current_offset_bytes += layout.final_guard_bytes;

    // Ensure final size is aligned to 64-byte boundary
    layout.total_bytes = (tg_mem_current_offset_bytes + kAlignmentMask) & ~kAlignmentMask;
    spdlog::debug("[PAL TGMemCalc] Total calculated tg_memory_bytes: {} bytes", layout.total_bytes);

    return layout;
}

static ThreadgroupMemoryLayout prepare_inputs_layout_memory(
    PagedAttentionParams& params,
    const CoreDims& extracted_core_dims,
    const mx::array& k_pool_arr,
    const mx::array& page_table_arr,
    MTL::Device* mtl_device_ptr,
    size_t threads_per_item_group_for_dispatch
) {
    params.num_q_heads              = extracted_core_dims.num_q_heads;
    params.num_kv_heads             = extracted_core_dims.num_kv_heads;
    params.head_dim                 = extracted_core_dims.head_dim;
    params.tokens_per_page          = extracted_core_dims.tokens_per_page;

    // Kernel constants
    params.pad_floats_per_row       = kDefaultPaddingFloatsPerRow;
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
        threads_per_item_group_for_dispatch
    ).total_bytes - kFinalTgMemoryPaddingGuardBytes;
    // Add padding for alignment and final guard bytes
    precise_fixed_tg_mem_bytes = (precise_fixed_tg_mem_bytes + kAlignmentMask) & ~kAlignmentMask;

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

    // 3. Compute "bytes per history token" for dynamic tiles (K_tile + V_tile) with padding
    uint32_t padded_head_dim_for_tile_calc = params.head_dim + params.pad_floats_per_row;
    size_t bytes_per_token_in_tile = (padded_head_dim_for_tile_calc * sizeof(float)) /*K_tile_padded*/ +
                                    (padded_head_dim_for_tile_calc * sizeof(float)) /*V_tile_padded*/;
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

    if (params.tile_size_T_runtime == 0) {
        throw std::runtime_error(
            "[PAL Primitive] head_dim/tokens_per_page combination leaves no "
            "scratch space for a single KV token.");
    }

    return calculate_threadgroup_memory_breakdown_and_total(params, threads_per_item_group_for_dispatch);
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
    if (page_table.dtype() != mx::uint32) {
         throw std::invalid_argument("[PAL Primitive Validate] page_table must be uint32.");
    }
    if (sequence_lengths.dtype() != mx::int32 || query_to_seq_map.dtype() != mx::int32 || query_token_offset.dtype() != mx::int32) {
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

    constexpr uint32_t kMaxHeadDimMetalInKernel = 256;
    if (dims.head_dim == 0) {
        throw std::invalid_argument("[PAL Primitive Validate] head_dim from K-pool cannot be 0.");
    }
    if (dims.head_dim > kMaxHeadDimMetalInKernel) {
         throw std::invalid_argument("[PAL Primitive Validate] head_dim (" + std::to_string(dims.head_dim) +
                                   ") exceeds kernel's internal processing limit kMaxHeadDimMetal (" +
                                   std::to_string(kMaxHeadDimMetalInKernel) + ").");
    }
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

    // --- GQA Validation ---
    if (dims.num_q_heads > dims.num_kv_heads) {
        if (dims.num_kv_heads == 0) {
            throw std::invalid_argument("[PAL Primitive Validate] GQA: num_kv_heads cannot be 0 if num_q_heads > 0.");
        }
        if (dims.num_q_heads % dims.num_kv_heads != 0) {
            throw std::invalid_argument("[PAL Primitive Validate] GQA: num_q_heads (" + std::to_string(dims.num_q_heads) +
                                       ") must be an integer multiple of num_kv_heads (" + std::to_string(dims.num_kv_heads) + ").");
        }
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

  const std::string library_name_for_mlx = "pal";
  const std::string kernel_name = "paged_attn_kernel";
  // Get the device from the stream
  auto& s = stream();
  auto& d = mlx::core::metal::device(s.device);
  kernel_state_ = d.get_kernel(kernel_name, library_name_for_mlx);
  if (!kernel_state_) {
    throw std::runtime_error("[PAL Primitive] Failed to load kernel: " + kernel_name);
  }
}

void PagedAttentionPrimitive::eval_cpu(const std::vector<mx::array>& inputs, mx::array& out) {
    std::ostringstream oss;
    this->print(oss);
    spdlog::error("[PAL Primitive] CPU evaluation is not supported for {}.", oss.str());
    throw std::runtime_error(
        "[PagedAttentionPrimitive] CPU evaluation is not supported for " + oss.str());
}

void PagedAttentionPrimitive::dispatch_metal_kernel(
    mlx::core::metal::CommandEncoder& compute_encoder,
    MTL::ComputePipelineState* kernel_pso,
    const std::vector<mx::array>& kernel_inputs,
    mx::array& kernel_out_array,
    const PagedAttentionParams& kernel_params,
    size_t total_tg_memory_bytes,
    size_t items_to_process_count,
    size_t threads_per_group_count) {

  compute_encoder.set_compute_pipeline_state(kernel_pso); // Set PSO here

  // Set input arrays for the compute encoder
  compute_encoder.set_input_array(kernel_inputs[0], 0); // q
  compute_encoder.set_input_array(kernel_inputs[1], 1); // k_pool
  compute_encoder.set_input_array(kernel_inputs[2], 2); // v_pool
  compute_encoder.set_input_array(kernel_inputs[3], 3); // page_table
  compute_encoder.set_input_array(kernel_inputs[4], 4); // sequence_lengths
  compute_encoder.set_input_array(kernel_inputs[5], 5); // query_to_seq_map
  compute_encoder.set_input_array(kernel_inputs[6], 6); // query_token_offset

  // Upload the parameter struct to the GPU
  compute_encoder.set_bytes(&kernel_params, sizeof(PagedAttentionParams), 7);

  // Set the output array
  compute_encoder.set_output_array(kernel_out_array, 8);

  // Configure dispatch grid sizes
  MTL::Size threadgroups_per_grid = MTL::Size(items_to_process_count, 1, 1);
  MTL::Size threads_per_threadgroup = MTL::Size(threads_per_group_count, 1, 1);

  // Set the threadgroup memory length and dispatch the kernel
  compute_encoder.set_threadgroup_memory_length(total_tg_memory_bytes, 0);
  compute_encoder.dispatch_threadgroups(threadgroups_per_grid, threads_per_threadgroup);
}

void PagedAttentionPrimitive::eval_gpu(const std::vector<mx::array>& inputs,
                                       mx::array& out) {
  // Prepare Metal kernel and command encoder
  auto& s = stream();
  auto& d = mlx::core::metal::device(s.device);

  size_t bytes = out.nbytes();
  out.set_data(mx::allocator::malloc(bytes));
  auto& compute_encoder = d.get_command_encoder(s.index);

  // Create and populate PagedAttentionParams struct
  PagedAttentionParams params_struct;
  CoreDims core_dims = validate_inputs_and_populate_initial_params(inputs);

  // Early exit if no items to process
  if (core_dims.num_items_to_process == 0) {
      spdlog::info("[PAL Primitive] No items to process. Returning empty result or as per output_shapes.");
      return;
  }

  // Populate initial parts directly for head_dim check
  params_struct.head_dim = core_dims.head_dim;

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

  // Configure thread dimensions
  const size_t default_threads_per_item_group = 64;
  auto max_threads = kernel_state_->maxTotalThreadsPerThreadgroup();
  size_t threads_per_item_group = std::min(default_threads_per_item_group, max_threads);

  // Call helper to populate the remaining attention parameters
  ThreadgroupMemoryLayout memory_layout = prepare_inputs_layout_memory(
      params_struct,
      core_dims,
      inputs[1], // k_pool
      inputs[3], // page_table
      device_ptr,
      threads_per_item_group
  );

  // Adjust threadgroup size based on computed tile_size_T_runtime
  threads_per_item_group =
      std::min(static_cast<size_t>(params_struct.tile_size_T_runtime), max_threads);

  // Verify all pointers are valid before passing to Metal
  if (!inputs[0].data<void>() || !inputs[1].data<void>() || !inputs[2].data<void>() ||
      !inputs[3].data<void>() || !inputs[4].data<void>() || !inputs[5].data<void>() ||
      !inputs[6].data<void>()) {
    spdlog::error("[PAL Primitive] One or more input arrays have null data pointers");
    throw std::runtime_error("Null input data pointers detected in paged attention primitive");
  }

  spdlog::debug("[PAL Primitive Dispatch] Dispatching kernel with tile_size_T_runtime: {}", params_struct.tile_size_T_runtime);

  // Call the dispatch_metal_kernel helper with all needed parameters
  dispatch_metal_kernel(
      compute_encoder,
      kernel_state_,
      inputs,
      out,
      params_struct,
      memory_layout.total_bytes,
      core_dims.num_items_to_process,
      threads_per_item_group
  );
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

void PagedAttentionPrimitive::print(std::ostream& os) {
  os << "PagedAttentionPrimitive(num_q_heads=" << num_q_heads_ << ", num_kv_heads=" << num_kv_heads_ << ", head_dim=" << head_dim_ << ", tokens_per_page=" << tokens_per_page_ << ")";
}

}  // namespace pal::cpp
