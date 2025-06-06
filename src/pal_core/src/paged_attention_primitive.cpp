// paged_attention_primitive.cpp
// Implementation of the PagedAttentionPrimitive class that provides GPU-accelerated
// paged attention operations for transformer models.
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

#include "pal_core/paged_attention_primitive.hpp"
#include "pal_core/metal/dispatch.hpp"
#include "kernels/paged_attention_types.h"

// Standard library
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

// MLX and Metal includes
#include <mlx/allocator.h>
#include <mlx/array.h>
#include <mlx/ops.h>
#include <mlx/backend/cpu/encoder.h>
#include "mlx/backend/gpu/copy.h"
#include <mlx/backend/metal/device.h>
#include <mlx/backend/metal/metal.h>
#include "pal_core/metal/metal_loader.hpp"

// Define half type for half precision (K, V tiles)
using half = short;

#include <mlx/utils.h>
namespace mx = mlx::core;

namespace pal::cpp {

static std::tuple<mx::array, size_t> create_work_items_buffer(
    const PagedAttentionParams& params,
    const mx::array& sequence_lengths
);

static mx::array create_query_starts_buffer(
    uint32_t num_sequences_in_batch,
    const mx::array& sequence_lengths_arr,
    mx::StreamOrDevice s
);

PagedAttentionPrimitive::PagedAttentionPrimitive(
    mx::StreamOrDevice stream_or_device,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int tokens_per_page,
    bool use_fused_kernel
) : mx::UnaryPrimitive(mx::to_stream(stream_or_device)),
      num_q_heads_(num_q_heads),
      num_kv_heads_(num_kv_heads),
      head_dim_(head_dim),
      tokens_per_page_(tokens_per_page),
      use_fused_kernel_(use_fused_kernel) { }

void PagedAttentionPrimitive::eval_cpu(const std::vector<mx::array>& inputs, mx::array& out) {
    std::ostringstream oss;
    this->print(oss);
    spdlog::error("[PAL Primitive] CPU evaluation is not supported for {}.", oss.str());
    throw std::runtime_error(
        "[PagedAttentionPrimitive] CPU evaluation is not supported for " + oss.str());
}

void PagedAttentionPrimitive::eval_gpu(const std::vector<mx::array>& inputs, mx::array& out) {
    // Allocate output memory
    size_t bytes = out.nbytes();
    out.set_data(mx::allocator::malloc(bytes));

    auto& s = stream();
    auto& d = mlx::core::metal::device(s.device);

    CoreDims core_dims = extract_dims(inputs);
    PagedAttentionParams params;

    params.num_q_heads = core_dims.num_q_heads;
    params.num_kv_heads = core_dims.num_kv_heads;
    params.head_dim = core_dims.head_dim;
    params.tokens_per_page = core_dims.tokens_per_page;
    params.query_token_count_total = core_dims.query_token_count;
    params.max_logical_blocks_per_seq = inputs[3].shape(1);
    params.num_sequences_in_batch = inputs[3].shape(0);
    params.num_physical_pages_in_pool = inputs[1].shape(0);

    params.log_exp_min_clamp = kLogFp16DenormMinVal;
    params.pass2_token_block_size = PASS2_TOKEN_BLOCK_SIZE;
    params.pass2_qhead_block_size = PASS2_QHEAD_BLOCK_SIZE;
    params.inv_sqrt_head_dim = 1.0f / std::sqrt(static_cast<float>(core_dims.head_dim));
    params.num_active_batch_logical_pages = 1;

    spdlog::debug("[PAL Primitive] num_items_to_process: {}, use_fused_kernel: {}", core_dims.num_items_to_process, use_fused_kernel_);

    // Dispatch to appropriate implementation
    if (use_fused_kernel_) {
        _eval_gpu_fused(s, d, inputs, out, core_dims, params);
    } else {
        _eval_gpu_2pass(s, d, inputs, out, core_dims, params);
    }
}

void PagedAttentionPrimitive::_eval_gpu_fused(
    const mlx::core::Stream& stream,
    mlx::core::metal::Device& device,
    const std::vector<mx::array>& inputs,
    mx::array& out,
    const CoreDims& core_dims,
    PagedAttentionParams& params
) {
    // Prepare Metal kernel and command encoder
    const std::string library_name = "pal";
    const std::string dtype_suffix = mx::type_to_name(inputs[0].dtype());
    const std::string kernel_name = "paged_attn_fused_kernel_" + dtype_suffix;
    auto kernel_state = device.get_kernel(kernel_name, library_name);
    if (!kernel_state) {
        throw std::runtime_error("[PAL Primitive] Failed to load kernel: " + kernel_name);
    }

    // Calculate thread configuration
    size_t desired_threads_per_tg = kernel_state->threadExecutionWidth() * FUSED_SIMD_GROUPS_PER_THREADGROUP;
    auto thread_config = metal::MetalDispatcher::calculate_optimal_threads(
        kernel_state,
        desired_threads_per_tg
    );

    // Calculate memory layout
    auto memory_layout = AttentionMemoryLayout::calculate_attention_memory_layout(
        params,
        thread_config.threads_per_group,
        thread_config.execution_width,
        false  // use_2pass_kernel
    );

    // Calculate dispatch grid for decode
    metal::DispatchGrid grid;
    grid.width = core_dims.num_items_to_process;
    grid.height = 1;
    grid.depth = 1;

    // Setup compute encoder
    auto& compute_encoder = device.get_command_encoder(stream.index);
    compute_encoder.set_compute_pipeline_state(kernel_state);

    // Set parameters and output
    compute_encoder.set_input_array(inputs[0], 0);
    compute_encoder.set_input_array(inputs[1], 1);
    compute_encoder.set_input_array(inputs[2], 2);
    compute_encoder.set_input_array(inputs[3], 3);
    compute_encoder.set_input_array(inputs[4], 4);
    compute_encoder.set_input_array(inputs[5], 5);
    compute_encoder.set_input_array(inputs[6], 6);
    compute_encoder.set_bytes(&params, sizeof(PagedAttentionParams), 7);
    compute_encoder.set_output_array(out, 8);

    // Dispatch kernel
    metal::MetalDispatcher::dispatch_kernel(
        compute_encoder, grid,
        thread_config.threads_per_group,
        memory_layout.total_bytes
    );
}

void PagedAttentionPrimitive::_eval_gpu_2pass(
    const mlx::core::Stream& stream,
    mlx::core::metal::Device& device,
    const std::vector<mx::array>& inputs,
    mx::array& out,
    const CoreDims& core_dims,
    PagedAttentionParams& params
) {
    auto* metal_device_ptr = device.mtl_device();
    const std::string library_name = "pal";

    // --- NEW: Dynamic kernel name generation ---
    const std::string dtype_suffix = mx::type_to_name(inputs[0].dtype());
    const std::string pass1_kernel_name = "paged_attn_pass1_kernel_" + dtype_suffix;
    const std::string pass2_kernel_name = "paged_attn_pass2_kernel_" + dtype_suffix;

    spdlog::debug("[PAL 2PASS] Dispatching to kernels: {} and {}", pass1_kernel_name, pass2_kernel_name);
    // --- End new section ---

    auto kernel_state = device.get_kernel(pass1_kernel_name, library_name);
    if (!kernel_state) {
        throw std::runtime_error("[PAL Primitive] Failed to load kernel: " + pass1_kernel_name);
    }

    auto [tile_size, final_threads_per_tg, actual_simd_width] = get_optimal_tile_size_and_thread_info(
        core_dims.head_dim,
        core_dims.num_q_heads,
        core_dims.num_kv_heads,
        stream,
        kernel_state
    );

    spdlog::debug("[PAL Prefill] Final params: tokens_per_page={}, num_q_heads={}, num_kv_heads={}, head_dim={}",
                  params.tokens_per_page, params.num_q_heads, params.num_kv_heads, params.head_dim);

    // Calculate memory layout
    auto memory_layout = AttentionMemoryLayout::calculate_attention_memory_layout(
        params,
        final_threads_per_tg,
        actual_simd_width,
        true  // use_2pass_kernel
    );
    spdlog::debug("[PAL 2PASS Debug] TGMem Assertion Check: Calculated layout total_bytes = {}", memory_layout.total_bytes);
    if (memory_layout.total_bytes > metal_device_ptr->maxThreadgroupMemoryLength()) {
        throw std::runtime_error("[PAL Primitive] Prefill TGMem budget EXCEEDED with tile_size: " + std::to_string(tile_size));
    }

    // Create work items buffer and update params BEFORE allocating arrays
    auto [work_items_buffer, num_active_batch_logical_pages_val] = create_work_items_buffer(params, inputs[4]);
    params.num_active_batch_logical_pages = num_active_batch_logical_pages_val;
    spdlog::debug("[PAL 2PASS Debug] params.num_active_batch_logical_pages SET to: {}", params.num_active_batch_logical_pages);

    mx::array query_starts_buffer = create_query_starts_buffer(
        params.num_sequences_in_batch,
        inputs[4],
        stream
    );
    spdlog::debug("[PAL 2PASS Debug] query_starts_buffer created. Shape: [{}]", query_starts_buffer.shape(0));

    // Before allocating intermediate arrays
    spdlog::debug("[PAL 2PASS Debug] Allocating intermediate arrays with: query_token_count_total={}, num_q_heads={}, num_active_batch_logical_pages={}",
                  params.query_token_count_total, params.num_q_heads, params.num_active_batch_logical_pages);

    // intermediate arrays for Pass 1 outputs
    // Shape: [TotalQueryTokensInBatch, NumQHeads, NumActiveBatchLogicalPages]
    mx::Shape m_s_shape = {
        static_cast<int32_t>(params.query_token_count_total),
        static_cast<int32_t>(params.num_q_heads),
        static_cast<int32_t>(params.num_active_batch_logical_pages)
    };
    // Shape: [TotalQueryTokensInBatch, NumQHeads, NumActiveBatchLogicalPages, HeadDim]
    mx::Shape o_shape = {
        static_cast<int32_t>(params.query_token_count_total),
        static_cast<int32_t>(params.num_q_heads),
        static_cast<int32_t>(params.num_active_batch_logical_pages),
        static_cast<int32_t>(params.head_dim)
    };

    // Calculate dispatch grid for prefill Pass 1
    metal::DispatchGrid grid;
    grid.width = params.num_active_batch_logical_pages;
    grid.height = params.num_kv_heads;
    grid.depth = 1;

    // Before Pass 1 dispatch
    spdlog::debug("[PAL 2PASS Debug] Pass 1 Grid: width={}, height={}, depth={}",
                  grid.width, grid.height, grid.depth);
    spdlog::debug("[PAL 2PASS Debug] Passing params to Pass 1 with num_active_batch_logical_pages = {}",
                  params.num_active_batch_logical_pages);

    // Setup compute encoder
    auto& compute_encoder = device.get_command_encoder(stream.index);
    compute_encoder.set_compute_pipeline_state(kernel_state);
    // Set active work items buffer
    compute_encoder.set_input_array(inputs[0], 0);
    compute_encoder.set_input_array(inputs[1], 1);
    compute_encoder.set_input_array(inputs[2], 2);
    compute_encoder.set_input_array(inputs[3], 3);
    compute_encoder.set_input_array(inputs[4], 4);
    // we don't need query_to_seq_map_in for 2 pass, so skip it
    // we then map inputs[6] -> buffer(5)
    compute_encoder.set_input_array(inputs[6], 5);
    compute_encoder.set_bytes(&params, sizeof(PagedAttentionParams), 6);
    compute_encoder.set_input_array(work_items_buffer, 7);
    compute_encoder.set_input_array(query_starts_buffer, 8);

    // set intermediate outputs for prefill Pass 1
    mx::array m_locals_pass1_out = mx::array(m_s_shape, mx::float32, nullptr, {});
    m_locals_pass1_out.set_data(mx::allocator::malloc(m_locals_pass1_out.nbytes()));
    mx::array s_locals_pass1_out = mx::array(m_s_shape, mx::float32, nullptr, {});
    s_locals_pass1_out.set_data(mx::allocator::malloc(s_locals_pass1_out.nbytes()));
    mx::array o_partials_pass1_out = mx::array(o_shape, mx::float16, nullptr, {});
    o_partials_pass1_out.set_data(mx::allocator::malloc(o_partials_pass1_out.nbytes()));

    // Initialize intermediate arrays
    // costs ~10-20ms of cpu time. <- yuck, i cringe everytime. need to fix this.
    std::fill_n(m_locals_pass1_out.data<float>(), m_locals_pass1_out.size(), -std::numeric_limits<float>::infinity());
    std::fill_n(s_locals_pass1_out.data<float>(), s_locals_pass1_out.size(), 0.0f);

    device.add_temporary(m_locals_pass1_out, stream.index);
    device.add_temporary(s_locals_pass1_out, stream.index);
    device.add_temporary(o_partials_pass1_out, stream.index);

    compute_encoder.set_output_array(m_locals_pass1_out, 9);
    compute_encoder.set_output_array(s_locals_pass1_out, 10);
    compute_encoder.set_output_array(o_partials_pass1_out, 11);

    // Dispatch kernel
    metal::MetalDispatcher::dispatch_kernel(
        compute_encoder, grid,
        final_threads_per_tg,
        memory_layout.total_bytes
    );

    kernel_state = device.get_kernel(pass2_kernel_name, library_name);
    if (!kernel_state) {
        throw std::runtime_error("[PAL Primitive] Failed to load Pass 2 kernel: " + pass2_kernel_name);
    }

    compute_encoder.set_compute_pipeline_state(kernel_state);

    size_t pass2_grid_width = (params.query_token_count_total + params.pass2_token_block_size - 1) / params.pass2_token_block_size;
    size_t pass2_grid_height = (params.num_q_heads + params.pass2_qhead_block_size - 1) / params.pass2_qhead_block_size;

    spdlog::debug("[PAL Primitive] Pass 2 grid width: {}, height: {}", pass2_grid_width, pass2_grid_height);
    spdlog::debug("[PAL Primitive] Pass 2's query token count: {}, num q heads: {}", core_dims.query_token_count, core_dims.num_q_heads);

    metal::DispatchGrid pass2_grid;
    pass2_grid.width = pass2_grid_width;
    pass2_grid.height = pass2_grid_height;
    pass2_grid.depth = 1;

    // Calculate thread configuration
    size_t desired_threads_per_tg = kernel_state->threadExecutionWidth() * PASS2_SIMD_GROUPS_PER_THREADGROUP;
    auto thread_config = metal::MetalDispatcher::calculate_optimal_threads(kernel_state, desired_threads_per_tg);
    size_t num_simd_groups_pass2 = thread_config.threads_per_group / kernel_state->threadExecutionWidth();

    // 2. Calculate Pass 2 Threadgroup Memory (Reflects cooperative processing of ONE item)
    size_t pass2_tg_mem_bytes = 0;
    uintptr_t current_offset_for_calc = 0; // For careful carving simulation

    // Component 1: M_item_shared_scalar (float)
    current_offset_for_calc += sizeof(float);
    current_offset_for_calc = AttentionMemoryLayout::align_size(current_offset_for_calc); // Align after each logical block

    // Component 2: S_item_shared_scalar (float)
    current_offset_for_calc += sizeof(float);
    current_offset_for_calc = AttentionMemoryLayout::align_size(current_offset_for_calc);

    // Component 3: S_item_kahan_c_shared (float)
    current_offset_for_calc += sizeof(float);
    current_offset_for_calc = AttentionMemoryLayout::align_size(current_offset_for_calc);

    // Component 4: O_item_shared_accumulator[params.head_dim] (float array)
    current_offset_for_calc += params.head_dim * sizeof(float);
    current_offset_for_calc = AttentionMemoryLayout::align_size(current_offset_for_calc);

    // Component 5: Scratch for M-Reduction (simdgroup_m_scratch[NumSIMDgroups_Pass2])
    current_offset_for_calc += num_simd_groups_pass2 * sizeof(float);
    current_offset_for_calc = AttentionMemoryLayout::align_size(current_offset_for_calc);

    // Component 6: Scratch for S-Reduction (simdgroup_s_scratch[NumSIMDgroups_Pass2])
    current_offset_for_calc += num_simd_groups_pass2 * sizeof(float);
    current_offset_for_calc = AttentionMemoryLayout::align_size(current_offset_for_calc);

    // Component 7: Scratch for O-Reduction (simdgroup_o_partials[NumSIMDgroups_Pass2][params.head_dim])
    current_offset_for_calc += num_simd_groups_pass2 * params.head_dim * sizeof(float);
    current_offset_for_calc = AttentionMemoryLayout::align_size(current_offset_for_calc);

    // Component 8: Final Guard Padding (optional but good practice)
    current_offset_for_calc += kFinalMemoryPaddingGuardBytes;
    pass2_tg_mem_bytes = AttentionMemoryLayout::align_size(current_offset_for_calc);

    // Set intermediate input arrays from Pass 1
    spdlog::debug("[dispatch_prefill_pass2] m_locals_in shape: [{}, {}, {}]",
                  m_locals_pass1_out.shape(0), m_locals_pass1_out.shape(1), m_locals_pass1_out.shape(2));
    spdlog::debug("[dispatch_prefill_pass2] s_locals_in shape: [{}, {}, {}]",
                  s_locals_pass1_out.shape(0), s_locals_pass1_out.shape(1), s_locals_pass1_out.shape(2));
    spdlog::debug("[dispatch_prefill_pass2] o_partials_in shape: [{}, {}, {}, {}]",
                  o_partials_pass1_out.shape(0), o_partials_pass1_out.shape(1), o_partials_pass1_out.shape(2), o_partials_pass1_out.shape(3));
    spdlog::debug("[dispatch_prefill_pass2] work_items_buffer shape: [{}, {}]",
                  work_items_buffer.shape(0), work_items_buffer.shape(1));

    // Set input arrays
    compute_encoder.set_input_array(m_locals_pass1_out, 0);
    compute_encoder.set_input_array(s_locals_pass1_out, 1);
    compute_encoder.set_input_array(o_partials_pass1_out, 2);
    compute_encoder.set_bytes(&params, sizeof(PagedAttentionParams), 3);
    compute_encoder.set_output_array(out, 4);

    metal::MetalDispatcher::dispatch_kernel(
        compute_encoder, pass2_grid,
        thread_config.threads_per_group,
        pass2_tg_mem_bytes
    );
}

std::vector<mx::array> PagedAttentionPrimitive::vjp(
    const std::vector<mx::array>& primals,
    const std::vector<mx::array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<mx::array>& outputs) {
    throw std::runtime_error("[PagedAttentionPrimitive] VJP not implemented.");
}

std::vector<mx::array> PagedAttentionPrimitive::jvp(
    const std::vector<mx::array>& primals,
    const std::vector<mx::array>& tangents,
    const std::vector<int>& argnums) {
    throw std::runtime_error("[PagedAttentionPrimitive] JVP not implemented.");
}

std::pair<std::vector<mx::array>, std::vector<int>>
PagedAttentionPrimitive::vmap(const std::vector<mx::array>& inputs,
                              const std::vector<int>& axes) {
    throw std::runtime_error("[PagedAttentionPrimitive] Vmap not implemented.");
}

std::vector<mx::Shape> PagedAttentionPrimitive::output_shapes(
    const std::vector<mx::array>& inputs) {
    if (inputs.empty()) {
        throw std::invalid_argument(
            "[PagedAttentionPrimitive::output_shapes] Requires at least one input (query).");
    }
    if (inputs.size() < 2 || inputs[1].ndim() != 4) {
        throw std::invalid_argument(
            "[PagedAttentionPrimitive::output_shapes] K-pool (inputs[1]) is needed "
            "and must be 4D to determine head_dim for output shape.");
    }

    const auto& q = inputs[0];
    uint32_t current_head_dim = inputs[1].shape(3);

    if (q.ndim() == 3) {
        return {{q.shape(0) * q.shape(1), static_cast<int>(current_head_dim)}};
    } else if (q.ndim() == 2) {
        return {{q.shape(0), static_cast<int>(current_head_dim)}};
    } else if (q.ndim() == 1) {
        return {{q.shape(0), static_cast<int>(current_head_dim)}};
    } else {
        throw std::invalid_argument(
            "[PagedAttentionPrimitive::output_shapes] Query input 'q' must be 1D, 2D, or 3D.");
    }
}

void PagedAttentionPrimitive::print(std::ostream& os) {
    os << "PagedAttentionPrimitive(num_q_heads=" << num_q_heads_
       << ", num_kv_heads=" << num_kv_heads_
       << ", head_dim=" << head_dim_
       << ", tokens_per_page=" << tokens_per_page_
       << ")";
}

size_t PagedAttentionPrimitive::calculate_per_gqa_group_compute_scratch(
    uint32_t head_dimension,
    uint32_t number_of_simd_groups,
    uint32_t threads_per_group
) {
    size_t per_gqa_group_compute_scratch = 0;

    // Component 1: Potential TGMem for V-sum accumulators (one per active SIMD group in the GQA group)
    per_gqa_group_compute_scratch += number_of_simd_groups * head_dimension * sizeof(float);
    per_gqa_group_compute_scratch = AttentionMemoryLayout::align_size(per_gqa_group_compute_scratch);

    // Component 2: Potential TGMem for M/L stats (one pair per active SIMD group)
    per_gqa_group_compute_scratch += number_of_simd_groups * 2 * sizeof(float);
    per_gqa_group_compute_scratch = AttentionMemoryLayout::align_size(per_gqa_group_compute_scratch);

    // Component 3: General TG-wide reduction scratch
    per_gqa_group_compute_scratch += threads_per_group * sizeof(float);
    per_gqa_group_compute_scratch = AttentionMemoryLayout::align_size(per_gqa_group_compute_scratch);

    // Component 4: Small constant safety/alignment buffer
    per_gqa_group_compute_scratch += kFinalMemoryPaddingGuardBytes;
    per_gqa_group_compute_scratch = AttentionMemoryLayout::align_size(per_gqa_group_compute_scratch);

    spdlog::debug("[PAL 2PASS Debug] Calculated per_gqa_group_compute_scratch = {} bytes for D_s calculation", per_gqa_group_compute_scratch);

    return per_gqa_group_compute_scratch;
}

uint32_t PagedAttentionPrimitive::calculate_symmetric_tile_depth(
    uint32_t head_dimension,
    uint32_t num_query_heads,
    uint32_t num_kv_heads,
    size_t max_threadgroup_memory_bytes,
    size_t per_gqa_group_compute_scratch_bytes
) {
    constexpr size_t S_h = sizeof(half);  // 2 bytes for half precision (K, V tiles)
    constexpr size_t S_f = sizeof(float); // 4 bytes for single precision (Q tile, accumulators)

    if (num_kv_heads == 0) { // Avoid division by zero
        spdlog::error("[calculate_symmetric_tile_depth] num_kv_heads is zero.");
        return 8; // Default to a minimum safe depth
    }
    // Calculate GQA factor: number of query heads per key-value head group
    uint32_t query_heads_per_kv_group = std::max(1u, num_query_heads / num_kv_heads);

    // Fixed memory overheads not part of the main Q, K, V tiles.
    // This includes accumulators and stats for one GQA group's worth of Q-heads
    // actively computing against a K/V tile, plus other general scratch.
    size_t fixed_overhead_bytes = per_gqa_group_compute_scratch_bytes;
    fixed_overhead_bytes = AttentionMemoryLayout::align_size(fixed_overhead_bytes);

    spdlog::debug("[calculate_symmetric_tile_depth] Max TGMem: {} bytes, Fixed Overhead: {} bytes",
                  max_threadgroup_memory_bytes, fixed_overhead_bytes);

    if (fixed_overhead_bytes >= max_threadgroup_memory_bytes) {
        spdlog::error("[calculate_symmetric_tile_depth] Fixed overhead ({}) exceeds device TGMem limit ({}).",
                      fixed_overhead_bytes, max_threadgroup_memory_bytes);
        return 4; // Return minimum depth
    }

    size_t memory_for_qkv_tiles = max_threadgroup_memory_bytes - fixed_overhead_bytes;
    spdlog::debug("[calculate_symmetric_tile_depth] Memory available for QKV tiles: {} bytes", memory_for_qkv_tiles);

    // Bytes needed for one "layer" (depth=1) of symmetric Q, K, V tiles:
    // K-tile layer: head_dimension * sizeof(half)
    // V-tile layer: head_dimension * sizeof(half)
    // Q-block layer: query_heads_per_kv_group * head_dimension * sizeof(float)
    size_t bytes_per_unit_depth = head_dimension * (2 * S_h + query_heads_per_kv_group * S_f);
    spdlog::debug("[calculate_symmetric_tile_depth] Bytes per unit depth for QKV tiles: {} bytes", bytes_per_unit_depth);

    if (bytes_per_unit_depth == 0) {
        spdlog::error("[calculate_symmetric_tile_depth] Denominator (bytes_per_unit_depth) is zero.");
        return 4; // Return minimum depth
    }

    uint32_t unaligned_depth = static_cast<uint32_t>(memory_for_qkv_tiles / bytes_per_unit_depth);
    spdlog::debug("[calculate_symmetric_tile_depth] Unaligned symmetric depth (D_s): {}", unaligned_depth);
    uint32_t symmetric_depth = std::max(4u, (unaligned_depth / 4) * 4); // Align down to multiple of 4
    spdlog::debug("[calculate_symmetric_tile_depth] Final symmetric depth (D_s): {}", symmetric_depth);
    return std::min(MAX_TILE_SIZE_PRACTICAL, symmetric_depth);
}

std::tuple<uint32_t, uint32_t, uint32_t> PagedAttentionPrimitive::get_optimal_tile_size_and_thread_info(
    uint32_t head_dimension,
    uint32_t num_query_heads,
    uint32_t num_kv_heads,
    mx::StreamOrDevice stream_or_device,
    std::optional<MTL::ComputePipelineState*> pipeline_state
  ) {
    // Prepare Metal kernel and command encoder
    auto s = mx::to_stream(stream_or_device);
    auto& d = mlx::core::metal::device(s.device);
    auto* metal_device_ptr = d.mtl_device();

    MTL::ComputePipelineState* kernel_state = nullptr;
    if (!pipeline_state) {
        pal::cpp::MetalLibRegistrar::ensure_pal_metallib_registered(stream_or_device);
        const std::string library_name = "pal";
        const std::string kernel_name = "get_device_info";
        kernel_state = d.get_kernel(kernel_name, library_name);
        if (!kernel_state) {
            throw std::runtime_error("[PAL Primitive] Failed to load kernel: " + kernel_name);
        }
    } else {
        kernel_state = pipeline_state.value();
    }

    // Calculate thread configuration for prefill based on model parameters
    uint32_t actual_simd_width = kernel_state->threadExecutionWidth();

    // Calculate N_q_per_kv (GQA factor)
    uint32_t N_q_per_kv = 1;
    if (num_kv_heads > 0) { // GQA or MHA
        N_q_per_kv = std::max(1u, num_query_heads / num_kv_heads);
    }

    uint32_t desired_threads_per_tg = actual_simd_width * N_q_per_kv * PASS1_SIMD_GROUPS_PER_GQA_GROUP;
    uint32_t max_threads_device = kernel_state->maxTotalThreadsPerThreadgroup();
    uint32_t final_threads_per_tg = std::min(desired_threads_per_tg, max_threads_device);
    final_threads_per_tg = ((final_threads_per_tg + actual_simd_width - 1) / actual_simd_width) * actual_simd_width;
    spdlog::debug("[PAL Debug] final_threads_per_tg = {}", final_threads_per_tg);

    uint32_t total_simd_groups_in_tg = final_threads_per_tg / actual_simd_width;
    spdlog::debug("[PAL Debug] total_simd_groups_in_tg = {}", total_simd_groups_in_tg);

    size_t per_gqa_group_compute_scratch = calculate_per_gqa_group_compute_scratch(
        head_dimension,
        total_simd_groups_in_tg,
        final_threads_per_tg
    );

    // Calculate D_s using the new symmetric tile depth function
    uint32_t D_s = calculate_symmetric_tile_depth(
        head_dimension,
        num_query_heads,
        num_kv_heads,
        metal_device_ptr->maxThreadgroupMemoryLength(),
        per_gqa_group_compute_scratch
    );

    spdlog::debug("[PAL Debug] D_s = {}", D_s);

    return std::make_tuple(D_s, final_threads_per_tg, actual_simd_width);
  }


AttentionMemoryLayout AttentionMemoryLayout::calculate_attention_memory_layout(
    const PagedAttentionParams& params,
    size_t threads_per_group,
    size_t actual_simd_lanes_per_group,
    bool use_2pass_kernel
) {
    uint32_t query_heads_per_kv_group = std::max(1u, params.num_q_heads / params.num_kv_heads);
    const uint32_t num_simd_groups_in_tg = (threads_per_group + actual_simd_lanes_per_group - 1) / actual_simd_lanes_per_group;
    size_t partial_reduce_scratch = threads_per_group * sizeof(float);
    size_t simd_reduced_maxes_scratch = num_simd_groups_in_tg * sizeof(float);
    size_t simd_reduced_sum_exps_scratch = num_simd_groups_in_tg * sizeof(float);
    size_t global_stats_scratch = num_simd_groups_in_tg * 2 * sizeof(float); // Max 2 floats per SIMD group for M/L
    size_t kahan_comp_scratch = num_simd_groups_in_tg * sizeof(float); // Kahan compensation per SIMD group
    size_t simd_v_sums_scratch = num_simd_groups_in_tg * params.head_dim * sizeof(float);

    AttentionMemoryLayout layout;
    uintptr_t current_offset_bytes = 0;

    if (use_2pass_kernel) {
        // --- 2 pass kernel with symmetric QKV tiling  ---

        // 1. Q_shmem_block: Stores D_s Q-vectors, each for query_heads_per_kv_group heads, as float.
        layout.q_shmem_bytes = params.tokens_per_page * query_heads_per_kv_group * params.head_dim * sizeof(float);
        current_offset_bytes = AttentionMemoryLayout::align_size(layout.q_shmem_bytes);

        // 2. K_tile: Stores D_s K-vectors as half.
        layout.k_tile_bytes = params.tokens_per_page * params.head_dim * sizeof(half);
        current_offset_bytes = AttentionMemoryLayout::align_size(current_offset_bytes + layout.k_tile_bytes);

        // 3. V_tile: Stores D_s V-vectors as half.
        layout.v_tile_bytes = params.tokens_per_page * params.head_dim * sizeof(half);
        current_offset_bytes = AttentionMemoryLayout::align_size(current_offset_bytes + layout.v_tile_bytes);

        // Component 1: V-sum accumulator space (matches Component 1 from _eval_gpu_2pass)
        // The kernel allocates V_Sum_Accumulators_Area for total_simd_groups_in_tg_metal
        current_offset_bytes = AttentionMemoryLayout::align_size(current_offset_bytes + simd_v_sums_scratch);

        // Component 2: M/L stats space (matches Component 2 from _eval_gpu_2pass)
        // Each SIMD group needs space for M and L stats
        current_offset_bytes = AttentionMemoryLayout::align_size(current_offset_bytes + global_stats_scratch);

        // Component 3: General reduction scratch (matches Component 3 from _eval_gpu_2pass)
        layout.partial_reduce_scratch_bytes = partial_reduce_scratch;
        current_offset_bytes = AttentionMemoryLayout::align_size(current_offset_bytes + layout.partial_reduce_scratch_bytes);

        // Zero out other decode-specific items
        layout.simd_reduced_maxes_bytes = 0;
        layout.simd_reduced_adjusted_sum_exps_bytes = 0;
        layout.global_stats_bytes = 0;
        layout.s_global_compensation_bytes = 0;
        layout.simd_v_chunk_sums_bytes = 0;
        layout.page_table_slice_bytes = 0;

    } else {
        // --- FUSED PATH ---

        // 1. Q_shmem: For one Q-vector, as float.
        layout.q_shmem_bytes = params.head_dim * sizeof(float);
        current_offset_bytes = AttentionMemoryLayout::align_size(layout.q_shmem_bytes);

        // 2. K_tile: Uses params.tokens_per_page for depth.
        layout.k_tile_bytes = params.tokens_per_page * params.head_dim * sizeof(half);
        current_offset_bytes = AttentionMemoryLayout::align_size(current_offset_bytes + layout.k_tile_bytes);

        // 3. V_tile: Uses params.tokens_per_page for depth.
        layout.v_tile_bytes = params.tokens_per_page * params.head_dim * sizeof(half);
        current_offset_bytes = AttentionMemoryLayout::align_size(current_offset_bytes + layout.v_tile_bytes);

        // 4. Fixed Scratch components for Decode (as originally designed for it)
        layout.partial_reduce_scratch_bytes = partial_reduce_scratch;
        current_offset_bytes = AttentionMemoryLayout::align_size(current_offset_bytes + layout.partial_reduce_scratch_bytes);

        layout.simd_reduced_maxes_bytes = simd_reduced_maxes_scratch;
        current_offset_bytes = AttentionMemoryLayout::align_size(current_offset_bytes + layout.simd_reduced_maxes_bytes);

        layout.simd_reduced_adjusted_sum_exps_bytes = simd_reduced_sum_exps_scratch;
        current_offset_bytes = AttentionMemoryLayout::align_size(current_offset_bytes + layout.simd_reduced_adjusted_sum_exps_bytes);

        layout.global_stats_bytes = 2 * sizeof(float);
        current_offset_bytes = AttentionMemoryLayout::align_size(current_offset_bytes + layout.global_stats_bytes);

        layout.s_global_compensation_bytes = 1 * sizeof(float);
        current_offset_bytes = AttentionMemoryLayout::align_size(current_offset_bytes + layout.s_global_compensation_bytes);

        layout.simd_v_chunk_sums_bytes = simd_v_sums_scratch;
        current_offset_bytes = AttentionMemoryLayout::align_size(current_offset_bytes + layout.simd_v_chunk_sums_bytes);

        layout.page_table_slice_bytes = params.max_logical_blocks_per_seq * sizeof(uint32_t);
        current_offset_bytes = AttentionMemoryLayout::align_size(current_offset_bytes + layout.page_table_slice_bytes);
    }

    // --- Final Guard and Total ---
    layout.final_guard_bytes = kFinalMemoryPaddingGuardBytes;
    current_offset_bytes = AttentionMemoryLayout::align_size(current_offset_bytes + layout.final_guard_bytes);
    layout.total_bytes = current_offset_bytes;

    spdlog::debug("[Memory Layout {}] Q_shmem: {}, K_tile: {}, V_tile: {}, ReduceScratch: {}, Total: {} bytes",
                  use_2pass_kernel ? "2pass" : "Fused",
                  layout.q_shmem_bytes, layout.k_tile_bytes, layout.v_tile_bytes,
                  layout.partial_reduce_scratch_bytes, layout.total_bytes);

    return layout;
}

static std::tuple<mx::array, size_t> create_work_items_buffer(const PagedAttentionParams& params, const mx::array& sequence_lengths) {
    std::vector<std::tuple<uint32_t, uint32_t>> active_work_items;
    // Get sequence lengths array data
    const int32_t* sequence_lengths_ptr = sequence_lengths.data<int32_t>();

    spdlog::debug("[create_work_items_buffer] num_sequences_in_batch: {}, tokens_per_page: {}",
                  params.num_sequences_in_batch, params.tokens_per_page);

    // Iterate through batch items and their logical pages
    for (uint32_t b_idx = 0; b_idx < params.num_sequences_in_batch; ++b_idx) {
        int32_t seq_len = sequence_lengths_ptr[b_idx];
        if (seq_len > 0) {
            // Calculate number of logical pages for this sequence
            uint32_t num_logical_pages = (seq_len + params.tokens_per_page - 1) / params.tokens_per_page;
            spdlog::debug("[create_work_items_buffer] Batch {}: seq_len={}, num_logical_pages={}",
                          b_idx, seq_len, num_logical_pages);
            // Add all logical pages for this batch item
            for (uint32_t lp_idx = 0; lp_idx < num_logical_pages; ++lp_idx) {
                active_work_items.push_back(std::make_tuple(b_idx, lp_idx));
            }
        } else {
            spdlog::debug("[create_work_items_buffer] Batch {}: seq_len={} (skipped)", b_idx, seq_len);
        }
    }

    // Create work_items_buffer mx::array
    std::vector<uint32_t> flat_work_items_data;
    flat_work_items_data.reserve(active_work_items.size() * 2);
    for (const auto& item : active_work_items) {
        flat_work_items_data.push_back(std::get<0>(item));
        flat_work_items_data.push_back(std::get<1>(item));
    }

    spdlog::debug("[create_work_items_buffer] Total active_work_items: {}", active_work_items.size());

    // Create work_items_buffer from the flat data
    mx::array work_items_buffer(
        flat_work_items_data.data(),
        {static_cast<int>(active_work_items.size()), 2},
        mx::uint32
    );

    spdlog::debug("[create_work_items_buffer] work_items_buffer created with shape [{}, {}]",
                  work_items_buffer.shape(0), work_items_buffer.shape(1));

    return std::make_tuple(work_items_buffer, active_work_items.size());
}

static mx::array create_query_starts_buffer(
    uint32_t num_sequences_in_batch,
    const mx::array& sequence_lengths_arr,
    mx::StreamOrDevice s
) {
    if (num_sequences_in_batch == 0) {
        return mx::array({}, mx::uint32);
    }

    std::vector<uint32_t> query_starts_cpu(num_sequences_in_batch);
    uint32_t current_offset = 0;
    const int32_t* sequence_lengths_ptr = sequence_lengths_arr.data<int32_t>();

    for (uint32_t b_idx = 0; b_idx < num_sequences_in_batch; ++b_idx) {
        query_starts_cpu[b_idx] = current_offset;
        if (b_idx < sequence_lengths_arr.size()) { // Safety check
             current_offset += static_cast<uint32_t>(std::max(0, sequence_lengths_ptr[b_idx]));
        }
    }

    return mx::array(query_starts_cpu.data(), {static_cast<int>(num_sequences_in_batch)}, mx::uint32);
}


CoreDims extract_dims(const std::vector<mx::array>& inputs) {
    // Extract early for CoreDims population
    CoreDims extracted_dims_;
    const auto& q = inputs[0];
    const auto& k_pool = inputs[1];
    extracted_dims_.tokens_per_page = k_pool.shape(1);
    extracted_dims_.num_kv_heads = k_pool.shape(2);
    extracted_dims_.head_dim = k_pool.shape(3);

    if (q.ndim() == 3) {
        extracted_dims_.num_q_heads = q.shape(1);
        extracted_dims_.query_token_count = q.shape(0); // Number of "token rows" or "item groups"
        extracted_dims_.num_items_to_process = extracted_dims_.query_token_count * extracted_dims_.num_q_heads; // Total Q-Head items
    } else if (q.ndim() == 2) {
        extracted_dims_.num_q_heads = 1; // Implicitly 1 for 2D queries
        extracted_dims_.query_token_count = q.shape(0); // Number of query items/tokens
        extracted_dims_.num_items_to_process = extracted_dims_.query_token_count;
    } else if (q.ndim() == 1) {
        extracted_dims_.num_q_heads = 1; // Implicitly 1
        extracted_dims_.query_token_count = q.shape(0); // Number of query items/tokens
        extracted_dims_.num_items_to_process = extracted_dims_.query_token_count;
    }

    return extracted_dims_;
}

}  // namespace pal::cpp
