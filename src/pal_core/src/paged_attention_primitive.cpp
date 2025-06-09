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

#include <mlx/utils.h>
namespace mx = mlx::core;

namespace pal::cpp {

PagedAttentionPrimitive::PagedAttentionPrimitive(
    mx::StreamOrDevice stream_or_device,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int tokens_per_page
) : mx::UnaryPrimitive(mx::to_stream(stream_or_device)),
      num_q_heads_(num_q_heads),
      num_kv_heads_(num_kv_heads),
      head_dim_(head_dim),
      tokens_per_page_(tokens_per_page) { }

void PagedAttentionPrimitive::eval_cpu(const std::vector<mx::array>& inputs, mx::array& out) {
    std::ostringstream oss;
    this->print(oss);
    spdlog::error("[PAL Primitive] CPU evaluation is not supported for {}.", oss.str());
    throw std::runtime_error(
        "[PagedAttentionPrimitive] CPU evaluation is not supported for " + oss.str());
}

void PagedAttentionPrimitive::eval_gpu(const std::vector<mx::array>& inputs, mx::array& out) {
    // 1. Allocate output buffer
    size_t bytes = out.nbytes();
    out.set_data(mx::allocator::malloc(bytes));

    // 2. Get stream and device, ensure PAL library is registered
    auto& s = stream();
    auto& d = mlx::core::metal::device(s.device);
    MetalLibRegistrar::ensure_pal_metallib_registered(s);

    // 3. Populate parameters directly from input shapes
    PagedAttentionParams params;
    // inputs[0] is query
    params.num_q_heads = inputs[0].shape(1);
    // inputs[1] is k-pool
    params.num_physical_pages_in_pool = inputs[1].shape(0);
    params.tokens_per_page = inputs[1].shape(1);
    params.num_kv_heads = inputs[1].shape(2);
    // inputs[3] is page table
    params.max_logical_pages_per_seq = inputs[3].shape(1);
    params.num_sequences_in_batch = inputs[3].shape(0);
    // parameters that are not directly from input shapes
    params.log_exp_min_clamp = -88.0f; // tune
    params.inv_sqrt_head_dim = 1.0f / std::sqrt(static_cast<float>(head_dim_));
    params.simd_width = 32; // placeholder, will be set by the kernel state

    // 4. Define dispatch grid
    metal::DispatchGrid dispatch_grid;
    dispatch_grid.width = params.num_sequences_in_batch;
    dispatch_grid.height = params.num_q_heads;
    dispatch_grid.depth = 1;

    // 5. Get the unified kernel
    const std::string dtype_suffix = mx::type_to_name(inputs[0].dtype());
    const std::string kernel_name = "pal_paged_attention_" + dtype_suffix + "_" + std::to_string(head_dim_);
    auto kernel_state = d.get_kernel(kernel_name, "pal");
    if (!kernel_state) {
        throw std::runtime_error("[PAL] Failed to load kernel: " + kernel_name);
    }

    // 6. Determine threadgroup configuration
    params.simd_width = kernel_state->threadExecutionWidth();
    const size_t threads_per_group = 256; // 1024 is max on apple silicon
    const size_t tg_memory_bytes = calculate_attention_memory_layout(
        params,
        threads_per_group,
        params.simd_width,
        inputs[1].dtype(),
        head_dim_
    );

    if (tg_memory_bytes > d.mtl_device()->maxThreadgroupMemoryLength()) {
        throw std::runtime_error("[PAL] Calculated threadgroup memory exceeds device limits.");
    }

    // 7. Set up the encoder and dispatch
    auto& compute_encoder = d.get_command_encoder(s.index);
    compute_encoder.set_compute_pipeline_state(kernel_state);

    // Set buffers from existing inputs. The order MUST match the new kernel's signature.
    compute_encoder.set_input_array(inputs[0], 0); // queries
    compute_encoder.set_input_array(inputs[1], 1); // k_cache_pool
    compute_encoder.set_input_array(inputs[2], 2); // v_cache_pool
    compute_encoder.set_input_array(inputs[3], 3); // page_table
    compute_encoder.set_input_array(inputs[4], 4); // context_lens (not sequence_lengths!)
    compute_encoder.set_output_array(out, 5);       // output at index 5!
    // Skip 6,7,8 for now (two-pass buffers)
    compute_encoder.set_bytes(&params, sizeof(PagedAttentionParams), 9); // params at index 9!

    metal::MetalDispatcher::dispatch_kernel(
        compute_encoder,
        dispatch_grid,
        threads_per_group,
        tg_memory_bytes
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

std::tuple<uint32_t, uint32_t, uint32_t> PagedAttentionPrimitive::get_optimal_tile_size_and_thread_info() {
    size_t tile_size = 64;
    size_t threads_per_group = 256; // 1024 is the max on apple silicon
    uint32_t actual_simd_width = 32;

    return std::make_tuple(tile_size, threads_per_group, actual_simd_width);
  }


static size_t calculate_attention_memory_layout(
    const PagedAttentionParams& params,
    size_t threads_per_group,
    size_t simd_width,
    mx::Dtype kv_cache_dtype,
    int head_dim
) {
    // This handles float16, bfloat16, float32, etc.
    size_t kv_item_size = mx::size_of(kv_cache_dtype);

    const size_t num_simd_groups = threads_per_group / simd_width;
    size_t q_tile_bytes = head_dim * sizeof(float);
    size_t k_tile_bytes = params.tokens_per_page * head_dim * kv_item_size;
    size_t v_tile_bytes = params.tokens_per_page * head_dim * kv_item_size;
    size_t page_table_bytes = params.max_logical_pages_per_seq * sizeof(uint32_t);
    size_t reduction_scratch_bytes = params.tokens_per_page * sizeof(float); // logits tile
    size_t simd_stats_bytes = num_simd_groups * sizeof(float) * 2; // max_scores + sum_exps
    size_t acc_tile_bytes = num_simd_groups * head_dim * sizeof(float);

    size_t total_bytes = q_tile_bytes + k_tile_bytes + v_tile_bytes + page_table_bytes + reduction_scratch_bytes + simd_stats_bytes + acc_tile_bytes;

    // Align to a 16-byte boundary
    return (total_bytes + 15) & ~15;
}


}  // namespace pal::cpp
