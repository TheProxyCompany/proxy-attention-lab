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

    const auto& queries = inputs[0];
    const auto& k_cache_pool = inputs[1];
    const auto& v_cache_pool = inputs[2];
    const auto& page_table = inputs[3];
    const auto& context_lens = inputs[4];

    // 3. Populate parameters directly from input shapes
    PagedAttentionParams params;
    params.num_q_heads = num_q_heads_;
    params.num_physical_pages_in_pool = k_cache_pool.shape(0);
    params.num_kv_heads = k_cache_pool.shape(1);
    params.tokens_per_page = k_cache_pool.shape(2);
    params.max_logical_pages_per_seq = page_table.shape(1);
    params.num_sequences_in_batch = page_table.shape(0);
    // parameters that are not directly from input shapes
    params.log_exp_min_clamp = -88.0f; // tune
    params.inv_sqrt_head_dim = 1.0f / std::sqrt(static_cast<float>(head_dim_));
    params.simd_width = 32; // placeholder, will be set by the kernel state

    const int max_tokens = params.max_logical_pages_per_seq * params.tokens_per_page;
    const int num_chunks = (max_tokens + CHUNK_SIZE - 1) / CHUNK_SIZE;
    const bool use_two_pass = max_tokens > CHUNK_SIZE;

    // 4. Define dispatch grid
    metal::DispatchGrid dispatch_grid;
    dispatch_grid.width = params.num_sequences_in_batch;
    dispatch_grid.height = params.num_q_heads;
    dispatch_grid.depth = use_two_pass ? num_chunks : 1;

    // 5. Get the unified kernel
    const std::string dtype_suffix = mx::type_to_name(inputs[0].dtype());
    const std::string kernel_name = "pal_paged_attention_" + dtype_suffix + "_" + std::to_string(head_dim_);

    auto kernel_state = d.get_kernel(
        kernel_name,
        "pal",
        "", // hash
        { {&use_two_pass, MTL::DataType::DataTypeBool, 0} }
    );
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
        throw std::runtime_error("[PAL] Calculated threadgroup memory exceeds device limits: " + std::to_string(tg_memory_bytes) + " > " + std::to_string(d.mtl_device()->maxThreadgroupMemoryLength()));
    }

    if (params.num_q_heads % params.num_kv_heads != 0) {
        throw std::runtime_error("num_q_heads must be divisible by num_kv_heads");
    }

    // 7. Set up the encoder and dispatch
    auto& compute_encoder = d.get_command_encoder(s.index);
    if (use_two_pass) {
        int num_seq = static_cast<int>(params.num_sequences_in_batch);

        // Allocate intermediate buffers for two-pass
        mx::array max_logits({num_seq, num_q_heads_, num_chunks}, mx::float32, nullptr, {});
        mx::array exp_sums({num_seq, num_q_heads_, num_chunks}, mx::float32, nullptr, {});
        mx::array tmp_out({num_seq, num_q_heads_, num_chunks, head_dim_}, queries.dtype(), nullptr, {});

        max_logits.set_data(mx::allocator::malloc(max_logits.nbytes()));
        exp_sums.set_data(mx::allocator::malloc(exp_sums.nbytes()));
        tmp_out.set_data(mx::allocator::malloc(tmp_out.nbytes()));

        d.add_temporaries({max_logits, exp_sums, tmp_out}, s.index);

        // Pass 1: Compute attention per chunk
        compute_encoder.set_compute_pipeline_state(kernel_state);
        compute_encoder.set_input_array(queries, 0);
        compute_encoder.set_input_array(k_cache_pool, 1);
        compute_encoder.set_input_array(v_cache_pool, 2);
        compute_encoder.set_input_array(page_table, 3);
        compute_encoder.set_input_array(context_lens, 4);
        compute_encoder.set_output_array(out, 5); // Not used in pass 1, but needed for signature
        compute_encoder.set_output_array(max_logits, 6);
        compute_encoder.set_output_array(exp_sums, 7);
        compute_encoder.set_output_array(tmp_out, 8);
        compute_encoder.set_bytes(&params, sizeof(PagedAttentionParams), 9);
        compute_encoder.set_threadgroup_memory_length(tg_memory_bytes, 0);

        // Dispatch with chunks in z-dimension
        metal::DispatchGrid dispatch_grid_pass1;
        dispatch_grid_pass1.width = params.num_sequences_in_batch;
        dispatch_grid_pass1.height = params.num_q_heads;
        dispatch_grid_pass1.depth = num_chunks;

        metal::MetalDispatcher::dispatch_kernel(
            compute_encoder,
            dispatch_grid_pass1,
            threads_per_group,
            tg_memory_bytes
        );

        // Pass 2: Reduce across chunks
        const std::string reduce_kernel_name = "pal_paged_reduce_" + dtype_suffix + "_" + std::to_string(head_dim_);
        auto reduce_kernel_state = d.get_kernel(reduce_kernel_name, "pal");
        if (!reduce_kernel_state) {
            throw std::runtime_error("[PAL] Failed to load reduce kernel: " + reduce_kernel_name);
        }

        compute_encoder.set_compute_pipeline_state(reduce_kernel_state);
        compute_encoder.set_output_array(out, 0);
        compute_encoder.set_input_array(max_logits, 1);
        compute_encoder.set_input_array(exp_sums, 2);
        compute_encoder.set_input_array(tmp_out, 3);
        compute_encoder.set_input_array(context_lens, 4);
        compute_encoder.set_bytes(&params, sizeof(PagedAttentionParams), 5);

        compute_encoder.set_threadgroup_memory_length(tg_memory_bytes, 0);

        metal::DispatchGrid dispatch_grid_pass2;
        dispatch_grid_pass2.width = params.num_sequences_in_batch;
        dispatch_grid_pass2.height = params.num_q_heads;
        dispatch_grid_pass2.depth = 1;

        metal::MetalDispatcher::dispatch_kernel(
            compute_encoder,
            dispatch_grid_pass2,
            threads_per_group,
            tg_memory_bytes
        );
    } else {
        // Single-pass execution
        compute_encoder.set_compute_pipeline_state(kernel_state);
        compute_encoder.set_input_array(queries, 0);
        compute_encoder.set_input_array(k_cache_pool, 1);
        compute_encoder.set_input_array(v_cache_pool, 2);
        compute_encoder.set_input_array(page_table, 3);
        compute_encoder.set_input_array(context_lens, 4);
        compute_encoder.set_output_array(out, 5);
        // Skip buffers 6, 7, 8 (not used in single-pass)
        compute_encoder.set_bytes(&params, sizeof(PagedAttentionParams), 9);
        compute_encoder.set_threadgroup_memory_length(tg_memory_bytes, 0);

        metal::DispatchGrid dispatch_grid;
        dispatch_grid.width = params.num_sequences_in_batch;
        dispatch_grid.height = params.num_q_heads;
        dispatch_grid.depth = 1;

        metal::MetalDispatcher::dispatch_kernel(
            compute_encoder,
            dispatch_grid,
            threads_per_group,
            tg_memory_bytes
        );
    }
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

// todo make this dynamic
size_t PagedAttentionPrimitive::get_optimal_page_size() {
    size_t page_size = 16;

    return page_size;
  }


size_t PagedAttentionPrimitive::calculate_attention_memory_layout(
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
    size_t logits_tile_bytes = params.tokens_per_page * sizeof(float); // logits tile
    size_t simd_stats_bytes = num_simd_groups * sizeof(float) * 2; // max_scores + sum_exps
    size_t reduction_scratchpad_bytes = num_simd_groups * sizeof(float);
    size_t acc_tile_bytes = num_simd_groups * head_dim * sizeof(float);

    size_t total_bytes = q_tile_bytes + k_tile_bytes + v_tile_bytes + logits_tile_bytes + simd_stats_bytes + reduction_scratchpad_bytes + acc_tile_bytes;

    // Align to a 16-byte boundary
    return (total_bytes + 15) & ~15;
}


}  // namespace pal::cpp
