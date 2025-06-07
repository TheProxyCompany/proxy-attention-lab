// fill_kv_pages_primitive.cpp
// Implementation of the FillKVPagesPrimitive class that provides GPU-accelerated
// KV cache page filling operations for transformer models.
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

#include "pal_core/fill_kv_pages_primitive.hpp"
#include "pal_core/metal/dispatch.hpp"
#include "pal_core/metal/metal_loader.hpp"
#include "kernels/paged_attention_types.h"

// Standard library
#include <cmath>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <spdlog/spdlog.h>

// MLX and Metal includes
#include <mlx/allocator.h>
#include <mlx/array.h>
#include <mlx/ops.h>
#include <mlx/backend/metal/device.h>
#include <mlx/backend/metal/metal.h>
#include <mlx/backend/metal/utils.h>
#include "mlx/backend/gpu/copy.h"
#include <mlx/utils.h>

namespace mx = mlx::core;

namespace pal::cpp {

FillKVPagesPrimitive::FillKVPagesPrimitive(
    mx::StreamOrDevice stream_or_device,
    int num_kv_heads,
    int head_dim,
    int tokens_per_page)
    : mx::Primitive(mx::to_stream(stream_or_device)),
      num_kv_heads_(num_kv_heads),
      head_dim_(head_dim),
      tokens_per_page_(tokens_per_page) {
    spdlog::debug("[FillKVPagesPrimitive] Constructor called with kv_heads={}, head_dim={}, tokens_per_page={}",
                  num_kv_heads_, head_dim_, tokens_per_page_);
}

void FillKVPagesPrimitive::eval_cpu(const std::vector<mx::array>& inputs, std::vector<mx::array>& outputs) {
    spdlog::debug("[FillKVPagesPrimitive] eval_cpu called");
    std::ostringstream oss;
    this->print(oss);
    spdlog::error("[FillKVPagesPrimitive] CPU evaluation is not supported for {}.", oss.str());
    throw std::runtime_error(
        "[FillKVPagesPrimitive] CPU evaluation is not supported for " + oss.str());
}

void FillKVPagesPrimitive::eval_gpu(const std::vector<mx::array>& inputs, std::vector<mx::array>& outputs) {
    spdlog::debug("[FillKVPagesPrimitive] eval_gpu called");

    // Validate inputs
    if (inputs.size() < 7) {
        throw std::invalid_argument(
            "[FillKVPagesPrimitive] Expected 7 inputs: new_keys, new_values, global_k_pool, "
            "global_v_pool, page_table, current_token_write_positions, query_to_seq_map");
    }

    // Extract input arrays
    const auto& new_keys = inputs[0];               // [num_new_tokens, num_kv_heads, head_dim]
    const auto& new_values = inputs[1];              // [num_new_tokens, num_kv_heads, head_dim]
    const auto& global_k_pool = inputs[2];          // [num_pages, tokens_per_page, num_kv_heads, head_dim]
    const auto& global_v_pool = inputs[3];         // [num_pages, tokens_per_page, num_kv_heads, head_dim]
    const auto& page_table = inputs[4];             // [num_sequences, max_logical_blocks]
    const auto& current_token_write_positions = inputs[5]; // [num_sequences]
    const auto& query_to_seq_map = inputs[6];       // [num_new_tokens]

    // Get stream and device
    auto& s = stream();
    auto& d = mlx::core::metal::device(s.device);

    // Ensure PAL Metal library is registered
    MetalLibRegistrar::ensure_pal_metallib_registered(s);

    // Populate FillKVPagesParams struct
    FillKVPagesParams params;
    params.num_kv_heads = num_kv_heads_; // From primitive member
    params.head_dim = head_dim_;         // From primitive member
    params.tokens_per_page = tokens_per_page_; // From primitive member
    params.total_new_tokens_to_write = new_keys.shape(0);
    params.kv_pairs_per_threadgroup = KVPairsPerThreadgroup;

    if (page_table.ndim() < 2) {
        throw std::invalid_argument("[FillKVPagesPrimitive] page_table must be at least 2D.");
    }
    params.page_table_max_logical_blocks = page_table.shape(1);

    spdlog::debug("[FillKVPagesPrimitive] Kernel Params: kv_heads={}, head_dim={}, tokens_per_page={}, total_new_tokens={}, pt_max_logical_blocks={}",
                  params.num_kv_heads, params.head_dim, params.tokens_per_page,
                  params.total_new_tokens_to_write, params.page_table_max_logical_blocks);

    const std::string dtype_identifier = mx::type_to_name(global_k_pool.dtype());
    // Get Metal kernel
    const std::string library_name = "pal";
    const std::string kernel_name = "fill_kv_pages_kernel_" + dtype_identifier;
    auto kernel_state = d.get_kernel(kernel_name, library_name);
    if (!kernel_state) {
        throw std::runtime_error("[FillKVPagesPrimitive] Failed to load kernel: " + kernel_name);
    }

    // Setup compute encoder
    auto& compute_encoder = d.get_command_encoder(s.index);
    compute_encoder.set_compute_pipeline_state(kernel_state);

    // Set kernel arguments
    compute_encoder.set_input_array(new_keys, 0);
    compute_encoder.set_input_array(new_values, 1);
    compute_encoder.set_input_array(page_table, 2);
    compute_encoder.set_input_array(current_token_write_positions, 3);
    compute_encoder.set_input_array(query_to_seq_map, 4);
    compute_encoder.set_bytes(&params, sizeof(FillKVPagesParams), 5);

    outputs[0].copy_shared_buffer(global_k_pool);
    outputs[1].copy_shared_buffer(global_v_pool);

    compute_encoder.set_output_array(outputs[0], 6);
    compute_encoder.set_output_array(outputs[1], 7);

    // Calculate dispatch grid
    metal::DispatchGrid grid;
    size_t num_threadgroups = (params.total_new_tokens_to_write + KVPairsPerThreadgroup - 1) / KVPairsPerThreadgroup;
    grid.width = num_threadgroups;
    grid.height = 1;
    grid.depth = 1;

    size_t threads_per_group = kernel_state->threadExecutionWidth() * SIMD_GROUPS_PER_THREADGROUP;
    threads_per_group = std::min(threads_per_group, kernel_state->maxTotalThreadsPerThreadgroup());

    // Dispatch kernel
    metal::MetalDispatcher::dispatch_kernel(
        compute_encoder,
        grid,
        threads_per_group,   // threads_per_group
        0                   // threadgroup_memory_bytes
    );
}

void FillKVPagesPrimitive::print(std::ostream& os) {
    spdlog::debug("[FillKVPagesPrimitive] print called");
    os << "FillKVPagesPrimitive(num_kv_heads=" << num_kv_heads_
       << ", head_dim=" << head_dim_
       << ", tokens_per_page=" << tokens_per_page_ << ")";
}

std::vector<mx::Shape> FillKVPagesPrimitive::output_shapes(const std::vector<mx::array>& inputs) {
    spdlog::debug("[FillKVPagesPrimitive] output_shapes called");

    // Validate input count
    if (inputs.size() < 7) {
        throw std::invalid_argument(
            "[FillKVPagesPrimitive] output_shapes: Expected 7 inputs (new_keys, new_values, "
            "global_k_pool, global_v_pool, page_table, current_token_write_positions, query_to_seq_map)");
    }

    // Validate global pool dimensions
    if (inputs[2].ndim() != 4 || inputs[3].ndim() != 4) {
        throw std::invalid_argument(
            "[FillKVPagesPrimitive] output_shapes: global_k_pool and global_v_pool must be 4D arrays");
    }

    // Return shapes of the global pools which are modified in-place
    // inputs[2] is global_key_pool, inputs[3] is global_value_pool
    return {inputs[2].shape(), inputs[3].shape()};
}

}  // namespace pal::cpp
