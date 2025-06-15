// paged_prefill_primitive.cpp
// Implementation of the PagedPrefillPrimitive class for tiled, high-performance
// prefill operations in paged-attention transformer models.
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

#include "pal_core/paged_attention_prefill_primitive.hpp"
#include "pal_core/metal/dispatch.hpp"
#include "pal_core/metal/metal_loader.hpp"
#include "kernels/paged_attention_types.h"

#include <cmath>
#include <stdexcept>
#include <string>
#include <spdlog/spdlog.h>
#include <mlx/backend/metal/device.h>
#include <mlx/backend/metal/utils.h>
#include <mlx/allocator.h>

namespace mx = mlx::core;

namespace pal::cpp {

// --- Forward declarations for internal helpers ---
namespace {

int calculate_q_tile_size(int head_dim, size_t max_threadgroup_mem, mx::Dtype dtype);
size_t calculate_prefill_threadgroup_memory(int head_dim, int q_tile_size, int simd_width, mx::Dtype dtype);

} // namespace

PagedAttentionPrefillPrimitive::PagedAttentionPrefillPrimitive(
    mx::StreamOrDevice stream_or_device,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int tokens_per_page
) : mx::UnaryPrimitive(mx::to_stream(stream_or_device)),
      num_q_heads_(num_q_heads),
      num_kv_heads_(num_kv_heads),
      head_dim_(head_dim),
      tokens_per_page_(tokens_per_page) {
    spdlog::debug("[PagedPrefillPrimitive] Constructed with head_dim={}, num_q_heads={}", head_dim_, num_q_heads_);
}

void PagedAttentionPrefillPrimitive::eval_cpu(const std::vector<mx::array>& inputs, mx::array& out) {
    throw std::runtime_error("[PagedPrefillPrimitive] CPU evaluation not implemented");
}

void PagedAttentionPrefillPrimitive::eval_gpu(const std::vector<mx::array>& inputs, mx::array& out) {
    // 1. --- Input Validation and Context Setup ---
    if (inputs.size() < 7) {
        throw std::invalid_argument("[PagedPrefillPrimitive] Expected 7 inputs: Q_prompt, K_prompt, V_prompt, K_cache, V_cache, page_table, context_len");
    }

    auto& s = stream();
    auto& d = mx::metal::device(s.device);
    MetalLibRegistrar::ensure_pal_metallib_registered(s);

    const auto& q_prompt = inputs[0];
    const auto& k_prompt = inputs[1];
    const auto& v_prompt = inputs[2];
    const auto& k_cache_paged = inputs[3];
    const auto& v_cache_paged = inputs[4];
    const auto& page_table = inputs[5];
    const auto& context_len_arr = inputs[6];

    // 2. --- Dynamic Tile Size Calculation ---
    const size_t simd_width = 32;
    const size_t max_threadgroup_memory = d.mtl_device()->maxThreadgroupMemoryLength();
    const int q_tile_size = calculate_q_tile_size(head_dim_, max_threadgroup_memory, q_prompt.dtype());
    if (q_tile_size == 0) {
        throw std::runtime_error("Calculated Q_TILE_SIZE is 0. Check head_dim and memory budget.");
    }
    spdlog::info("[PagedPrefillPrimitive] Using calculated Q_TILE_SIZE = {}", q_tile_size);
    // Each SIMD Group processes one query from a tile.
    const size_t threads_per_group = q_tile_size * simd_width;

    // 3. --- Kernel Selection ---
    const std::string attention_kernel_name = "pal_paged_attention_prefill_";
    const std::string dtype_suffix = mx::type_to_name(q_prompt.dtype());
    const std::string kernel_name = attention_kernel_name + dtype_suffix
                                    + "_" + std::to_string(head_dim_)
                                    + "_" + std::to_string(q_tile_size)
                                    + "_" + std::to_string(simd_width);

    auto kernel_state = d.get_kernel(kernel_name, "pal");
    if (!kernel_state) {
        throw std::runtime_error("[PagedPrefillPrimitive] Failed to load required Metal kernel: " + kernel_name);
    }

    // 4. --- Parameter and Grid Setup ---
    // Allocate the output buffer on the device.
    out.set_data(mx::allocator::malloc(out.nbytes()));

    const int num_prompt_tokens = q_prompt.shape(0);
    const int num_sequences = page_table.shape(0);

    // Populate the parameter struct to pass to the kernel.
    PagedAttentionParams params;
    params.num_q_heads = num_q_heads_;
    params.num_kv_heads = num_kv_heads_;
    params.tokens_per_page = tokens_per_page_;
    params.num_physical_pages_in_pool = v_cache_paged.shape(0);
    params.max_logical_pages_per_seq = page_table.shape(1);
    params.num_sequences_in_batch = num_sequences;
    params.num_prompt_tokens = num_prompt_tokens;
    params.inv_sqrt_head_dim = 1.0f / std::sqrt(static_cast<float>(head_dim_));

    // Calculate grid dimensions based on tiling strategy.
    metal::DispatchGrid grid;
    grid.width = (num_prompt_tokens + q_tile_size - 1) / q_tile_size; // num_q_blocks
    grid.height = num_sequences;
    grid.depth = num_q_heads_;

    // 5. --- Dispatch Kernel ---
    size_t tg_memory_bytes = calculate_prefill_threadgroup_memory(
        head_dim_,
        q_tile_size,
        simd_width,
        q_prompt.dtype()
    );
    if (tg_memory_bytes > d.mtl_device()->maxThreadgroupMemoryLength()) {
        throw std::runtime_error("[PagedPrefillPrimitive] Calculated threadgroup memory (" + std::to_string(tg_memory_bytes) + ") exceeds device limits.");
    }

    auto& compute_encoder = d.get_command_encoder(s.index);
    compute_encoder.set_compute_pipeline_state(kernel_state);

    // Set all kernel arguments
    compute_encoder.set_input_array(q_prompt, 0);
    compute_encoder.set_input_array(k_prompt, 1);
    compute_encoder.set_input_array(v_prompt, 2);
    compute_encoder.set_input_array(k_cache_paged, 3);
    compute_encoder.set_input_array(v_cache_paged, 4);
    compute_encoder.set_input_array(page_table, 5);
    compute_encoder.set_input_array(context_len_arr, 6);
    compute_encoder.set_bytes(&params, sizeof(PagedAttentionParams), 7);
    compute_encoder.set_output_array(out, 8);
    compute_encoder.set_threadgroup_memory_length(tg_memory_bytes, 0);

    metal::MetalDispatcher::dispatch_kernel(
        compute_encoder,
        grid,
        threads_per_group,
        tg_memory_bytes
    );
}

void PagedAttentionPrefillPrimitive::print(std::ostream& os) {
    os << "PagedPrefillPrimitive(num_q_heads=" << num_q_heads_
       << ", num_kv_heads=" << num_kv_heads_
       << ", head_dim=" << head_dim_
       << ", tokens_per_page=" << tokens_per_page_
       << ")";
}

bool PagedAttentionPrefillPrimitive::is_equivalent(const mx::Primitive& other) const {
    if (typeid(*this) != typeid(other)) {
        return false;
    }
    const auto& other_prim = static_cast<const PagedAttentionPrefillPrimitive&>(other);
    return num_q_heads_ == other_prim.num_q_heads_ &&
           num_kv_heads_ == other_prim.num_kv_heads_ &&
           head_dim_ == other_prim.head_dim_ &&
           tokens_per_page_ == other_prim.tokens_per_page_;
}

std::vector<mx::Shape> PagedAttentionPrefillPrimitive::output_shapes(const std::vector<mx::array>& inputs) {
    // The output shape is determined by the shape of the input queries.
    const auto& q_prompt = inputs[0];
    return {{q_prompt.shape(0) * q_prompt.shape(1), head_dim_}};
}

// --- Internal Helper Implementations ---
namespace {

int calculate_q_tile_size(int head_dim, size_t max_threadgroup_mem, mx::Dtype dtype) {
    const float budget = max_threadgroup_mem * MEMORY_BUDGET_FRACTION;
    const float cost_per_query =
        (head_dim * mx::size_of(dtype)) +   // Q_tile
        (head_dim * sizeof(float)) +        // Output accumulator (must be float32)
        (2 * sizeof(float));                // Softmax stats (max_score, sum_exp)

    if (cost_per_query == 0) return 0;

    // Use std::bit_floor (C++20) to round down to the nearest power of two.
    uint32_t budget_per_query = static_cast<uint32_t>(budget / cost_per_query);
    uint32_t rounded_down_power_of_two = std::bit_floor(budget_per_query);
    return std::min(32, static_cast<int>(rounded_down_power_of_two));
}

size_t calculate_prefill_threadgroup_memory(
    int head_dim,
    int q_tile_size,
    int simd_width,
    mx::Dtype dtype
) {
    // Memory for Q_tile
    size_t q_tile_mem = q_tile_size * head_dim * mx::size_of(dtype);
    // Memory for Output Accumulator (float32)
    size_t out_acc_mem = q_tile_size * head_dim * sizeof(float);
    // Memory for Softmax stats (2 floats per query)
    size_t stats_mem = q_tile_size * (2 * sizeof(float));
    // Memory for the reduction scratchpads (one pad per SIMD group, one float per thread)
    size_t reduction_mem =  q_tile_size * simd_width * sizeof(float);

    return q_tile_mem + out_acc_mem + stats_mem + reduction_mem;
}

} // namespace

} // namespace pal::cpp
