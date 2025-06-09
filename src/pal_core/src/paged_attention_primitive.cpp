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
    int tokens_per_page,
    bool use_two_pass,
    int chunk_size
) : mx::UnaryPrimitive(mx::to_stream(stream_or_device)),
      num_q_heads_(num_q_heads),
      num_kv_heads_(num_kv_heads),
      head_dim_(head_dim),
      tokens_per_page_(tokens_per_page),
      use_two_pass_(use_two_pass),
      chunk_size_(chunk_size) { }

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

    // 3. Extract input arrays
    const auto& queries = inputs[0];
    const auto& k_cache_pool = inputs[1];
    const auto& v_cache_pool = inputs[2];
    const auto& page_table = inputs[3];
    const auto& context_lens = inputs[4];

    // 4. Populate parameters from input shapes
    PagedAttentionParams params;
    params.num_q_heads = queries.shape(1);
    params.num_kv_heads = k_cache_pool.shape(2);
    params.num_physical_pages_in_pool = k_cache_pool.shape(0);
    params.tokens_per_page = k_cache_pool.shape(1);
    params.max_logical_pages_per_seq = page_table.shape(1);
    params.num_sequences_in_batch = queries.shape(0);
    params.inv_sqrt_head_dim = 1.0f / std::sqrt(static_cast<float>(head_dim_));
    params.simd_width = 32; // Will be updated by kernel state

    // 5. Get kernel and update SIMD width
    const std::string dtype_suffix = mx::type_to_name(queries.dtype());
    const std::string kernel_name = "pal_paged_attention_" + dtype_suffix + "_" + std::to_string(head_dim_);

    mx::metal::MTLFCList func_consts = {
      {&use_two_pass_, MTL::DataType::DataTypeBool, 0}
    };

    auto kernel_state = d.get_kernel(
        kernel_name,
        "pal",
        "", // hash
        func_consts
    );
    if (!kernel_state) {
        throw std::runtime_error("[PAL] Failed to load kernel: " + kernel_name);
    }
    params.simd_width = kernel_state->threadExecutionWidth();

    // 6. Calculate threadgroup memory requirements
    const size_t threads_per_group = 256;
    const size_t num_simd_groups = threads_per_group / params.simd_width;

    size_t tg_memory_bytes = calculate_attention_memory_layout(
        params,
        threads_per_group,
        params.simd_width,
        k_cache_pool.dtype(),
        head_dim_
    );

    if (tg_memory_bytes > d.mtl_device()->maxThreadgroupMemoryLength()) {
        throw std::runtime_error("[PAL] Calculated threadgroup memory exceeds device limits.");
    }

    // 7. Handle two-pass vs single-pass execution
    auto& compute_encoder = d.get_command_encoder(s.index);

    if (use_two_pass_) {
        // Calculate number of chunks for two-pass
        const int max_tokens = params.max_logical_pages_per_seq * params.tokens_per_page;
        const int num_chunks = (max_tokens + chunk_size_ - 1) / chunk_size_;

        // Allocate intermediate buffers for two-pass
        int num_sequences_in_batch = static_cast<int>(params.num_sequences_in_batch);
        int num_q_heads = static_cast<int>(params.num_q_heads);
        mx::array max_logits({num_sequences_in_batch, num_q_heads, num_chunks}, mx::float32, nullptr, {});
        mx::array exp_sums({num_sequences_in_batch, num_q_heads, num_chunks}, mx::float32, nullptr, {});
        mx::array tmp_out({num_sequences_in_batch, num_q_heads, num_chunks, head_dim_}, queries.dtype(), nullptr, {});

        max_logits.set_data(mx::allocator::malloc(max_logits.nbytes()));
        exp_sums.set_data(mx::allocator::malloc(exp_sums.nbytes()));
        tmp_out.set_data(mx::allocator::malloc(tmp_out.nbytes()));

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

        // Calculate reduce memory requirements
        size_t reduce_memory_bytes = calculate_reduce_memory_layout(
            params,
            num_chunks,
            threads_per_group,
            params.simd_width,
            head_dim_
        );
        compute_encoder.set_threadgroup_memory_length(reduce_memory_bytes, 0);

        metal::DispatchGrid dispatch_grid_pass2;
        dispatch_grid_pass2.width = params.num_sequences_in_batch;
        dispatch_grid_pass2.height = params.num_q_heads;
        dispatch_grid_pass2.depth = 1;

        metal::MetalDispatcher::dispatch_kernel(
            compute_encoder,
            dispatch_grid_pass2,
            threads_per_group,
            reduce_memory_bytes
        );

        // Add temporaries for cleanup
        d.add_temporaries({max_logits, exp_sums, tmp_out}, s.index);

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
    if (inputs.size() < 5) {
        throw std::invalid_argument(
            "[PagedAttentionPrimitive::output_shapes] Requires 5 inputs: "
            "queries, k_cache_pool, v_cache_pool, page_table, context_lens.");
    }

    const auto& queries = inputs[0];

    // Output shape is [num_seqs, num_q_heads, head_dim]
    if (queries.ndim() == 3) {
        return {{queries.shape(0), queries.shape(1), head_dim_}};
    } else {
        throw std::invalid_argument(
            "[PagedAttentionPrimitive::output_shapes] Query input must be 3D [num_seqs, num_q_heads, head_dim].");
    }
}

void PagedAttentionPrimitive::print(std::ostream& os) {
    os << "PagedAttentionPrimitive(num_q_heads=" << num_q_heads_
       << ", num_kv_heads=" << num_kv_heads_
       << ", head_dim=" << head_dim_
       << ", tokens_per_page=" << tokens_per_page_
       << ", use_two_pass=" << use_two_pass_
       << ", chunk_size=" << chunk_size_
       << ")";
}

std::tuple<uint32_t, uint32_t, uint32_t> PagedAttentionPrimitive::get_optimal_tile_size_and_thread_info() {
    size_t tile_size = 64; // need to make this match the upstream invocations
    size_t threads_per_group = 256;
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
    size_t kv_item_size = mx::size_of(kv_cache_dtype);
    const size_t num_simd_groups = threads_per_group / simd_width;

    // Memory layout matching the kernel's threadgroup memory partitioning
    size_t q_tile_bytes = head_dim * sizeof(float);
    size_t k_tile_bytes = params.tokens_per_page * head_dim * kv_item_size;
    size_t v_tile_bytes = params.tokens_per_page * head_dim * kv_item_size;
    size_t page_table_bytes = params.max_logical_pages_per_seq * sizeof(uint32_t);
    size_t reduction_scratch_bytes = params.tokens_per_page * sizeof(float); // logits tile
    size_t simd_stats_bytes = num_simd_groups * sizeof(float) * 2; // max_scores + sum_exps
    size_t acc_tile_bytes = num_simd_groups * head_dim * sizeof(float);

    size_t total_bytes = q_tile_bytes + k_tile_bytes + v_tile_bytes + page_table_bytes +
                        reduction_scratch_bytes + simd_stats_bytes + acc_tile_bytes;

    // Align to 16-byte boundary
    return (total_bytes + 15) & ~15;
}

static size_t calculate_reduce_memory_layout(
    const PagedAttentionParams& params,
    int num_chunks,
    size_t threads_per_group,
    size_t simd_width,
    int head_dim
) {
    const size_t num_simd_groups = threads_per_group / simd_width;

    // Memory layout for reduce kernel
    size_t shared_max_logits_bytes = num_chunks * sizeof(float);
    size_t shared_exp_sums_bytes = num_chunks * sizeof(float);
    size_t red_smem_bytes = 2 * num_simd_groups * sizeof(float);

    size_t total_bytes = shared_max_logits_bytes + shared_exp_sums_bytes + red_smem_bytes;

    // Align to 16-byte boundary
    return (total_bytes + 15) & ~15;
}

}  // namespace pal::cpp
