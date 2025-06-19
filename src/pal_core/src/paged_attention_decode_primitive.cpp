// paged_attention_decode_primitive.cpp
// Implementation of the PagedAttentionDecodePrimitive class that provides GPU-accelerated
// paged attention decode operations for transformer models.
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

#include "pal_core/paged_attention_decode_primitive.hpp"
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

PagedAttentionDecodePrimitive::PagedAttentionDecodePrimitive(
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

void PagedAttentionDecodePrimitive::eval_cpu(const std::vector<mx::array>& inputs, mx::array& out) {
    std::ostringstream oss;
    this->print(oss);
    spdlog::error("[PAL Primitive] CPU evaluation is not supported for {}.", oss.str());
    throw std::runtime_error(
        "[PagedAttentionPrimitive] CPU evaluation is not supported for " + oss.str());
}

void PagedAttentionDecodePrimitive::eval_gpu(const std::vector<mx::array>& inputs, mx::array& out) {
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
    params.num_physical_pages_in_pool = v_cache_pool.shape(0);
    params.num_kv_heads = v_cache_pool.shape(1);
    params.tokens_per_page = v_cache_pool.shape(3);
    params.max_logical_pages_per_seq = page_table.shape(1);
    params.num_sequences_in_batch = page_table.shape(0);
    params.num_prompt_tokens = 1; // decode only supports 1 prompt token
    // parameters that are not directly from input shapes
    params.log_exp_min_clamp = -88.0f; // tune
    params.inv_sqrt_head_dim = 1.0f / std::sqrt(static_cast<float>(head_dim_));

    const int max_tokens = params.max_logical_pages_per_seq * params.tokens_per_page;
    const int num_chunks = (max_tokens + CHUNK_SIZE - 1) / CHUNK_SIZE;
    const bool use_two_pass = max_tokens > CHUNK_SIZE;

    // need to get simd_width from device info
    // hardcode it to 32 for now
    const size_t simd_width = 32;

    // --- Dynamic thread count calculation ---
    // Calculate total threadgroups for occupancy consideration
    const int total_threadgroups = params.num_sequences_in_batch * params.num_q_heads *
                                   (use_two_pass ? num_chunks : 1);

    // Pass 1: Scale threads based on occupancy
    size_t threads_per_group = (total_threadgroups > 64) ? 128 :
                               (total_threadgroups > 32) ? 256 : 512;
    // Ensure it's a multiple of tokens_per_page for efficient token processing
    threads_per_group = std::max(threads_per_group, (size_t)params.tokens_per_page);
    threads_per_group = ((threads_per_group + simd_width - 1) / simd_width) * simd_width;

    // 4. Define dispatch grid
    metal::DispatchGrid dispatch_grid;
    dispatch_grid.width = params.num_sequences_in_batch;
    dispatch_grid.height = params.num_q_heads;
    dispatch_grid.depth = use_two_pass ? num_chunks : 1;

    // 5. Get the pass 1 kernel
    const std::string dtype_suffix = mx::type_to_name(inputs[0].dtype());
    const std::string attention_kernel_name = "pal_paged_attention_decode_"
        + dtype_suffix + "_" + std::to_string(head_dim_)
        + "_" + std::to_string(params.tokens_per_page) + "_" + std::to_string(simd_width);

    const std::string kernel_hash = use_two_pass ? "two_pass" : "single_pass";
    auto kernel_state = d.get_kernel(
        attention_kernel_name,
        "pal",
        attention_kernel_name + "_" + kernel_hash,
        { {&use_two_pass, MTL::DataType::DataTypeBool, 0} }
    );
    if (!kernel_state) {
        throw std::runtime_error("[PAL] Failed to load kernel: " + attention_kernel_name);
    }

    // 6. Determine threadgroup configuration
    const size_t tg_memory_bytes = calculate_attention_memory_layout(
        params,
        threads_per_group,
        simd_width,
        inputs[1].dtype(),
        head_dim_
    );

    if (tg_memory_bytes > d.mtl_device()->maxThreadgroupMemoryLength()) {
        throw std::runtime_error("[PAL] Calculated threadgroup memory exceeds device limits: " + std::to_string(tg_memory_bytes) + " > " + std::to_string(d.mtl_device()->maxThreadgroupMemoryLength()));
    }

    if (params.num_q_heads % params.num_kv_heads != 0) {
        spdlog::error("[PAL] num_q_heads {} must be divisible by num_kv_heads {}", params.num_q_heads, params.num_kv_heads);
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
        const std::string reduce_kernel_name = "pal_paged_reduce_decode_" + dtype_suffix + "_"
                                                + std::to_string(head_dim_)
                                                + "_" + std::to_string(simd_width);
        kernel_state = d.get_kernel(reduce_kernel_name, "pal");
        if (!kernel_state) {
            throw std::runtime_error("[PAL] Failed to load reduce kernel: " + reduce_kernel_name);
        }

        const int reduce_threadgroups = params.num_sequences_in_batch * params.num_q_heads;
        const size_t threads_per_group_pass2 = (reduce_threadgroups > 64) ? 128 : 256;
        const size_t tg_memory_bytes_pass2 = calculate_reduce_memory_layout(
            params,
            threads_per_group_pass2,
            simd_width
        );

        compute_encoder.set_compute_pipeline_state(kernel_state);
        compute_encoder.set_output_array(out, 0);
        compute_encoder.set_input_array(max_logits, 1);
        compute_encoder.set_input_array(exp_sums, 2);
        compute_encoder.set_input_array(tmp_out, 3);
        compute_encoder.set_input_array(context_lens, 4);
        compute_encoder.set_bytes(&params, sizeof(PagedAttentionParams), 5);
        compute_encoder.set_threadgroup_memory_length(tg_memory_bytes_pass2, 0);

        metal::DispatchGrid dispatch_grid_pass2;
        dispatch_grid_pass2.width = params.num_sequences_in_batch;
        dispatch_grid_pass2.height = params.num_q_heads;
        dispatch_grid_pass2.depth = 1;

        metal::MetalDispatcher::dispatch_kernel(
            compute_encoder,
            dispatch_grid_pass2,
            threads_per_group_pass2,
            tg_memory_bytes_pass2
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

std::vector<mx::array> PagedAttentionDecodePrimitive::vjp(
    const std::vector<mx::array>& primals,
    const std::vector<mx::array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<mx::array>& outputs) {
    throw std::runtime_error("[PagedAttentionPrimitive] VJP not implemented.");
}

std::vector<mx::array> PagedAttentionDecodePrimitive::jvp(
    const std::vector<mx::array>& primals,
    const std::vector<mx::array>& tangents,
    const std::vector<int>& argnums) {
    throw std::runtime_error("[PagedAttentionPrimitive] JVP not implemented.");
}

std::pair<std::vector<mx::array>, std::vector<int>>
PagedAttentionDecodePrimitive::vmap(const std::vector<mx::array>& inputs,
                              const std::vector<int>& axes) {
    throw std::runtime_error("[PagedAttentionPrimitive] Vmap not implemented.");
}

std::vector<mx::Shape> PagedAttentionDecodePrimitive::output_shapes(
    const std::vector<mx::array>& inputs) {
    if (inputs.empty()) {
        throw std::invalid_argument(
            "[PagedAttentionPrimitive::output_shapes] Requires at least one input (query).");
    }
    if (inputs.size() < 2 || inputs[1].ndim() != 5) {
        throw std::invalid_argument(
            "[PagedAttentionPrimitive::output_shapes] K-pool (inputs[1]) is needed "
            "and must be 5D to determine head_dim for output shape. "
            "The last dimension must be " + std::to_string(MEMORY_ALIGNMENT_BYTES / mx::size_of(inputs[1].dtype())) + ".");
    }

    const auto& queries = inputs[0];
    if (queries.ndim() == 3) {
        return {{queries.shape(0) * queries.shape(1), static_cast<int>(head_dim_)}};
    } else if (queries.ndim() == 2) {
        return {{queries.shape(0), static_cast<int>(head_dim_)}};
    } else if (queries.ndim() == 1) {
        return {{queries.shape(0), static_cast<int>(head_dim_)}};
    } else {
        throw std::invalid_argument(
            "[PagedAttentionPrimitive::output_shapes] Query input 'queries' must be 1D, 2D, or 3D.");
    }
}

void PagedAttentionDecodePrimitive::print(std::ostream& os) {
    os << "PagedAttentionPrimitive(num_q_heads=" << num_q_heads_
       << ", num_kv_heads=" << num_kv_heads_
       << ", head_dim=" << head_dim_
       << ", tokens_per_page=" << tokens_per_page_
       << ")";
}

size_t PagedAttentionDecodePrimitive::calculate_attention_memory_layout(
    const PagedAttentionParams& params,
    size_t threads_per_group,
    size_t simd_width,
    mx::Dtype kv_cache_dtype,
    int head_dim
) {
    // This handles float16, bfloat16, float32, etc.
    size_t kv_item_size = mx::size_of(kv_cache_dtype);
    size_t num_simd_groups = threads_per_group / simd_width;

    // calculate q tile size
    // 1. Calculate SUBGROUP_SIZE
    const size_t subgroup_size = std::max((size_t)1, simd_width / params.tokens_per_page);
    // 2. Calculate QK_VECTOR_WIDTH
    const size_t qk_vector_width = std::max((size_t)1, MEMORY_ALIGNMENT_BYTES / (subgroup_size * kv_item_size));
    // 3. Calculate num_vecs_per_thread
    const size_t num_vecs_per_thread = head_dim / (subgroup_size * qk_vector_width);
    size_t q_tile_bytes = (qk_vector_width * sizeof(float)) * subgroup_size * num_vecs_per_thread;
    size_t logits_tile_bytes = CHUNK_SIZE * sizeof(float); // logits tile
    size_t reduction_scratchpad_bytes = num_simd_groups * sizeof(float);

    // use same 16-byte alignment as in paged_attention.h.metal
    size_t total_bytes = 0;
    total_bytes += q_tile_bytes;
    total_bytes = (total_bytes + 15) & ~15;
    total_bytes += logits_tile_bytes;
    total_bytes = (total_bytes + 15) & ~15;
    total_bytes += reduction_scratchpad_bytes;
    total_bytes = (total_bytes + 15) & ~15;
    // Aligned to a 16-byte boundary

    return total_bytes;
}

size_t PagedAttentionDecodePrimitive::calculate_reduce_memory_layout(
    const PagedAttentionParams& params,
    size_t threads_per_group,
    size_t simd_width
) {
    // Calculate maximum possible chunks
    const int max_tokens = params.max_logical_pages_per_seq * params.tokens_per_page;
    const int max_num_chunks = (max_tokens + CHUNK_SIZE - 1) / CHUNK_SIZE;

    // Reduce kernel needs:
    // 1. shared_max_logits: max_num_chunks * sizeof(float)
    // 2. shared_exp_sums: max_num_chunks * sizeof(float)
    // 3. reduction_scratch: 2 * num_simd_groups * sizeof(float)

    size_t num_simd_groups = threads_per_group / simd_width;

    size_t total_bytes = 0;
    total_bytes += max_num_chunks * sizeof(float);      // shared_max_logits
    total_bytes += max_num_chunks * sizeof(float);      // shared_exp_sums
    total_bytes += 2 * num_simd_groups * sizeof(float); // reduction_scratch

    // Align to 16-byte boundary
    total_bytes = (total_bytes + 15) & ~15;

    return total_bytes;
}

// v shape is 4D: [num_total_pages, num_kv_heads, head_dim, tokens_per_page]
mx::Shape PagedAttentionDecodePrimitive::get_v_cache_shape(
    int num_total_pages,
    int num_kv_heads,
    int head_dim,
    int tokens_per_page,
    mx::Dtype dtype
) {
    return mx::Shape{num_total_pages, num_kv_heads, head_dim, tokens_per_page};
}

// k shape is 5D: [num_total_pages, num_kv_heads, head_dim / elements_per_thread, tokens_per_page, elements_per_thread]
mx::Shape PagedAttentionDecodePrimitive::get_k_cache_shape(
    int num_total_pages,
    int num_kv_heads,
    int head_dim,
    int tokens_per_page,
    mx::Dtype dtype
) {
    // Align our K cache to a MEMORY_ALIGNMENT_BYTES byte boundary for coalesced device memory access
    //
    // credit: https://github.com/EricLBuehler/mistral.rs/blob/master/mistralrs-paged-attn/src/metal/kernels/pagedattention.metal
    // for inspiration and reference (theirs uses a constant x = 16 / sizeof(T))
    const int elements_per_thread = MEMORY_ALIGNMENT_BYTES / mx::size_of(dtype);
    return mx::Shape{num_total_pages, num_kv_heads, head_dim / elements_per_thread, tokens_per_page, elements_per_thread};
}

}  // namespace pal::cpp
