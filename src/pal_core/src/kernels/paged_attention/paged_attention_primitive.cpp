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

// PAL utilities
#include "pal_core/types.hpp"
#include "pal_core/kernel_constants.hpp"
#include "pal_core/kernel_utils/memory_layout.hpp"
#include "pal_core/kernel_utils/validation.hpp"
#include "pal_core/kernel_utils/tiling.hpp"
#include "pal_core/metal/dispatch.hpp"
#include "pal_core/debug/kernel_debug.hpp"
#include "pal_core/param_builder.hpp"

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

// Forward declare the calculate_attention_memory_layout function
namespace kernel_utils {
    AttentionMemoryLayout calculate_attention_memory_layout(
        const PagedAttentionParams& params,
        size_t threads_per_group,
        size_t actual_simd_lanes_per_group
    );
}

// Paged attention specific validator
class PagedAttentionValidator : public kernel_utils::KernelValidator {
public:
    CoreDims validate_and_extract_dims(const std::vector<mx::array>& inputs) {
        validate(inputs);
        return extracted_dims_;
    }

    void validate(const std::vector<mx::array>& inputs) override {
        if (inputs.size() != 7) {
            throw std::invalid_argument(
                format_error("PAL Primitive", "Expected 7 inputs, received " + std::to_string(inputs.size())));
        }

        const auto& q = inputs[0];
        const auto& k_pool = inputs[1];
        const auto& page_table = inputs[3];
        const auto& sequence_lengths = inputs[4];
        const auto& query_to_seq_map = inputs[5];
        const auto& query_token_offset = inputs[6];

        // Use common validation utilities
        kernel_utils::ValidationUtils::check_dtype(page_table, mx::uint32, "page_table");
        kernel_utils::ValidationUtils::check_dtype(sequence_lengths, mx::int32, "sequence_lengths");
        kernel_utils::ValidationUtils::check_dtype(query_to_seq_map, mx::int32, "query_to_seq_map");
        kernel_utils::ValidationUtils::check_dtype(query_token_offset, mx::int32, "query_token_offset");

        // K/V Pool validation
        kernel_utils::ValidationUtils::check_ndim(k_pool, 4, "k_pool");

        // Extract dimensions
        extracted_dims_.tokens_per_page = k_pool.shape(1);
        extracted_dims_.num_kv_heads = k_pool.shape(2);
        extracted_dims_.head_dim = k_pool.shape(3);

        // Check vectorization requirement
        kernel_utils::ValidationUtils::check_divisibility(
            extracted_dims_.head_dim,
            kernels::paged_attention::HEAD_DIM_VECTORIZATION,
            "head_dim"
        );

        // Query validation
        validate_query_format(q);

        // Page table validation
        kernel_utils::ValidationUtils::check_ndim(page_table, 2, "page_table");

        // Sideband array validation
        validate_sideband_arrays(query_to_seq_map, query_token_offset);

        spdlog::debug("[PAL Primitive Validate] Input validation complete.");
    }

private:
    CoreDims extracted_dims_;

    void validate_query_format(const mx::array& q) {
        if (q.ndim() == 3) {
            if (q.shape(2) != static_cast<int>(extracted_dims_.head_dim)) {
                throw std::invalid_argument(format_error("PAL Primitive",
                    "3D Query HeadDim mismatch with K-pool head_dim."));
            }
            extracted_dims_.num_q_heads = q.shape(1);
            extracted_dims_.num_items_to_process = q.shape(0) * q.shape(1);
            extracted_dims_.query_token_count = q.shape(0);
        } else if (q.ndim() == 2) {
            if (q.shape(1) != static_cast<int>(extracted_dims_.head_dim)) {
                throw std::invalid_argument(format_error("PAL Primitive",
                    "2D Query HeadDim mismatch with K-pool head_dim."));
            }
            extracted_dims_.num_q_heads = 1;
            extracted_dims_.num_items_to_process = q.shape(0);
            extracted_dims_.query_token_count = q.shape(0);
        } else if (q.ndim() == 1) {
            if (extracted_dims_.head_dim != 1) {
                throw std::invalid_argument(format_error("PAL Primitive",
                    "1D Query requires K-pool head_dim to be 1."));
            }
            extracted_dims_.num_q_heads = 1;
            extracted_dims_.num_items_to_process = q.shape(0);
            extracted_dims_.query_token_count = q.shape(0);
        } else {
            throw std::invalid_argument(format_error("PAL Primitive",
                "Query 'q' ndim (" + std::to_string(q.ndim()) + ") not supported."));
        }

        if (extracted_dims_.num_items_to_process == 0 && q.size() > 0) {
            throw std::invalid_argument(format_error("PAL Primitive",
                "num_items_to_process is 0 but query array is not empty."));
        }
    }

    void validate_sideband_arrays(const mx::array& query_to_seq_map,
                                 const mx::array& query_token_offset) {
        if (query_to_seq_map.size() != extracted_dims_.query_token_count) {
            throw std::invalid_argument(format_error("PAL Primitive",
                "query_to_seq_map size mismatch"));
        }
        if (query_token_offset.size() != extracted_dims_.query_token_count) {
            throw std::invalid_argument(format_error("PAL Primitive",
                "query_token_offset size mismatch"));
        }
    }
};

// Parameter builder for paged attention
class PagedAttentionParamBuilder : public KernelParamBuilder<PagedAttentionParams> {
public:
    PagedAttentionParamBuilder(const CoreDims& dims,
                              const mx::array& k_pool_arr,
                              const mx::array& page_table_arr,
                              bool is_prefill)
        : dims_(dims), k_pool_arr_(k_pool_arr),
          page_table_arr_(page_table_arr), is_prefill_(is_prefill) {}

    PagedAttentionParams build(MTL::Device* device) override {
        PagedAttentionParams params;

        // Basic parameters
        params.num_q_heads = dims_.num_q_heads;
        params.num_kv_heads = dims_.num_kv_heads;
        params.head_dim = dims_.head_dim;
        params.tokens_per_page = dims_.tokens_per_page;
        params.log_exp_min_clamp = kernels::paged_attention::kLogFp16DenormMinVal;

        // Array-derived parameters
        params.max_logical_blocks_per_seq = page_table_arr_.shape(1);
        params.num_sequences_in_batch = page_table_arr_.shape(0);
        params.num_physical_pages_in_pool = k_pool_arr_.shape(0);

        // Calculate inv_sqrt_head_dim
        params.inv_sqrt_head_dim = 1.0f / std::sqrt(static_cast<float>(params.head_dim));

        // Pre-calculate strides
        params.per_token_stride_in_cache = params.num_kv_heads * params.head_dim;
        params.per_page_stride_in_cache = params.tokens_per_page * params.per_token_stride_in_cache;

        // Calculate tile size
        calculate_tile_size(params, device);

        return params;
    }

private:
    const CoreDims& dims_;
    const mx::array& k_pool_arr_;
    const mx::array& page_table_arr_;
    bool is_prefill_;

    void calculate_tile_size(PagedAttentionParams& params, MTL::Device* device) {
        // Calculate fixed memory usage with zero tile size
        PagedAttentionParams params_for_sizing = params;
        params_for_sizing.tile_size_T_runtime = 0;

        auto fixed_layout = kernel_utils::calculate_attention_memory_layout(
            params_for_sizing,
            kernels::DEFAULT_THREADS_PER_GROUP,
            kernels::SIMD_WIDTH
        );

        // Get memory constraints
        auto constraints = get_memory_constraints(device, fixed_layout.total_bytes);

        // Calculate bytes per token for tiling
        size_t bytes_per_token = calculate_bytes_per_token(params);

        // Use tiling utility to calculate optimal tile size
        tiling::TileRequirements requirements{
            .bytes_per_element = bytes_per_token,
            .alignment = kernels::paged_attention::TILE_SIZE_ALIGNMENT,
            .min_size = kernels::paged_attention::MIN_TILE_SIZE_SOFT,
            .max_size = std::min(kernels::paged_attention::MAX_TILE_SIZE_PRACTICAL,
                               static_cast<uint32_t>(params.tokens_per_page)),
            .preferred_size = (params.head_dim <= 256) ? 64u :
                            kernels::paged_attention::MIN_TILE_SIZE_SOFT
        };

        auto tile_config = tiling::calculate_optimal_tile_size(
            constraints.available_memory(), requirements);

        if (!tile_config.is_valid()) {
            throw std::runtime_error(
                "[PAL Primitive] head_dim/tokens_per_page combination leaves no "
                "scratch space for a single KV token.");
        }

        params.tile_size_T_runtime = tile_config.tile_size;

        debug::KernelDebugger::log_tile_config("PAL Primitive",
                                              params.tile_size_T_runtime, "T");
    }

    size_t calculate_bytes_per_token(const PagedAttentionParams& params) {
        uint32_t padded_head_dim = params.head_dim;

        if (is_prefill_) {
            // For prefill Pass 1: Calculate unique KV heads for a Q-head block
            uint32_t q_heads_per_kv_head = params.num_q_heads / params.num_kv_heads;
            uint32_t max_unique_kv_heads_per_block = std::min(
                static_cast<uint32_t>(kernels::paged_attention::PREFILL_PASS1_Q_HEAD_BLOCK_SIZE),
                (kernels::paged_attention::PREFILL_PASS1_Q_HEAD_BLOCK_SIZE + q_heads_per_kv_head - 1) /
                q_heads_per_kv_head
            );

            size_t bytes_for_kv_tiles = params.tokens_per_page * max_unique_kv_heads_per_block *
                                       padded_head_dim * sizeof(half) * 2; // K and V

            return bytes_for_kv_tiles / params.tokens_per_page;
        } else {
            // For decode: one K/V pair per token
            return 2 * padded_head_dim * sizeof(half);
        }
    }
};

// Dispatch helpers for paged attention
void PagedAttentionPrimitive::dispatch_prefill_pass2(
    mlx::core::metal::Device& d,
    const mx::Stream& s,
    const CoreDims& core_dims,
    mx::array& out
) {
    using namespace kernels::paged_attention;

    debug::KernelDebugger::log_kernel_start("PAL Prefill Pass 2");

    const std::string library_name = "pal";
    const std::string kernel_name = "paged_attn_prefill_pass2_kernel";

    auto kernel_state = d.get_kernel(kernel_name, library_name);
    if (!kernel_state) {
        throw std::runtime_error("[PAL Primitive] Failed to load Pass 2 kernel: " + kernel_name);
    }

    auto& encoder = d.get_command_encoder(s.index);
    encoder.set_compute_pipeline_state(kernel_state);

    // Calculate dispatch grid
    metal::DispatchGrid grid = metal::MetalDispatcher::calculate_2d_dispatch(
        core_dims.query_token_count * core_dims.num_q_heads,
        (core_dims.query_token_count + PREFILL_PASS2_TOKEN_BLOCK_SIZE - 1) / PREFILL_PASS2_TOKEN_BLOCK_SIZE,
        (core_dims.num_q_heads + PREFILL_PASS2_QHEAD_BLOCK_SIZE - 1) / PREFILL_PASS2_QHEAD_BLOCK_SIZE
    );

    // Calculate thread configuration
    auto thread_config = metal::MetalDispatcher::calculate_optimal_threads(
        kernel_state, 64);

    // Calculate Pass 2 memory requirements
    size_t pass2_tg_mem_bytes = 0;
    pass2_tg_mem_bytes += thread_config.threads_per_group * sizeof(float) * 2;  // Max and sum scratch
    pass2_tg_mem_bytes += 2 * sizeof(float);                                    // Global stats
    pass2_tg_mem_bytes += thread_config.threads_per_group * core_dims.head_dim * sizeof(float); // O accumulators
    pass2_tg_mem_bytes += core_dims.head_dim * sizeof(float);                   // Final O
    pass2_tg_mem_bytes = kernel_utils::AttentionMemoryLayout::align_size(pass2_tg_mem_bytes);

    encoder.set_output_array(out, 8);

    debug::KernelDebugger::log_dispatch("PAL Prefill Pass 2", grid, thread_config);
    metal::MetalDispatcher::dispatch_kernel(encoder, grid, thread_config, pass2_tg_mem_bytes);

    debug::KernelDebugger::log_kernel_end("PAL Prefill Pass 2");
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

void PagedAttentionPrimitive::eval_gpu(const std::vector<mx::array>& inputs, mx::array& out) {
    debug::KernelDebugger::log_kernel_start("PAL Primitive");

    // Prepare Metal kernel and command encoder
    auto& s = stream();
    auto& d = mlx::core::metal::device(s.device);

    const std::string library_name = "pal";
    const std::string kernel_name = is_prefill_ ? "paged_attn_prefill_kernel" : "paged_attn_decode_kernel";
    auto kernel_state = d.get_kernel(kernel_name, library_name);
    if (!kernel_state) {
        throw std::runtime_error("[PAL Primitive] Failed to load kernel: " + kernel_name);
    }

    // Allocate output memory
    size_t bytes = out.nbytes();
    out.set_data(mx::allocator::malloc(bytes));

    // Validate inputs and extract dimensions
    PagedAttentionValidator validator;
    CoreDims core_dims = validator.validate_and_extract_dims(inputs);

    // Log input information
    std::vector<std::string> input_names = {
        "q", "k_pool", "v_pool", "page_table",
        "sequence_lengths", "query_to_seq_map", "query_token_offset"
    };
    debug::KernelDebugger::validate_and_log_inputs("PAL Primitive", inputs, input_names);

    // Build parameters
    PagedAttentionParamBuilder param_builder(core_dims, inputs[1], inputs[3], is_prefill_);
    auto* device_ptr = d.mtl_device();
    PagedAttentionParams params = param_builder.build(device_ptr);

    // Calculate thread configuration
    auto thread_config = metal::MetalDispatcher::calculate_optimal_threads(
        kernel_state, kernels::DEFAULT_THREADS_PER_GROUP);

    // Calculate memory layout
    auto memory_layout = kernel_utils::calculate_attention_memory_layout(
        params, thread_config.threads_per_group, thread_config.execution_width);

    debug::KernelDebugger::log_memory_layout("PAL Primitive", memory_layout);

    // Validate input pointers
    metal::MetalDispatcher::validate_input_pointers(inputs, "PAL Primitive");

    // Calculate dispatch grid
    metal::DispatchGrid grid;
    if (is_prefill_) {
        grid.width = core_dims.num_items_to_process / params.tile_size_T_runtime;
    } else {
        grid.width = core_dims.num_items_to_process;
    }
    grid.height = 1;
    grid.depth = 1;

    // Setup compute encoder
    auto& compute_encoder = d.get_command_encoder(s.index);
    compute_encoder.set_compute_pipeline_state(kernel_state);

    // Set input arrays
    metal::MetalDispatcher::setup_input_arrays(compute_encoder, inputs, 0);

    // Set parameters and output
    compute_encoder.set_bytes(&params, sizeof(PagedAttentionParams), 7);
    compute_encoder.set_output_array(out, 8);

    // Dispatch kernel
    debug::KernelDebugger::log_dispatch("PAL Primitive", grid, thread_config);
    metal::MetalDispatcher::dispatch_kernel(compute_encoder, grid, thread_config,
                                          memory_layout.total_bytes);

    // Launch Pass 2 for prefill
    if (is_prefill_) {
        dispatch_prefill_pass2(d, s, core_dims, out);
    }

    debug::KernelDebugger::log_kernel_end("PAL Primitive");
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
       << ", is_prefill=" << (is_prefill_ ? "true" : "false") << ")";
}

}  // namespace pal::cpp
