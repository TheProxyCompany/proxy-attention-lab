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
#include "pal_core/kernel_utils/kernel_constants.hpp"
#include "pal_core/kernel_utils/memory_layout.hpp"
#include "pal_core/kernel_utils/validation.hpp"
#include "pal_core/kernel_utils/tiling.hpp"
#include "pal_core/metal/dispatch.hpp"
#include "pal_core/kernel_utils/param_builder.hpp"

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
#include <mlx/ops.h>
#include <mlx/backend/cpu/encoder.h>
#include <mlx/backend/metal/device.h>
#include <mlx/backend/metal/metal.h>
#include "pal_core/metal/metal_loader.hpp"

#include <mlx/utils.h>

#include "shaders/paged_attention_types.h"

namespace mx = mlx::core;

namespace pal::cpp {

// Define BatchLogicalPage structure for work items
struct BatchLogicalPage {
    uint32_t batch_item_idx;
    uint32_t logical_page_idx_in_sequence;
};

static std::tuple<mx::array, size_t> create_work_items_buffer(const PagedAttentionParams& params, const mx::array& sequence_lengths) {
    std::vector<BatchLogicalPage> active_work_items;
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
                active_work_items.push_back({b_idx, lp_idx});
            }
        } else {
            spdlog::debug("[create_work_items_buffer] Batch {}: seq_len={} (skipped)", b_idx, seq_len);
        }
    }

    // Create work_items_buffer mx::array
    std::vector<uint32_t> flat_work_items_data;
    flat_work_items_data.reserve(active_work_items.size() * 2);
    for (const auto& item : active_work_items) {
        flat_work_items_data.push_back(item.batch_item_idx);
        flat_work_items_data.push_back(item.logical_page_idx_in_sequence);
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


// Paged attention specific validator
class PagedAttentionValidator : public kernel_utils::KernelValidator {
public:
    // Constructor to store the prefill mode
    explicit PagedAttentionValidator(bool is_prefill) : is_prefill_(is_prefill) {}

    CoreDims validate_and_extract_dims(const std::vector<mx::array>& inputs) {
        if (inputs.size() != 7) {
            throw std::invalid_argument(
                format_error("PAL Primitive", "Expected 7 inputs, received " + std::to_string(inputs.size())));
        }
        // Extract early for CoreDims population
        const auto& k_pool = inputs[1];
        kernel_utils::ValidationUtils::check_ndim(k_pool, 4, "k_pool");
        extracted_dims_.tokens_per_page = k_pool.shape(1);
        extracted_dims_.num_kv_heads = k_pool.shape(2);
        extracted_dims_.head_dim = k_pool.shape(3);

        // Now call the main validation logic
        validate(inputs);
        return extracted_dims_;
    }

    void validate(const std::vector<mx::array>& inputs) override {
        const auto& q = inputs[0];
        const auto& k_pool = inputs[1];
        const auto& v_pool = inputs[2];
        const auto& page_table = inputs[3];
        const auto& sequence_lengths = inputs[4];
        const auto& query_to_seq_map = inputs[5];
        const auto& query_token_offset = inputs[6];

        // --- Common Validations ---
        kernel_utils::ValidationUtils::check_dtype(page_table, mx::uint32, "page_table");
        kernel_utils::ValidationUtils::check_dtype(sequence_lengths, mx::int32, "sequence_lengths");
        kernel_utils::ValidationUtils::check_dtype(query_to_seq_map, mx::int32, "query_to_seq_map");
        kernel_utils::ValidationUtils::check_dtype(query_token_offset, mx::int32, "query_token_offset");

        // K/V Pool validation (ndim check already done for CoreDims, check V-pool matches K-pool)
        kernel_utils::ValidationUtils::check_ndim(v_pool, 4, "v_pool");
        if (v_pool.shape() != k_pool.shape()) {
            throw std::invalid_argument(format_error("PAL Primitive", "K-pool and V-pool shapes must match."));
        }
        // extracted_dims_ for tokens_per_page, num_kv_heads, head_dim are from k_pool.

        kernel_utils::ValidationUtils::check_divisibility(
            extracted_dims_.head_dim,
            kernels::paged_attention::HEAD_DIM_VECTORIZATION,
            "head_dim"
        );

        // Query format validation (populates num_q_heads, num_items_to_process, query_token_count in extracted_dims_)
        validate_query_format(q); // This uses extracted_dims_.head_dim

        // Page table validation
        kernel_utils::ValidationUtils::check_ndim(page_table, 2, "page_table");
        if (sequence_lengths.ndim() != 1) {
             throw std::invalid_argument(format_error("PAL Primitive", "sequence_lengths must be a 1D array."));
        }
        if (page_table.shape(0) != sequence_lengths.size()) {
            throw std::invalid_argument(format_error("PAL Primitive", "page_table batch dimension must match size of sequence_lengths."));
        }


        // --- Mode-Specific Validations ---
        if (is_prefill_) {
            spdlog::debug("[PAL Validate] Prefill mode specific input validation...");
            size_t total_tokens_from_seq_lengths = 0;
            const int32_t* sl_ptr = sequence_lengths.data<int32_t>();
            for (int i = 0; i < sequence_lengths.size(); ++i) {
                if (sl_ptr[i] < 0) {
                    throw std::invalid_argument(format_error("PAL Prefill", "Sequence lengths cannot be negative."));
                }
                total_tokens_from_seq_lengths += static_cast<size_t>(sl_ptr[i]);
            }

            if (extracted_dims_.query_token_count != total_tokens_from_seq_lengths) {
                std::ostringstream oss;
                oss << "For prefill mode, total query tokens from 'queries' input (" << extracted_dims_.query_token_count
                    << ") must equal the sum of 'sequence_lengths' (" << total_tokens_from_seq_lengths
                    << "). 'queries' must contain all tokens for all sequences, batched contiguously.";
                throw std::invalid_argument(format_error("PAL Prefill", oss.str()));
            }

            // query_to_seq_map and query_token_offset must also match this total token count.
            if (query_to_seq_map.size() != extracted_dims_.query_token_count) {
                 throw std::invalid_argument(format_error("PAL Prefill", "query_to_seq_map size must match total query tokens."));
            }
            if (query_token_offset.size() != extracted_dims_.query_token_count) {
                 throw std::invalid_argument(format_error("PAL Prefill", "query_token_offset size must match total query tokens."));
            }

        } else { // Decode Mode
            spdlog::debug("[PAL Validate] Decode mode specific input validation...");
            // In decode, query_token_count is the number of individual queries being decoded (e.g., batch_size if one token per seq).
            // query_to_seq_map and query_token_offset must match this count.
            if (query_to_seq_map.size() != extracted_dims_.query_token_count) {
                 std::ostringstream oss;
                 oss << "For decode mode, query_to_seq_map size (" << query_to_seq_map.size()
                     << ") must match number of query items (" << extracted_dims_.query_token_count << ").";
                 throw std::invalid_argument(format_error("PAL Decode", oss.str()));
            }
            if (query_token_offset.size() != extracted_dims_.query_token_count) {
                 std::ostringstream oss;
                 oss << "For decode mode, query_token_offset size (" << query_token_offset.size()
                     << ") must match number of query items (" << extracted_dims_.query_token_count << ").";
                 throw std::invalid_argument(format_error("PAL Decode", oss.str()));
            }

            // Check that sequence indices in query_to_seq_map are valid
            const int32_t* q_to_s_ptr = query_to_seq_map.data<int32_t>();
            uint32_t num_sequences_in_batch_val = sequence_lengths.size();
            for (size_t i = 0; i < query_to_seq_map.size(); ++i) {
                if (q_to_s_ptr[i] < 0 || static_cast<uint32_t>(q_to_s_ptr[i]) >= num_sequences_in_batch_val) {
                    std::ostringstream oss;
                    oss << "For decode mode, query_to_seq_map contains invalid sequence index " << q_to_s_ptr[i]
                        << " at query item " << i << ". Valid range [0, " << num_sequences_in_batch_val - 1 << "].";
                    throw std::out_of_range(format_error("PAL Decode", oss.str()));
                }
            }
        }

        spdlog::debug("[PAL Primitive Validate] Input validation complete.");
    }

private:
    CoreDims extracted_dims_;
    bool is_prefill_;

    void validate_query_format(const mx::array& q) {
        if (q.ndim() == 3) {
            // For 3D queries, q.shape(0) is number of tokens per q_head group if prefill, or num individual query items if decode
            // q.shape(1) is num_q_heads
            // q.shape(2) is head_dim
            if (q.shape(2) != static_cast<int>(extracted_dims_.head_dim)) {
                throw std::invalid_argument(format_error("PAL Primitive", "3D Query HeadDim mismatch with K-pool head_dim."));
            }
            extracted_dims_.num_q_heads = q.shape(1);
            extracted_dims_.query_token_count = q.shape(0); // Number of "token rows" or "item groups"
            extracted_dims_.num_items_to_process = extracted_dims_.query_token_count * extracted_dims_.num_q_heads; // Total Q-Head items
        } else if (q.ndim() == 2) {
            // For 2D queries, q.shape(0) is number of query items/tokens
            // q.shape(1) is head_dim
            if (q.shape(1) != static_cast<int>(extracted_dims_.head_dim)) {
                throw std::invalid_argument(format_error("PAL Primitive", "2D Query HeadDim mismatch with K-pool head_dim."));
            }
            extracted_dims_.num_q_heads = 1; // Implicitly 1 for 2D queries
            extracted_dims_.query_token_count = q.shape(0); // Number of query items/tokens
            extracted_dims_.num_items_to_process = extracted_dims_.query_token_count;
        } else if (q.ndim() == 1) {
            // For 1D queries, q.shape(0) is number of query items/tokens, head_dim must be 1
            if (extracted_dims_.head_dim != 1) {
                throw std::invalid_argument(format_error("PAL Primitive", "1D Query requires K-pool head_dim to be 1."));
            }
            extracted_dims_.num_q_heads = 1; // Implicitly 1
            extracted_dims_.query_token_count = q.shape(0); // Number of query items/tokens
            extracted_dims_.num_items_to_process = extracted_dims_.query_token_count;
        } else {
            throw std::invalid_argument(format_error("PAL Primitive", "Query 'q' ndim (" + std::to_string(q.ndim()) + ") not supported."));
        }

        if (extracted_dims_.num_items_to_process == 0 && q.size() > 0) { // Or check extracted_dims_.query_token_count
            throw std::invalid_argument(format_error("PAL Primitive", "Number of items/tokens to process is 0 but query array is not empty."));
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

        params.num_q_heads = dims_.num_q_heads;
        params.num_kv_heads = dims_.num_kv_heads;
        params.head_dim = dims_.head_dim;
        params.tokens_per_page = dims_.tokens_per_page;
        params.log_exp_min_clamp = kernels::paged_attention::kLogFp16DenormMinVal;
        params.max_logical_blocks_per_seq = page_table_arr_.shape(1);
        params.num_sequences_in_batch = page_table_arr_.shape(0);
        params.num_physical_pages_in_pool = k_pool_arr_.shape(0);
        params.num_active_batch_logical_pages = 1;  // Placeholder
        params.query_token_count_total = dims_.query_token_count;
        params.pass2_token_block_size = kernels::paged_attention::PREFILL_PASS2_TOKEN_BLOCK_SIZE;
        params.pass2_qhead_block_size = kernels::paged_attention::PREFILL_PASS2_QHEAD_BLOCK_SIZE;

        // Calculate inv_sqrt_head_dim
        params.inv_sqrt_head_dim = 1.0f / std::sqrt(static_cast<float>(params.head_dim));

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
        if (is_prefill_) {
            params.tile_size_T_runtime = params.tokens_per_page;
        } else {
            // For decode mode, use existing tile size calculation logic
            // Calculate fixed memory usage with zero tile size
            PagedAttentionParams params_for_sizing = params;
            params_for_sizing.tile_size_T_runtime = 0;

            auto fixed_layout = kernel_utils::calculate_attention_memory_layout(
                params_for_sizing,
                kernels::DEFAULT_THREADS_PER_GROUP,
                kernels::SIMD_WIDTH,
                false  // is_prefill
            );

            // Get memory constraints
            auto constraints = get_memory_constraints(device, fixed_layout.total_bytes);

            // Calculate bytes per token for tiling
            size_t bytes_per_token = 2 * params.head_dim * sizeof(half);

            // Use tiling utility to calculate optimal tile size
            tiling::TileRequirements requirements{
                .bytes_per_element = bytes_per_token,
                .alignment = kernels::paged_attention::TILE_SIZE_ALIGNMENT,
                .min_size = kernels::paged_attention::MIN_TILE_SIZE_SOFT,
                .max_size = kernels::paged_attention::MAX_TILE_SIZE_PRACTICAL,
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
        }
    }
};

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
    // Allocate output memory
    size_t bytes = out.nbytes();
    out.set_data(mx::allocator::malloc(bytes));

    auto& s = stream();
    auto& d = mlx::core::metal::device(s.device);

    // Dispatch to appropriate implementation
    if (is_prefill_) {
        _eval_gpu_prefill(s, d, inputs, out);
    } else {
        _eval_gpu_decode(s, d, inputs, out);
    }
}

void PagedAttentionPrimitive::_eval_gpu_decode(
    const mlx::core::Stream& stream,
    mlx::core::metal::Device& device,
    const std::vector<mx::array>& inputs,
    mx::array& out
) {
    // Prepare Metal kernel and command encoder
    const std::string library_name = "pal";
    const std::string kernel_name = "paged_attn_decode_kernel";
    auto kernel_state = device.get_kernel(kernel_name, library_name);
    if (!kernel_state) {
        throw std::runtime_error("[PAL Primitive] Failed to load kernel: " + kernel_name);
    }

    // Validate inputs and extract dimensions
    PagedAttentionValidator validator(/* is_prefill */ false);
    CoreDims core_dims = validator.validate_and_extract_dims(inputs);

    // Build parameters for decode
    PagedAttentionParamBuilder param_builder(core_dims, inputs[1], inputs[3], false);
    PagedAttentionParams params = param_builder.build(device.mtl_device());

    // Calculate thread configuration
    auto thread_config = metal::MetalDispatcher::calculate_optimal_threads(
        kernel_state,
        kernels::DEFAULT_THREADS_PER_GROUP
    );

    // Calculate memory layout
    auto memory_layout = kernel_utils::calculate_attention_memory_layout(
        params,
        thread_config.threads_per_group,
        thread_config.execution_width,
        false  // is_prefill
    );

    // Calculate dispatch grid for decode
    metal::DispatchGrid grid;
    grid.width = core_dims.num_items_to_process;
    grid.height = 1;
    grid.depth = params.num_sequences_in_batch;

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

void PagedAttentionPrimitive::_eval_gpu_prefill(
    const mlx::core::Stream& stream,
    mlx::core::metal::Device& device,
    const std::vector<mx::array>& inputs,
    mx::array& out
) {
    auto* metal_device_ptr = device.mtl_device();

    const std::string library_name = "pal";
    auto kernel_state = device.get_kernel("paged_attn_prefill_pass1_kernel", library_name);
    if (!kernel_state) {
        throw std::runtime_error("[PAL Primitive] Failed to load kernel: paged_attn_prefill_pass1_kernel");
    }

    // Validate inputs and extract dimensions
    PagedAttentionValidator validator(/*is_prefill*/ true);
    CoreDims core_dims = validator.validate_and_extract_dims(inputs);

    auto [tile_size, final_threads_per_tg, actual_simd_width] = get_optimal_tile_size_and_thread_info(
        core_dims.head_dim,
        core_dims.num_q_heads,
        core_dims.num_kv_heads,
        stream,
        kernel_state
    );

    // Build parameters for prefill
    PagedAttentionParamBuilder param_builder(core_dims, inputs[1], inputs[3], true);
    PagedAttentionParams params = param_builder.build(metal_device_ptr);

    // Set both tokens_per_page and tile_size_T_runtime to tile_size for prefill
    params.tokens_per_page = tile_size;
    params.tile_size_T_runtime = tile_size; // K/V tile depth is tile_size

    // Calculate memory layout
    auto memory_layout = kernel_utils::calculate_attention_memory_layout(
        params,
        final_threads_per_tg,
        actual_simd_width,
        true  // is_prefill
    );
    spdlog::debug("[PAL Prefill Debug] TGMem Assertion Check: Calculated layout total_bytes = {}", memory_layout.total_bytes);
    if (memory_layout.total_bytes > metal_device_ptr->maxThreadgroupMemoryLength()) {
        throw std::runtime_error("[PAL Primitive] Prefill TGMem budget EXCEEDED with tile_size: " + std::to_string(tile_size));
    }

    // Create work items buffer and update params BEFORE allocating arrays
    auto [work_items_buffer, num_active_batch_logical_pages_val] = create_work_items_buffer(params, inputs[4]);

    spdlog::debug("[PAL Prefill Debug] After create_work_items_buffer: num_active_batch_logical_pages_val = {}",
                  num_active_batch_logical_pages_val);
    spdlog::debug("[PAL Prefill Debug] work_items_buffer shape: [{}, {}]",
                  work_items_buffer.shape(0), work_items_buffer.shape(1));

    params.num_active_batch_logical_pages = num_active_batch_logical_pages_val;
    spdlog::debug("[PAL Prefill Debug] params.num_active_batch_logical_pages SET to: {}",
                  params.num_active_batch_logical_pages);

    if (params.num_active_batch_logical_pages == 0) {
        spdlog::warn("[PAL Prefill] No active batch logical pages to process. Skipping prefill kernel launches and zeroing output.");
        return; // Exit _eval_gpu_prefill early
    }

    mx::array query_starts_buffer = create_query_starts_buffer(
        params.num_sequences_in_batch,
        inputs[4],
        stream
    );
    spdlog::debug("[PAL Prefill Debug] query_starts_buffer created. Shape: [{}]", query_starts_buffer.shape(0));

    // Before allocating intermediate arrays
    spdlog::debug("[PAL Prefill Debug] Allocating intermediate arrays with: query_token_count_total={}, num_q_heads={}, num_active_batch_logical_pages={}",
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
    spdlog::debug("[PAL Prefill Debug] Pass 1 Grid: width={}, height={}, depth={}",
                  grid.width, grid.height, grid.depth);
    spdlog::debug("[PAL Prefill Debug] Passing params to Pass 1 with num_active_batch_logical_pages = {}",
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
    compute_encoder.set_input_array(inputs[5], 5);
    compute_encoder.set_input_array(inputs[6], 6);
    compute_encoder.set_bytes(&params, sizeof(PagedAttentionParams), 7);
    compute_encoder.set_input_array(work_items_buffer, 8);
    compute_encoder.set_input_array(query_starts_buffer, 9);

    // set intermediate outputs for prefill Pass 1
    mx::array m_locals_pass1_out = mx::array(m_s_shape, mx::float32, nullptr, {});
    m_locals_pass1_out.set_data(mx::allocator::malloc(m_locals_pass1_out.nbytes()));

    mx::array s_locals_pass1_out = mx::array(m_s_shape, mx::float32, nullptr, {});
    s_locals_pass1_out.set_data(mx::allocator::malloc(s_locals_pass1_out.nbytes()));

    mx::array o_partials_pass1_out = mx::array(o_shape, mx::float16, nullptr, {});
    o_partials_pass1_out.set_data(mx::allocator::malloc(o_partials_pass1_out.nbytes()));

    // Initialize intermediate arrays
    std::fill_n(m_locals_pass1_out.data<float>(), m_locals_pass1_out.size(), -std::numeric_limits<float>::infinity());
    std::fill_n(s_locals_pass1_out.data<float>(), s_locals_pass1_out.size(), 0.0f);

    device.add_temporary(m_locals_pass1_out, stream.index);
    device.add_temporary(s_locals_pass1_out, stream.index);
    device.add_temporary(o_partials_pass1_out, stream.index);

    compute_encoder.set_output_array(m_locals_pass1_out, 10);
    compute_encoder.set_output_array(s_locals_pass1_out, 11);
    compute_encoder.set_output_array(o_partials_pass1_out, 12);

    // Dispatch kernel
    metal::MetalDispatcher::dispatch_kernel(
        compute_encoder, grid,
        final_threads_per_tg,
        memory_layout.total_bytes
    );

    kernel_state = device.get_kernel("paged_attn_prefill_pass2_kernel", library_name);
    if (!kernel_state) {
        throw std::runtime_error("[PAL Primitive] Failed to load Pass 2 kernel: paged_attn_prefill_pass2_kernel");
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
    size_t PASS_2_SIMD_GROUPS_PER_TG = 4;
    size_t desired_threads_per_tg = kernel_state->threadExecutionWidth() * PASS_2_SIMD_GROUPS_PER_TG;
    auto thread_config = metal::MetalDispatcher::calculate_optimal_threads(kernel_state, desired_threads_per_tg);

    // Calculate Pass 2 memory requirements
    size_t pass2_tg_mem_bytes = 0;
    pass2_tg_mem_bytes += thread_config.threads_per_group * sizeof(float) * 2;  // M and S scratch
    pass2_tg_mem_bytes += thread_config.threads_per_group * core_dims.head_dim * sizeof(float); // O accumulators
    pass2_tg_mem_bytes += core_dims.head_dim * sizeof(float);                   // Final O
    pass2_tg_mem_bytes += 2 * sizeof(float);                                    // Global stats
    pass2_tg_mem_bytes = kernel_utils::AttentionMemoryLayout::align_size(pass2_tg_mem_bytes);

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
    compute_encoder.set_input_array(work_items_buffer, 3);
    // Set parameters
    compute_encoder.set_bytes(&params, sizeof(PagedAttentionParams), 4);
    // Set output array
    compute_encoder.set_output_array(out, 5);

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
       << ", is_prefill=" << (is_prefill_ ? "true" : "false") << ")";
}

size_t PagedAttentionPrimitive::calculate_per_gqa_group_compute_scratch(
    uint32_t head_dimension,
    uint32_t number_of_simd_groups,
    uint32_t threads_per_group
) {
    size_t per_gqa_group_compute_scratch = 0;

    // Component 1: Potential TGMem for V-sum accumulators (one per active SIMD group in the GQA group)
    per_gqa_group_compute_scratch += number_of_simd_groups * head_dimension * sizeof(float);
    per_gqa_group_compute_scratch = kernel_utils::AttentionMemoryLayout::align_size(per_gqa_group_compute_scratch);

    // Component 2: Potential TGMem for M/L stats (one pair per active SIMD group)
    per_gqa_group_compute_scratch += number_of_simd_groups * 2 * sizeof(float);
    per_gqa_group_compute_scratch = kernel_utils::AttentionMemoryLayout::align_size(per_gqa_group_compute_scratch);

    // Component 3: General TG-wide reduction scratch
    per_gqa_group_compute_scratch += threads_per_group * sizeof(float);
    per_gqa_group_compute_scratch = kernel_utils::AttentionMemoryLayout::align_size(per_gqa_group_compute_scratch);

    // Component 4: Small constant safety/alignment buffer
    per_gqa_group_compute_scratch += pal::cpp::kernel_utils::kFinalMemoryPaddingGuardBytes;
    per_gqa_group_compute_scratch = kernel_utils::AttentionMemoryLayout::align_size(per_gqa_group_compute_scratch);

    spdlog::debug("[PAL Prefill Debug] Calculated per_gqa_group_compute_scratch = {} bytes for D_s calculation", per_gqa_group_compute_scratch);

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
    fixed_overhead_bytes = kernel_utils::AttentionMemoryLayout::align_size(fixed_overhead_bytes);

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

    uint32_t symmetric_depth = std::max(4u, unaligned_depth); // Ensure at least a minimum depth
    symmetric_depth = (symmetric_depth / 4) * 4;             // Align down to multiple of 4
    if (symmetric_depth < 4) {
        symmetric_depth = 4; // Ensure minimum depth of 4 after alignment
    }

    spdlog::debug("[calculate_symmetric_tile_depth] Final symmetric depth (D_s): {}", symmetric_depth);
    return std::min(kernels::paged_attention::MAX_TILE_SIZE_PRACTICAL, symmetric_depth);
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

    uint32_t desired_threads_per_tg = actual_simd_width * N_q_per_kv * SIMD_GROUPS_PER_GQA_GROUP_FACTOR;
    uint32_t max_threads_device = kernel_state->maxTotalThreadsPerThreadgroup();
    uint32_t final_threads_per_tg = std::min(desired_threads_per_tg, max_threads_device);
    final_threads_per_tg = ((final_threads_per_tg + actual_simd_width - 1) / actual_simd_width) * actual_simd_width;
    spdlog::debug("[PAL Prefill Debug] final_threads_per_tg = {}", final_threads_per_tg);

    uint32_t total_simd_groups_in_tg = final_threads_per_tg / actual_simd_width;
    spdlog::debug("[PAL Prefill Debug] total_simd_groups_in_tg = {}", total_simd_groups_in_tg);

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

    spdlog::debug("[PAL Prefill Debug] D_s = {}", D_s);

    return std::make_tuple(D_s, final_threads_per_tg, actual_simd_width);
  }

}  // namespace pal::cpp
