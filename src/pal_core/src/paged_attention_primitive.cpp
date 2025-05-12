#include "pal_core/paged_attention_primitive.hpp"
#include <mlx/allocator.h>
#include <mlx/backend/metal/metal.h>
#include <mlx/backend/metal/device.h>
#include <mlx/backend/metal/utils.h>
#include <mlx/backend/cpu/encoder.h>
#include "shaders/paged_attention_types.h"

#include <stdexcept>
#include <iostream>

namespace mx = mlx::core;

namespace pal::cpp {

// --- Constructor ---
PagedAttentionPrimitive::PagedAttentionPrimitive(mx::StreamOrDevice stream_or_device)
    : mx::UnaryPrimitive(mx::to_stream(stream_or_device)) { // Correctly use the passed stream/device
    std::cerr << "[PAL Primitive] PagedAttentionPrimitive constructed." << std::endl;
}

// --- CPU Evaluation (Stub) ---
void PagedAttentionPrimitive::eval_cpu(const std::vector<mx::array>& inputs, mx::array& out) {
    std::cerr << "[PAL Primitive] PagedAttentionPrimitive::eval_cpu called (not supported)." << std::endl;
    throw std::runtime_error("[PagedAttentionPrimitive] CPU evaluation is not supported for paged attention.");
}

// --- GPU Evaluation (Kernel Launch Logic) ---
void PagedAttentionPrimitive::eval_gpu(const std::vector<mx::array>& inputs, mx::array& out) {
    std::cerr << "[PAL Primitive] PagedAttentionPrimitive::eval_gpu called." << std::endl;

    // --- 1. Input Validation ---
    // We now expect 7 tensor inputs based on ops.cpp
    if (inputs.size() != 7) {
        throw std::invalid_argument(
            "[PagedAttentionPrimitive::eval_gpu] Expected 7 inputs (Q, K_pool, V_pool, Page_table, SeqLens, QToSeqMap, QOffsets), received " +
            std::to_string(inputs.size()));
    }

    const auto& q = inputs[0];
    const auto& k_pool = inputs[1];
    const auto& v_pool = inputs[2];
    const auto& page_table = inputs[3];
    const auto& sequence_lengths = inputs[4];
    const auto& query_to_seq_map = inputs[5];
    const auto& query_token_offset = inputs[6];

    // Basic dtype checks (can be expanded)
    if (q.dtype() != mx::float16 || k_pool.dtype() != mx::float16 || v_pool.dtype() != mx::float16) {
         throw std::invalid_argument("[PagedAttentionPrimitive::eval_gpu] Kernel requires float16 Q, K_pool, and V_pool.");
    }
    if (page_table.dtype() != mx::uint32) {
         throw std::invalid_argument("[PagedAttentionPrimitive::eval_gpu] Page table must be uint32.");
    }
    if (sequence_lengths.dtype() != mx::int32 || query_to_seq_map.dtype() != mx::int32 || query_token_offset.dtype() != mx::int32) {
        throw std::invalid_argument("[PagedAttentionPrimitive::eval_gpu] Sequence metadata arrays (lengths, maps, offsets) must be int32.");
    }
    std::cerr << "[PAL Primitive] Input validation passed." << std::endl;

    // --- 2. Allocate Output Buffer ---
    // MLX ensures `out` has the correct shape and dtype (from `output_shapes()`) before calling `eval_gpu`.
    // We just need to allocate the underlying data buffer.
    out.set_data(mx::allocator::malloc(out.nbytes()));
    std::cerr << "[PAL Primitive] Output buffer allocated for shape: [";
    for(size_t i = 0; i < out.ndim(); ++i) std::cerr << (i > 0 ? ", " : "") << out.shape(i);
    std::cerr << "], dtype: " << out.dtype() << std::endl;


    // --- 3. Prepare Metal Kernel and Command Encoder ---
    auto& s = stream(); // Get the MLX stream for this operation
    auto& d = mlx::core::metal::device(s.device); // Get the MLX Metal device associated with the stream

    const std::string library_name_for_mlx = "pal";       // Alias used during MetalLibRegistrar
    const std::string kernel_name = "paged_attn_kernel"; // Name of your kernel function in .metal

    MTL::ComputePipelineState* kernel_pipeline_state = nullptr;
    try {
        kernel_pipeline_state = d.get_kernel(kernel_name, library_name_for_mlx);
    } catch (const std::runtime_error& e) {
        std::cerr << "[PAL Primitive] Error getting kernel: " << e.what() << std::endl;
        throw; // Re-throw if kernel cannot be retrieved
    }
    if (!kernel_pipeline_state) {
         throw std::runtime_error("[PagedAttentionPrimitive] Failed to get kernel '" + kernel_name +
                                  "' from library '" + library_name_for_mlx + "'.");
    }
    std::cerr << "[PAL Primitive] Metal kernel pipeline state retrieved." << std::endl;

    auto& compute_encoder = d.get_command_encoder(s.index); // Get a command encoder for this stream
    compute_encoder.set_compute_pipeline_state(kernel_pipeline_state);
    std::cerr << "[PAL Primitive] Compute pipeline state set on encoder." << std::endl;

    // --- 4. Create and Populate PagedAttentionParams Struct ---
    PagedAttentionParams params_struct; // Defined in shaders/paged_attention_types.h

    // Assuming Q is [TotalQueryTokens, NumQHeads, HeadDim] or [TotalQueryTokens, ModelDim]
    // If Q is [TotalQueryTokens, ModelDim], then NumQHeads might be 1 for this interpretation,
    // or this primitive is called *after* Q has been projected to heads.
    // For now, let's assume Q is already shaped for heads for simplicity in PAL.
    // PIE's Attention layer will handle the QKV projections and RoPE.
    if (q.ndim() < 1) throw std::invalid_argument("Queries 'q' must have at least 1 dimension.");
    params_struct.head_dim = q.shape(-1);
    params_struct.num_q_heads = (q.ndim() >= 2) ? q.shape(q.ndim() - 2) : 1;

    // num_kv_heads would ideally come from k_pool's shape or another explicit param.
    // For now, assume it's same as num_q_heads or derived from k_pool.
    // k_pool shape: [NumTotalPhysicalPages, TokensPerPage, NumKVHeads, HeadDim]
    if (k_pool.ndim() >= 3) {
        params_struct.num_kv_heads = k_pool.shape(k_pool.ndim() - 2);
    } else {
        params_struct.num_kv_heads = params_struct.num_q_heads; // Fallback
    }

    params_struct.tokens_per_page = 64; // Matching your C++ KVPage constexpr
    if (params_struct.head_dim > 0) {
        params_struct.scale = 1.0f / sqrtf(static_cast<float>(params_struct.head_dim));
    } else {
        params_struct.scale = 1.0f; // Avoid division by zero if head_dim is 0 (should not happen)
    }

    // max_logical_blocks_per_seq: This depends on how page_table is shaped.
    // If page_table is [NumSequencesInBatch, MaxLogicalBlocksPerSequence] (after PIE prepares it)
    // For this test, page_table is 1D. Let's make a placeholder assumption.
    // This should ideally be passed from PIE or be derivable from a known max context length.
    // If page_table is flat [TotalEntries], this param's meaning changes or isn't directly from its shape.
    // For now, let's assume page_table from Python test is flat [NumElements],
    // and we'll use a fixed value or derive it differently if needed by kernel logic.
    // Let's assume for the test it's effectively 1 sequence with page_table.size() blocks.
    params_struct.max_logical_blocks_per_seq = page_table.size() > 0 ? page_table.shape(0) : 1;
    // If page_table were 2D [NumSeq, MaxBlocks], it would be page_table.shape(1).

    std::cerr << "[PAL Primitive] Params: num_q_heads=" << params_struct.num_q_heads
              << ", num_kv_heads=" << params_struct.num_kv_heads
              << ", head_dim=" << params_struct.head_dim
              << ", tokens_per_page=" << params_struct.tokens_per_page
              << ", scale=" << params_struct.scale
              << ", max_logical_blocks_per_seq=" << params_struct.max_logical_blocks_per_seq << std::endl;

    // --- 5. Set Kernel Arguments ---
    // Order must match buffer indices in paged_attention.h.metal
    compute_encoder.set_input_array(q, 0);
    compute_encoder.set_input_array(k_pool, 1);
    compute_encoder.set_input_array(v_pool, 2);
    compute_encoder.set_input_array(page_table, 3);
    compute_encoder.set_input_array(sequence_lengths, 4);
    compute_encoder.set_input_array(query_to_seq_map, 5);
    compute_encoder.set_input_array(query_token_offset, 6);
    compute_encoder.set_bytes(&params_struct, sizeof(PagedAttentionParams), 7); // Pass struct by pointer
    compute_encoder.set_output_array(out, 8);
    std::cerr << "[PAL Primitive] All kernel buffers and params struct set." << std::endl;

    // --- 6. Dispatch Kernel ---
    // The threading strategy here is crucial and depends on your kernel's design.
    // For the current stub kernel that uses global_query_thread_idx to index into 1D q:
    // Grid size should be the number of elements we want to process independently.
    // If q is [TotalQueryTokens, NumQHeads, HeadDim], a common strategy is:
    // grid_dim_x = TotalQueryTokens
    // grid_dim_y = NumQHeads
    // And kernel uses [[thread_position_in_grid]].x and .y
    // For our current simple 1D q test and 1D grid dispatch:
    size_t grid_dim_x = q.shape(0); // Assuming q is effectively 1D for global_query_thread_idx access

    if (grid_dim_x == 0) {
        std::cerr << "[PAL Primitive Warning] Input 'q' first dimension is 0, skipping kernel dispatch." << std::endl;
        return;
    }

    MTL::Size grid_dims = MTL::Size(grid_dim_x, 1, 1);
    size_t tgp_size_max = kernel_pipeline_state->maxTotalThreadsPerThreadgroup();
    size_t tgp_size = std::min(tgp_size_max, grid_dim_x); // Don't request more threads per group than in grid
    if (grid_dim_x > 0 && tgp_size == 0) tgp_size = 1;   // Ensure tgp_size is at least 1

    MTL::Size group_dims = MTL::Size(tgp_size, 1, 1);
    std::cerr << "[PAL Primitive] Dispatching kernel: grid_dims=(" << grid_dims.width << "," << grid_dims.height << "," << grid_dims.depth
              << "), group_dims=(" << group_dims.width << "," << group_dims.height << "," << group_dims.depth << ")" << std::endl;

    compute_encoder.dispatch_threads(grid_dims, group_dims);
    std::cerr << "[PAL Primitive] Kernel dispatched." << std::endl;
}

// --- Print Method ---
void PagedAttentionPrimitive::print(std::ostream& os) {
    os << "PagedAttention";
    // Add any parameters if the primitive stores them, e.g.:
    // os << "(block_size=" << block_size_ << ")";
}

// --- Equivalence Check ---
bool PagedAttentionPrimitive::is_equivalent(const mx::Primitive& other) const {
    // Check if the other primitive is the same type
    if (typeid(*this) != typeid(other)) {
        return false;
    }
    // Cast and compare any parameters if they exist
    // const PagedAttentionPrimitive& other_pa = static_cast<const PagedAttentionPrimitive&>(other);
    // return this->block_size_ == other_pa.block_size_; // Example
    return true; // No parameters stored currently
}

// --- Gradient Stubs ---
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

// --- Vmap Stub ---
std::pair<std::vector<mx::array>, std::vector<int>> PagedAttentionPrimitive::vmap(
    const std::vector<mx::array>& inputs,
    const std::vector<int>& axes) {
    // Vmap for attention might require careful handling of batch/head dimensions.
    throw std::runtime_error("[PagedAttentionPrimitive] Vmap not implemented.");
}

// --- Output Shape Calculation ---
std::vector<mx::Shape> PagedAttentionPrimitive::output_shapes(const std::vector<mx::array>& inputs) {
    // Assuming the output shape is the same as the query shape (Q)
    // This might need adjustment based on the final kernel logic (e.g., if it concatenates heads)
    if (inputs.empty()) {
         throw std::invalid_argument("[PagedAttentionPrimitive::output_shapes] Requires at least one input (query).");
    }
    // For the stub, output shape matches query shape
    return {inputs[0].shape()};
}

} // namespace pal::cpp
