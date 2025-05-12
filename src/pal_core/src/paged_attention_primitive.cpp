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

    // (Dtype checks remain the same)
    if (q.dtype() != mx::float16 || k_pool.dtype() != mx::float16 || v_pool.dtype() != mx::float16) { /* ... */ }
    if (page_table.dtype() != mx::uint32) { /* ... */ }
    if (sequence_lengths.dtype() != mx::int32 || query_to_seq_map.dtype() != mx::int32 || query_token_offset.dtype() != mx::int32) { /* ... */ }
    std::cerr << "[PAL Primitive] Input validation passed." << std::endl;

    // --- 2. Allocate Output Buffer --- (no change)
    out.set_data(mx::allocator::malloc(out.nbytes()));

    // --- 3. Prepare Metal Kernel and Command Encoder --- (no change)
    auto& s = stream();
    auto& d = mlx::core::metal::device(s.device);
    const std::string library_name_for_mlx = "pal";
    const std::string kernel_name = "paged_attn_kernel";
    MTL::ComputePipelineState* kernel_pipeline_state = d.get_kernel(kernel_name, library_name_for_mlx); // Error handling already there
    auto& compute_encoder = d.get_command_encoder(s.index);
    compute_encoder.set_compute_pipeline_state(kernel_pipeline_state);

    // --- 4. Create and Populate PagedAttentionParams Struct (REVISED LOGIC) ---
    PagedAttentionParams params_struct;

    // 4.a K/V Pool Geometry - This defines the canonical head_dim for the cache
    if (k_pool.ndim() != 4) { // Expecting [NumPhysPages, TokensPerPage, NumKVHeads, HeadDim]
        throw std::invalid_argument("[PagedAttentionPrimitive] k_pool must be 4D.");
    }
    params_struct.tokens_per_page = k_pool.shape(1); // Tokens per page from k_pool's shape
    params_struct.num_kv_heads = k_pool.shape(2);    // Num KV heads from k_pool's shape
    params_struct.head_dim = k_pool.shape(3);        // Actual Head Dimension of K/V data

    // 4.b Query related (num_q_heads can be different from num_kv_heads for GQA/MQA)
    //     The kernel will primarily use params_struct.head_dim for K/V access.
    //     If Q has a different structure (e.g. [TotalQ, ModelDim] vs [TotalQ, NumQHeads, QHeadDim]),
    //     the PIE Attention layer is responsible for projecting Q to the correct head structure
    //     BEFORE calling this primitive. For PAL, we assume 'q' input is ready.
    //     If q is [TotalItems, NumQActualHeads, QActualHeadDim]
    if (q.ndim() < 1) throw std::invalid_argument("Queries 'q' must have at least 1 dimension.");
    if (q.ndim() >= 2 && q.shape(-1) != params_struct.head_dim) {
        // This check ensures that if Q is passed with an explicit head dimension, it matches the K/V head dimension.
        // If Q is, for example, [TotalTokens, ModelDim], then this check might not apply,
        // and num_q_heads might be considered 1 by the kernel if it processes Q token-wise.
        // For now, let's assume Q's innermost dim (if >1D) should match K/V head_dim.
        throw std::invalid_argument("[PagedAttentionPrimitive] Query's innermost dimension must match K/V head_dim if query is multi-dimensional representing heads.");
    }
    params_struct.num_q_heads = (q.ndim() >= 2 && q.shape(-1) == params_struct.head_dim) ? q.shape(q.ndim() - 2) : 1;


    // 4.c Scale
    if (params_struct.head_dim > 0) {
        params_struct.scale = 1.0f / sqrtf(static_cast<float>(params_struct.head_dim));
    } else {
        params_struct.scale = 1.0f; // Should not happen with valid inputs
    }

    // 4.d Page Table related
    //     page_table shape is expected to be [NumBatchSequences, MaxLogicalBlocksPerSequenceInTable]
    //     or [TotalLogicalBlocksAcrossAllSequences] if fully flattened with other indexing.
    //     The Python test passes it as 2D.
    if (page_table.ndim() != 2) {
         throw std::invalid_argument("[PagedAttentionPrimitive] page_table must be 2D [NumBatchSeq, MaxLogBlocksPerSeq].");
    }
    params_struct.max_logical_blocks_per_seq = page_table.shape(1);
    if (params_struct.max_logical_blocks_per_seq == 0 && page_table.size() > 0) {
        params_struct.max_logical_blocks_per_seq = 1; // Avoid division by zero if used as stride
    }

    std::cerr << "[PAL Primitive] Params: num_q_heads=" << params_struct.num_q_heads
              << ", num_kv_heads=" << params_struct.num_kv_heads
              << ", head_dim=" << params_struct.head_dim // This is now K/V head_dim
              << ", tokens_per_page=" << params_struct.tokens_per_page
              << ", scale=" << params_struct.scale
              << ", max_logical_blocks_per_seq=" << params_struct.max_logical_blocks_per_seq << std::endl;

    // --- 5. Set Kernel Arguments --- (no change in this section's structure)
    compute_encoder.set_input_array(q, 0);
    compute_encoder.set_input_array(k_pool, 1);
    compute_encoder.set_input_array(v_pool, 2);
    compute_encoder.set_input_array(page_table, 3);
    compute_encoder.set_input_array(sequence_lengths, 4);
    compute_encoder.set_input_array(query_to_seq_map, 5);
    compute_encoder.set_input_array(query_token_offset, 6);
    compute_encoder.set_bytes(&params_struct, sizeof(PagedAttentionParams), 7);
    compute_encoder.set_output_array(out, 8);
    std::cerr << "[PAL Primitive] All kernel buffers and params struct set." << std::endl;

    // --- 6. Dispatch Kernel ---
    size_t grid_dim_x = q.shape(0);

    if (grid_dim_x == 0) { /* ... */ return; }
    MTL::Size grid_dims = MTL::Size(grid_dim_x, 1, 1); // Assuming 1D dispatch for now
    size_t tgp_size_max = kernel_pipeline_state->maxTotalThreadsPerThreadgroup();
    size_t tgp_size = std::min(tgp_size_max, grid_dim_x);
    if (grid_dim_x > 0 && tgp_size == 0) tgp_size = 1;
    MTL::Size group_dims = MTL::Size(tgp_size, 1, 1);
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
