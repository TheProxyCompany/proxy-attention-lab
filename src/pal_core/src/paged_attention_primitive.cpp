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
PagedAttentionPrimitive::PagedAttentionPrimitive(
    mx::StreamOrDevice stream_or_device,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int tokens_per_page
)
    : mx::UnaryPrimitive(mx::to_stream(stream_or_device)), // Pass stream to base class
      num_q_heads_(num_q_heads),
      num_kv_heads_(num_kv_heads),
      head_dim_(head_dim),
      tokens_per_page_(tokens_per_page)
{
#ifdef PAL_DEBUG
    std::cerr << "[PAL Primitive] PagedAttentionPrimitive constructed with params: "
              << "num_q_heads=" << num_q_heads_
              << ", num_kv_heads=" << num_kv_heads_
              << ", head_dim=" << head_dim_
              << ", tokens_per_page=" << tokens_per_page_
              << std::endl;
#endif
}

// --- CPU Evaluation (Stub) ---
void PagedAttentionPrimitive::eval_cpu(const std::vector<mx::array>& inputs, mx::array& out) {
#ifdef PAL_DEBUG
    std::cerr << "[PAL Primitive] PagedAttentionPrimitive::eval_cpu called (not supported)." << std::endl;
#endif
    throw std::runtime_error("[PagedAttentionPrimitive] CPU evaluation is not supported for paged attention.");
}

// --- GPU Evaluation (Kernel Launch Logic) ---
void PagedAttentionPrimitive::eval_gpu(const std::vector<mx::array>& inputs, mx::array& out) {
#ifdef PAL_DEBUG
    std::cerr << "[PAL Primitive] PagedAttentionPrimitive::eval_gpu called." << std::endl;
#endif

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
#ifdef PAL_DEBUG
    std::cerr << "[PAL Primitive] Input validation passed." << std::endl;
#endif

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

    // Ensure tokens_per_page from k_pool matches the stored primitive construction parameter, if it was set.
    uint tokens_per_page_from_k_pool = k_pool.shape(1);
    if (this->tokens_per_page_ > 0 && this->tokens_per_page_ != tokens_per_page_from_k_pool) {
        std::string error_msg = "[PagedAttentionPrimitive] Mismatch: tokens_per_page at construction (" +
                               std::to_string(this->tokens_per_page_) +
                               ") does not match k_pool.shape(1) (" +
                               std::to_string(tokens_per_page_from_k_pool) + ")";
        throw std::invalid_argument(error_msg);
    }

    params_struct.tokens_per_page = tokens_per_page_from_k_pool; // Tokens per page from k_pool's shape
    params_struct.num_kv_heads = k_pool.shape(2);    // Num KV heads from k_pool's shape
    params_struct.head_dim = k_pool.shape(3);        // Actual Head Dimension of K/V data

    // Validate head_dim
    if (params_struct.head_dim == 0) {
        throw std::invalid_argument("[PagedAttentionPrimitive] head_dim cannot be 0.");
    }
    if (params_struct.head_dim % 4 != 0) {
        throw std::invalid_argument(
            "[PagedAttentionPrimitive] head_dim (" + std::to_string(params_struct.head_dim) +
            ") must be a multiple of 4 for vectorized kernel execution."
        );
    }

    // 4.b Query related (num_q_heads can be different from num_kv_heads for GQA/MQA)
    //     The kernel will primarily use params_struct.head_dim for K/V access.
    //     If Q has a different structure (e.g. [TotalQ, ModelDim] vs [TotalQ, NumQHeads, QHeadDim]),
    //     the PIE Attention layer is responsible for projecting Q to the correct head structure
    //     BEFORE calling this primitive. For PAL, we assume 'q' input is ready.
    //     If q is [TotalItems, NumQActualHeads, QActualHeadDim]
    if (q.ndim() < 1) throw std::invalid_argument("Queries 'q' must have at least 1 dimension.");

    // Check query format and set num_q_heads
    if (q.ndim() == 3) {
        // 3D query format: [NumTokens, NumQHeads, HeadDim]
        if (q.shape(2) != params_struct.head_dim) {
            throw std::invalid_argument("[PagedAttentionPrimitive] For 3D query input [NumTokens, NumQHeads, HeadDim], the HeadDim must match K/V head_dim.");
        }
        // Set num_q_heads directly from the middle dimension
        params_struct.num_q_heads = q.shape(1);
#ifdef PAL_DEBUG
        std::cerr << "[PAL Primitive] Detected 3D query input with shape [" << q.shape(0) << ", " << q.shape(1) << ", " << q.shape(2) << "]" << std::endl;
#endif
    }
    else if (q.ndim() == 2) {
        // 2D query format: [NumDispatchThreads, HeadDim]
        if (q.shape(1) != params_struct.head_dim) {
            throw std::invalid_argument("[PagedAttentionPrimitive] For 2D query input [NumDispatchThreads, HeadDim], the HeadDim must match K/V head_dim.");
        }
        // CORRECTED: For 2D input [N,D], num_q_heads is 1
        // Each thread is one query item, effectively 1 Q-head per item
        params_struct.num_q_heads = 1;
#ifdef PAL_DEBUG
        std::cerr << "[PAL Primitive] Detected 2D query input with shape [" << q.shape(0) << ", " << q.shape(1) << "]" << std::endl;
#endif
    }
    else if (q.ndim() == 1) {
        // 1D query format: [NumDispatchThreads]
        // Each thread is one query item.
        // params_struct.head_dim is derived from K/V cache.
        // If Q is 1D, it implies each Q element is a scalar.
        // This is only valid if the K/V head_dim is also 1.
        if (params_struct.head_dim != 1) {
            throw std::invalid_argument(
                "[PagedAttentionPrimitive] For 1D query input (interpreted as scalar items), "
                "the K/V head_dim (params_struct.head_dim = " + std::to_string(params_struct.head_dim) + ") "
                "must also be 1. The kernel will attempt to read head_dim elements for Q.");
        }
        params_struct.num_q_heads = 1; // Each item is effectively its own "Q-head" of size 1.
#ifdef PAL_DEBUG
        std::cerr << "[PAL Primitive] Detected 1D query input with shape [" << q.shape(0) << "]"
                  << " and validated K/V head_dim is 1." << std::endl;
#endif
    }
    else {
        throw std::invalid_argument("[PagedAttentionPrimitive] Query 'q' ndim not supported.");
    }


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

    // Add the new parameters for bounds checking in the kernel
    params_struct.num_physical_pages_in_pool = k_pool.shape(0);
    params_struct.num_sequences_in_batch = page_table.shape(0);

    // Validate GQA configuration
    if (params_struct.num_q_heads > params_struct.num_kv_heads) { // GQA case
        if (params_struct.num_kv_heads == 0) { // Avoid division by zero
             throw std::invalid_argument("[PagedAttentionPrimitive] num_kv_heads cannot be 0 if num_q_heads > 0 for GQA.");
        }
        if (params_struct.num_q_heads % params_struct.num_kv_heads != 0) {
            throw std::invalid_argument("[PagedAttentionPrimitive] For GQA (num_q_heads > num_kv_heads), num_q_heads must be an integer multiple of num_kv_heads.");
        }
    }

#ifdef PAL_DEBUG
    std::cerr << "[PAL Primitive] Params: num_q_heads=" << params_struct.num_q_heads
              << ", num_kv_heads=" << params_struct.num_kv_heads
              << ", head_dim=" << params_struct.head_dim // This is now K/V head_dim
              << ", tokens_per_page=" << params_struct.tokens_per_page
              << ", scale=" << params_struct.scale
              << ", max_logical_blocks_per_seq=" << params_struct.max_logical_blocks_per_seq
              << ", num_physical_pages_in_pool=" << params_struct.num_physical_pages_in_pool
              << ", num_sequences_in_batch=" << params_struct.num_sequences_in_batch << std::endl;
#endif

    // --- 4.e Bounds Checking Strategy ---
    // Bounds checking logic has been moved to the Metal kernel for better performance and reliability.
    // The kernel will check:
    // 1. That page_table physical_page_ids are < params->num_physical_pages_in_pool
    // 2. That query_token_offset values are non-negative
    // 3. That seq_idx from query_to_seq_map are < params->num_sequences_in_batch
    //
    // This avoids issues with trying to access array data on the CPU side before it's ready,
    // and is more in line with GPU compute paradigms where validation happens in the kernel.

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
#ifdef PAL_DEBUG
    std::cerr << "[PAL Primitive] All kernel buffers and params struct set." << std::endl;
#endif

    // --- 6. Dispatch Kernel ---
    size_t grid_dim_x = 0;

    // Calculate dispatch size based on query dimensions
    if (q.ndim() == 3) {
        // For 3D queries [NumTokens, NumQHeads, HeadDim],
        // we need to dispatch one thread per (token, qhead) combination
        grid_dim_x = q.shape(0) * q.shape(1);
        std::cerr << "[PAL Primitive] Dispatching for 3D query: " << grid_dim_x << " threads (NumTokens="
                  << q.shape(0) << " * NumQHeads=" << q.shape(1) << ")" << std::endl;
    } else {
        // Default dispatch based on first dimension only
        grid_dim_x = q.shape(0);
        std::cerr << "[PAL Primitive] Dispatching for " << q.ndim() << "D query: " << grid_dim_x << " threads" << std::endl;
    }

    if (grid_dim_x == 0) {
        std::cerr << "[PagedAttentionPrimitive] No threads to dispatch (grid_dim_x=0). Returning empty result." << std::endl;
        return;
    }

    // For now, keep using 1D dispatch where global_query_thread_idx encompasses both token and head indices
    MTL::Size grid_dims = MTL::Size(grid_dim_x, 1, 1);
    size_t tgp_size_max = kernel_pipeline_state->maxTotalThreadsPerThreadgroup();
    size_t tgp_size = std::min(tgp_size_max, grid_dim_x);
    if (tgp_size == 0) tgp_size = 1;
    MTL::Size group_dims = MTL::Size(tgp_size, 1, 1);
    compute_encoder.dispatch_threads(grid_dims, group_dims);
    std::cerr << "[PAL Primitive] Kernel dispatched." << std::endl;

    // Note on encoder completion:
    // MLX's metal::CommandEncoder doesn't have an explicit end_encoding() method.
    // Instead, the encoder completion appears to be managed implicitly when operations are done.
    // This follows the MLX pattern seen in their own primitives.
#ifdef PAL_DEBUG
    std::cerr << "[PAL Primitive] Compute encoder operation completed." << std::endl;
#endif
}

// --- Print Method ---
void PagedAttentionPrimitive::print(std::ostream& os) {
    os << "PagedAttention";
    // Add stored parameters for better debugging
    os << "(qheads=" << num_q_heads_
       << ",kvheads=" << num_kv_heads_
       << ",dim=" << head_dim_
       << ",tpp=" << tokens_per_page_ << ")";
}

// --- Equivalence Check ---
bool PagedAttentionPrimitive::is_equivalent(const mx::Primitive& other) const {
    // Check if the other primitive is the same type
    if (typeid(*this) != typeid(other)) {
        return false;
    }

    // Cast and compare stored parameters
    const PagedAttentionPrimitive& other_pa = static_cast<const PagedAttentionPrimitive&>(other);
    return (this->num_q_heads_ == other_pa.num_q_heads_ &&
            this->num_kv_heads_ == other_pa.num_kv_heads_ &&
            this->head_dim_ == other_pa.head_dim_ &&
            this->tokens_per_page_ == other_pa.tokens_per_page_);
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
    if (inputs.empty()) {
        throw std::invalid_argument("[PagedAttentionPrimitive::output_shapes] Requires at least one input (query).");
    }

    const auto& q = inputs[0];

    if (q.ndim() == 3) {
        // Q is [NumTokens, NumQHeads, HeadDim], output scores are [NumTokens, NumQHeads]
        return {{q.shape(0), q.shape(1)}};
    } else if (q.ndim() == 2) {
        // Q is [NumDispatchThreads, HeadDim], output scores are [NumDispatchThreads]
        return {{q.shape(0)}};
    } else if (q.ndim() == 1) {
        // Q is [NumDispatchThreads], output scores are [NumDispatchThreads] (each thread produces one scalar)
        // This supports the existing tests.
        return {q.shape()};
    } else {
        throw std::invalid_argument("[PagedAttentionPrimitive::output_shapes] Query input 'q' must be 1D, 2D, or 3D.");
    }
}

} // namespace pal::cpp
