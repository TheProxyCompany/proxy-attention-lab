#include "pal_core/paged_attention_primitive.hpp"

#include <mlx/array.h>
#include <mlx/allocator.h>
#include <mlx/backend/metal/metal.h>
#include <mlx/backend/metal/device.h>
#include <mlx/backend/metal/utils.h>
#include <mlx/backend/cpu/encoder.h>
#include "shaders/paged_attention_types.h"
#include <stdexcept>
#include <iostream>
#include <iostream>
#include <sstream>

#include "mlx/backend/common/utils.h"
#include "mlx/backend/cpu/encoder.h"
#include "mlx/utils.h"

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

    // --- 3. Prepare Metal Kernel and Command Encoder --- (no change)
    auto& s = stream();
    auto& d = mlx::core::metal::device(s.device);

    size_t bytes = out.nbytes();
    out.set_data(mx::allocator::malloc(bytes));

    const std::string library_name_for_mlx = "pal";
    const std::string kernel_name = "paged_attn_kernel";
    MTL::ComputePipelineState* kernel_pipeline_state = d.get_kernel(kernel_name, library_name_for_mlx);
    auto& compute_encoder = d.get_command_encoder(s.index);
    compute_encoder.set_compute_pipeline_state(kernel_pipeline_state);

    // --- 4. Create and Populate PagedAttentionParams Struct (REVISED LOGIC) ---
    PagedAttentionParams params_struct;

#ifdef PAL_DEBUG
    std::cerr << "[PAL DEBUG PARAMS] C++ sizeof(PagedAttentionParams): " << sizeof(PagedAttentionParams) << " bytes." << std::endl;
    std::cerr << "[PAL DEBUG PARAMS] params_struct ADDRESS: " << &params_struct << std::endl;
    std::cerr << "[PAL DEBUG PARAMS] params_struct.head_dim ADDRESS: " << &params_struct.head_dim << std::endl;
    std::cerr << "[PAL DEBUG PARAMS] params_struct.scale ADDRESS: " << &params_struct.scale << std::endl;
    std::cerr << "[PAL DEBUG PARAMS] offsetof(PagedAttentionParams, num_q_heads): " << offsetof(PagedAttentionParams, num_q_heads) << std::endl;
    std::cerr << "[PAL DEBUG PARAMS] offsetof(PagedAttentionParams, num_kv_heads): " << offsetof(PagedAttentionParams, num_kv_heads) << std::endl;
    std::cerr << "[PAL DEBUG PARAMS] offsetof(PagedAttentionParams, head_dim): " << offsetof(PagedAttentionParams, head_dim) << std::endl;
    std::cerr << "[PAL DEBUG PARAMS] offsetof(PagedAttentionParams, tokens_per_page): " << offsetof(PagedAttentionParams, tokens_per_page) << std::endl;
    std::cerr << "[PAL DEBUG PARAMS] offsetof(PagedAttentionParams, scale): " << offsetof(PagedAttentionParams, scale) << std::endl;
    std::cerr << "[PAL DEBUG PARAMS] offsetof(PagedAttentionParams, total_items_in_dispatch): " << offsetof(PagedAttentionParams, total_items_in_dispatch) << std::endl;
#endif

    // 4.a K/V Pool Geometry - This defines the canonical head_dim for the cache
    if (k_pool.ndim() != 4) { // Expecting [NumPhysPages, TokensPerPage, NumKVHeads, HeadDim]
        throw std::invalid_argument("[PagedAttentionPrimitive] k_pool must be 4D.");
    }

    int tokens_per_page_from_k_pool = k_pool.shape(1);
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

    if (q.ndim() < 1) throw std::invalid_argument("Queries 'q' must have at least 1 dimension.");

    // Check query format and set num_q_heads
    if (q.ndim() == 3) {
        if (q.shape(2) != params_struct.head_dim) {
            throw std::invalid_argument("[PagedAttentionPrimitive] For 3D query input [NumTokens, NumQHeads, HeadDim], the HeadDim must match K/V head_dim.");
        }
        params_struct.num_q_heads = q.shape(1);
    }
    else if (q.ndim() == 2) {
        if (q.shape(1) != params_struct.head_dim) {
            throw std::invalid_argument("[PagedAttentionPrimitive] For 2D query input [NumDispatchThreads, HeadDim], the HeadDim must match K/V head_dim.");
        }
        params_struct.num_q_heads = 1;
    }
    else if (q.ndim() == 1) {
        if (params_struct.head_dim != 1) {
            throw std::invalid_argument(
                "[PagedAttentionPrimitive] For 1D query input (interpreted as scalar items), "
                "the K/V head_dim (params_struct.head_dim = " + std::to_string(params_struct.head_dim) + ") "
                "must also be 1. The kernel will attempt to read head_dim elements for Q.");
        }
        params_struct.num_q_heads = 1; // Each item is effectively its own "Q-head" of size 1.
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

    if (page_table.ndim() != 2) {
        throw std::invalid_argument("[PagedAttentionPrimitive] page_table must be 2D [NumBatchSeq, MaxLogBlocksPerSeq].");
    }
    params_struct.max_logical_blocks_per_seq = page_table.shape(1);
    if (params_struct.max_logical_blocks_per_seq == 0 && page_table.size() > 0) {
        params_struct.max_logical_blocks_per_seq = 1; // Avoid division by zero if used as stride
    }

    params_struct.num_physical_pages_in_pool = k_pool.shape(0);
    params_struct.num_sequences_in_batch = page_table.shape(0);

    if (params_struct.num_q_heads > params_struct.num_kv_heads) { // GQA case
        if (params_struct.num_kv_heads == 0) { // Avoid division by zero
             throw std::invalid_argument("[PagedAttentionPrimitive] num_kv_heads cannot be 0 if num_q_heads > 0 for GQA.");
        }
        if (params_struct.num_q_heads % params_struct.num_kv_heads != 0) {
            throw std::invalid_argument("[PagedAttentionPrimitive] For GQA (num_q_heads > num_kv_heads), num_q_heads must be an integer multiple of num_kv_heads.");
        }
    }

    compute_encoder.set_input_array(q, 0);
    compute_encoder.set_input_array(k_pool, 1);
    compute_encoder.set_input_array(v_pool, 2);
    compute_encoder.set_input_array(page_table, 3);
    compute_encoder.set_input_array(sequence_lengths, 4);
    compute_encoder.set_input_array(query_to_seq_map, 5);
    compute_encoder.set_input_array(query_token_offset, 6);
    compute_encoder.set_bytes(&params_struct, sizeof(PagedAttentionParams), 7);
    compute_encoder.set_output_array(out, 8);

    // --- 6. Dispatch Kernel ---
    size_t num_items_to_process = 0; /* Calculate as before based on q */
    if (q.ndim() == 3) { num_items_to_process = q.shape(0) * q.shape(1); }
    else { num_items_to_process = q.shape(0); }
    if (num_items_to_process == 0) { return; }

    if (num_items_to_process == 0) {
        std::cerr << "[PagedAttentionPrimitive] No items to process (num_items_to_process=0). Returning empty result." << std::endl;
        return;
    }

    // Size sanity checks for side-band arrays
    // For 3D queries [NumTokens, NumQHeads, HeadDim], we need to handle the special case
    // where query_to_seq_map size might only match the NumTokens dimension
    size_t expected_size = q.ndim() == 3 ? q.shape(0) : num_items_to_process;

    if (query_to_seq_map.size() != expected_size) {
        throw std::invalid_argument("[PagedAttentionPrimitive] query_to_seq_map size must match number of tokens (or items for 1D/2D queries)");
    }
    if (query_token_offset.size() != expected_size) {
        throw std::invalid_argument("[PagedAttentionPrimitive] query_token_offset size must match number of tokens (or items for 1D/2D queries)");
    }

    // Set total_items_in_dispatch for planar output layout
    params_struct.total_items_in_dispatch = static_cast<uint32_t>(num_items_to_process);

    // DEBUG output for catching issues with parameter values
    std::cerr << "[PAL DEBUG PARAMS] C++ sizeof(PagedAttentionParams): " << sizeof(PagedAttentionParams) << " bytes." << std::endl;
    std::cerr << "[PAL DEBUG PARAMS] C++ offsetof(num_q_heads): " << offsetof(PagedAttentionParams, num_q_heads) << std::endl;
    std::cerr << "[PAL DEBUG PARAMS] C++ offsetof(num_kv_heads): " << offsetof(PagedAttentionParams, num_kv_heads) << std::endl;
    std::cerr << "[PAL DEBUG PARAMS] C++ offsetof(head_dim): " << offsetof(PagedAttentionParams, head_dim) << std::endl;
    std::cerr << "[PAL DEBUG PARAMS] C++ offsetof(tokens_per_page): " << offsetof(PagedAttentionParams, tokens_per_page) << std::endl;
    std::cerr << "[PAL DEBUG PARAMS] C++ offsetof(scale): " << offsetof(PagedAttentionParams, scale) << std::endl;

    // Print actual values being sent
    std::cerr << "[PAL SENDING PARAMS] num_q_heads: " << params_struct.num_q_heads << std::endl;
    std::cerr << "[PAL SENDING PARAMS] num_kv_heads: " << params_struct.num_kv_heads << std::endl;
    std::cerr << "[PAL SENDING PARAMS] head_dim: " << params_struct.head_dim << std::endl;
    std::cerr << "[PAL SENDING PARAMS] tokens_per_page: " << params_struct.tokens_per_page << std::endl;
    std::cerr << "[PAL SENDING PARAMS] scale: " << params_struct.scale << std::endl;

    // Original debug values for compatibility
    std::cerr << "[PAL DEBUG PARAMS] max_logical_blocks_per_seq = " << params_struct.max_logical_blocks_per_seq << std::endl;
    std::cerr << "[PAL DEBUG PARAMS] num_physical_pages_in_pool = " << params_struct.num_physical_pages_in_pool << std::endl;
    std::cerr << "[PAL DEBUG PARAMS] num_sequences_in_batch = " << params_struct.num_sequences_in_batch << std::endl;
    std::cerr << "[PAL DEBUG PARAMS] total_items_in_dispatch = " << params_struct.total_items_in_dispatch << std::endl;
    std::cerr << "[PAL DEBUG PARAMS] actual_threads_per_item_group = " << params_struct.actual_threads_per_item_group << std::endl;

    // Print the memory addresses of key fields
    std::cerr << "[PAL DEBUG PARAMS] params_struct ADDRESS: " << &params_struct << std::endl;
    std::cerr << "[PAL DEBUG PARAMS] params_struct.head_dim ADDRESS: " << &params_struct.head_dim << std::endl;
    std::cerr << "[PAL DEBUG PARAMS] params_struct.scale ADDRESS: " << &params_struct.scale << std::endl;

    const size_t default_threads_per_item_group = 64;
    auto max_threads = kernel_pipeline_state->maxTotalThreadsPerThreadgroup();
    const size_t threads_per_item_group = std::min(default_threads_per_item_group, max_threads);
    // No longer throw if max_threads < default_threads_per_item_group
    // The kernel will adapt to the actual threads_per_item_group dispatched
    #ifdef PAL_DEBUG
    std::cerr << "[PAL Primitive] Effective threads_per_item_group for dispatch: " << threads_per_item_group << std::endl;
    #endif

    // Fill the whole struct first
    params_struct.actual_threads_per_item_group = static_cast<uint32_t>(threads_per_item_group);
    params_struct.total_items_in_dispatch = static_cast<uint32_t>(num_items_to_process);

    // NOW upload the complete struct
    compute_encoder.set_bytes(&params_struct, sizeof(PagedAttentionParams), 7);

    MTL::Size group_dims = MTL::Size(num_items_to_process, 1, 1);
    MTL::Size grid_dims = MTL::Size(threads_per_item_group, 1, 1);

    compute_encoder.dispatch_threadgroups(group_dims, grid_dims);
    // mx::synchronize(s);
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

    // With planar output layout, we return a 1D array of size [total_dispatch_threads * 2]
    // where the first plane [0:total_dispatch_threads] contains max_scores
    // and the second plane [total_dispatch_threads:2*total_dispatch_threads] contains sum_exp_scores
    size_t num_items_to_process = 0;

    if (q.ndim() == 3) {
        // Q is [NumTokens, NumQHeads, HeadDim]
        // Output is [NumTokens * NumQHeads * 2] (planar layout)
        num_items_to_process = q.shape(0) * q.shape(1);
        return {{static_cast<int>(num_items_to_process * 2)}};
    } else if (q.ndim() == 2) {
        // Q is [NumDispatchThreads, HeadDim]
        // Output is [NumDispatchThreads * 2] (planar layout)
        num_items_to_process = q.shape(0);
        return {{static_cast<int>(num_items_to_process * 2)}};
    } else if (q.ndim() == 1) {
        // Q is [NumDispatchThreads]
        // Output is [NumDispatchThreads * 2] (planar layout)
        num_items_to_process = q.shape(0);
        return {{static_cast<int>(num_items_to_process * 2)}};
    } else {
        throw std::invalid_argument("[PagedAttentionPrimitive::output_shapes] Query input 'q' must be 1D, 2D, or 3D.");
    }
}

} // namespace pal::cpp
