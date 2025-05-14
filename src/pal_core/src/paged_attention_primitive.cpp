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
#include <sstream>

#include "mlx/backend/common/utils.h"
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
    // Don't set bytes here, we'll set them once after all params are populated
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
    #ifdef PAL_DEBUG
    std::cerr << "[PAL Primitive] Device maxTotalThreadsPerThreadgroup: " << max_threads << std::endl;
    #endif
    const size_t threads_per_item_group = std::min(default_threads_per_item_group, max_threads);
    if (threads_per_item_group == 0) {
        throw std::runtime_error("[PagedAttentionPrimitive] Calculated threads_per_item_group is 0. Device maxTotalThreadsPerThreadgroup might be 0 or default is 0. This is invalid.");
    }
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

    // Use correctly named variables: threadgroups_per_grid for group_dims, threads_per_threadgroup for grid_dims
    MTL::Size threadgroups_per_grid = MTL::Size(num_items_to_process, 1, 1);
    MTL::Size threads_per_threadgroup = MTL::Size(threads_per_item_group, 1, 1);

    // Calculate the size of threadgroup memory needed
    constexpr uint32_t MAX_SIMD_GROUPS_PER_TG = 8;
    const uint32_t head_dim = params_struct.head_dim;

    // Memory layout:
    // 1. q_shmem: head_dim floats
    // 2. G_partial_max_scores: threads_per_item_group floats
    // 3. G_simd_reduced_maxes: MAX_SIMD_GROUPS_PER_TG floats
    // 4. G_simd_reduced_adjusted_sum_exps: MAX_SIMD_GROUPS_PER_TG floats
    // 5. G_final_max_for_item: 1 float
    // 6. G_final_sum_exp_for_item: 1 float
    // 7. G_V_reduction_scratch: MAX_SIMD_GROUPS_PER_TG floats (for V component reduction)
    size_t tg_memory_bytes = sizeof(float) * (
        head_dim +                    // q_shmem
        threads_per_item_group +      // G_partial_max_scores
        MAX_SIMD_GROUPS_PER_TG +      // G_simd_reduced_maxes
        MAX_SIMD_GROUPS_PER_TG +      // G_simd_reduced_adjusted_sum_exps
        1 +                           // G_final_max_for_item
        1 +                           // G_final_sum_exp_for_item
        MAX_SIMD_GROUPS_PER_TG        // G_V_reduction_scratch (for one float component reduction)
    );

    // Check against device's maxThreadgroupMemoryLength
    MTL::Device* device = d.mtl_device();
    size_t max_tg_memory = device->maxThreadgroupMemoryLength();
    if (tg_memory_bytes > max_tg_memory) {
        throw std::invalid_argument(
            "[PagedAttentionPrimitive] Required threadgroup memory (" + std::to_string(tg_memory_bytes) +
            " bytes) exceeds device maximum (" + std::to_string(max_tg_memory) +
            " bytes). Try reducing head_dim or threads_per_item_group."
        );
    }

    // Set the threadgroup memory length at index 0 (matches [[threadgroup(0)]] in kernel)
    compute_encoder.set_threadgroup_memory_length(tg_memory_bytes, 0);

    compute_encoder.dispatch_threadgroups(threadgroups_per_grid, threads_per_threadgroup);
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
    if (inputs.size() < 2 || inputs[1].ndim() != 4) {
         throw std::invalid_argument("[PagedAttentionPrimitive::output_shapes] K-pool (inputs[1]) is needed and must be 4D to determine head_dim for output shape.");
    }

    const auto& q = inputs[0];
    uint32_t current_head_dim = inputs[1].shape(3); // Get head_dim from K-pool

    if (q.ndim() == 3) { // Q is [NumTokens, NumQHeads, QueryHeadDim]
        // Output will be [NumTokens * NumQHeads, AttentionOutputHeadDim]
        return {{ q.shape(0) * q.shape(1), static_cast<int>(current_head_dim) }};
    } else if (q.ndim() == 2) { // Q is [NumItems, QueryHeadDim]
        // Output will be [NumItems, AttentionOutputHeadDim]
        return {{ q.shape(0), static_cast<int>(current_head_dim) }};
    } else if (q.ndim() == 1) { // Q is [NumItems] (assuming QueryHeadDim=1 and AttentionOutputHeadDim=1)
         if (current_head_dim != 1) {
            // This case was problematic before; if head_dim is not 1, outputting [NumItems, head_dim]
            // might be unexpected if the input was truly scalar.
            // However, for consistency in output structure, we make it [NumItems, head_dim].
            // The C++ validation for 1D queries already ensures head_dim is 1.
         }
        return {{ q.shape(0), static_cast<int>(current_head_dim) }};
    } else {
        throw std::invalid_argument("[PagedAttentionPrimitive::output_shapes] Query input 'q' must be 1D, 2D, or 3D.");
    }
}

} // namespace pal::cpp
