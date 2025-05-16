#pragma once
// paged_attention_primitive.hpp
// Defines the PagedAttentionPrimitive class for MLX paged attention operations.
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

#include <mlx/array.h>
#include <mlx/primitives.h>
#include <mlx/stream.h>
#include <mlx/utils.h>
#include <mlx/backend/metal/device.h>
#include <mlx/backend/metal/metal.h>
#include <mlx/backend/metal/utils.h>

#include <optional>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>
#include <limits>

// Include the PagedAttentionParams struct definition
#include "shaders/paged_attention_types.h"

namespace mx = mlx::core;

namespace pal::cpp {

struct CoreDims {
    uint32_t head_dim{0};
    uint32_t num_q_heads{0};
    uint32_t tokens_per_page{0};
    uint32_t num_kv_heads{0};
    size_t num_items_to_process{0};
    size_t query_token_count{0};
};

struct ThreadgroupMemoryLayout {
    size_t q_shmem_bytes{0};
    size_t partial_reduce_scratch_bytes{0};
    size_t simd_reduced_maxes_bytes{0};
    size_t simd_reduced_adjusted_sum_exps_bytes{0};
    size_t global_stats_bytes{0};
    size_t s_global_compensation_bytes{0};
    size_t simd_v_chunk_sums_bytes{0};
    size_t k_tile_bytes{0}; // For caching K-vectors in threadgroup memory
    size_t v_tile_bytes{0}; // For caching V-vectors in threadgroup memory
    size_t final_guard_bytes{0};
    size_t total_bytes{0};

    size_t fixed_components_sum_for_t_calc() const {
        // Sum of q_shmem + all 6 scratch arrays (everything except k_tile, v_tile and final_guard)
        // K_tile_bytes and v_tile_bytes are removed from here as they depend on tile_T (creating a circular dependency)
        return q_shmem_bytes +
               partial_reduce_scratch_bytes +
               simd_reduced_maxes_bytes +
               simd_reduced_adjusted_sum_exps_bytes +
               global_stats_bytes +
               s_global_compensation_bytes +
               simd_v_chunk_sums_bytes;
    }
};

// Expected size for PagedAttentionParams: 10 uint32_t (40 bytes) + 1 float (4 bytes) = 44 bytes.
// alignas(16) means total size is 48, as it's padded to multiple of 16.
// Note: We use 64-byte alignment for threadgroup memory, but the struct itself remains 16-byte aligned.
constexpr size_t kExpectedPagedAttentionParamsSize = 48;
static_assert(
    sizeof(PagedAttentionParams) == kExpectedPagedAttentionParamsSize,
    "sizeof(PagedAttentionParams) mismatch between C++ and expected size (48 bytes). "
    "Check paged_attention_types.h, members, and padding.");

// Constants for memory padding and alignment
constexpr size_t kFinalTgMemoryPaddingGuardBytes = 32;
constexpr size_t kScoreTilePaddingFloatsPerSimdGroup = 8;
constexpr size_t kAlignmentBytes = 64;
constexpr size_t kAlignmentMask = kAlignmentBytes - 1;

// Constants for head_dim validation and processing
constexpr uint32_t kMaxAccumulationTile = 64;
constexpr uint32_t kMaxHeadDimMetalInKernel = 256; // Match Metal's kMaxHeadDimMetal
constexpr uint32_t kMaxSimdGroupsPerThreadgroup = 8;
constexpr uint32_t kDefaultAccTileChunkSizeInCPP = 64; // Must match Metal's kDefaultAccTileChunkSize

/**
 * @brief Custom primitive implementation for paged attention operations.
 *
 * This class implements a primitive for paged attention computation in the MLX
 * framework. It manages both CPU and GPU implementations of the paged attention
 * algorithm, handling the lifecycle and execution of the operation within the
 * computational graph.
 */
class PagedAttentionPrimitive : public mx::UnaryPrimitive {
 public:
  /**
   * @brief Constructs a PagedAttentionPrimitive with specified parameters.
   *
   * @param stream The MLX stream or device to execute on
   * @param num_q_heads Number of query heads in the attention mechanism
   * @param num_kv_heads Number of key/value heads in the attention mechanism
   * @param head_dim Hidden dimension size per attention head
   * @param tokens_per_page Number of tokens stored in each memory page
   */
  explicit PagedAttentionPrimitive(mx::StreamOrDevice stream,
                                   int num_q_heads = 0,
                                   int num_kv_heads = 0,
                                   int head_dim = 0,
                                   int tokens_per_page = 0);

  /**
   * @brief Evaluates the primitive on CPU.
   *
   * @param inputs Vector of input arrays
   * @param out Output array to store the result
   */
  void eval_cpu(const std::vector<mx::array>& inputs, mx::array& out) override;

  /**
   * @brief Evaluates the primitive on GPU.
   *
   * Launches the Metal kernel to perform paged attention computation.
   *
   * @param inputs Vector of input arrays
   * @param out Output array to store the result
   */
  void eval_gpu(const std::vector<mx::array>& inputs, mx::array& out) override;

  /**
   * @brief Prints a representation of the primitive to the output stream.
   *
   * @param os The output stream to print to
   */
  void print(std::ostream& os) override;

  /**
   * @brief Checks if this primitive is equivalent to another.
   *
   * Compares parameters that affect computation behavior.
   *
   * @param other The other primitive to compare with
   * @return true if primitives are equivalent, false otherwise
   */
  bool is_equivalent(const mx::Primitive& other) const override;

  /**
   * @brief Calculates output shapes based on input shapes.
   *
   * @param inputs Vector of input arrays
   * @return Vector of output shapes
   */
  std::vector<mx::Shape> output_shapes(
      const std::vector<mx::array>& inputs) override;

 private:
  // Parameters that define kernel behavior
  int num_q_heads_;
  int num_kv_heads_;
  int head_dim_;
  int tokens_per_page_;

  // Helper method declarations
  static CoreDims validate_inputs_and_populate_initial_params(
      const std::vector<mx::array>& inputs,
      int primitive_tokens_per_page // To access PagedAttentionPrimitive's construction param
  );

  static void populate_remaining_attention_params(
      PagedAttentionParams& params, // Pass by ref to populate (already has core dims from previous step)
      const CoreDims& extracted_core_dims, // Dimensions from previous helper
      const mx::array& k_pool_arr, // For num_physical_pages_in_pool
      const mx::array& page_table_arr, // For max_logical_blocks_per_seq, num_sequences_in_batch
      MTL::Device* mtl_device_ptr, // For maxThreadgroupMemoryLength
      size_t threads_per_item_group_for_dispatch // For num_simd_groups in tile_size_T calc
  );

  static ThreadgroupMemoryLayout calculate_threadgroup_memory_breakdown_and_total(
      const PagedAttentionParams& params,
      size_t threads_per_group
  );

  /**
   * @brief Implements vector-Jacobian product for backpropagation.
   *
   * @param primals Original input arrays
   * @param cotangents Gradients of the output
   * @param argnums Indices of arguments to compute gradients for
   * @param outputs Original outputs from the forward pass
   * @return Vector of gradients for the requested inputs
   */
  std::vector<mx::array> vjp(const std::vector<mx::array>& primals,
                             const std::vector<mx::array>& cotangents,
                             const std::vector<int>& argnums,
                             const std::vector<mx::array>& outputs) override;

  /**
   * @brief Implements Jacobian-vector product for forward differentiation.
   *
   * @param primals Original input arrays
   * @param tangents Directional derivatives of inputs
   * @param argnums Indices of arguments to compute derivatives for
   * @return Vector of directional derivatives of outputs
   */
  std::vector<mx::array> jvp(const std::vector<mx::array>& primals,
                             const std::vector<mx::array>& tangents,
                             const std::vector<int>& argnums) override;

  /**
   * @brief Implements vectorized mapping for batched execution.
   *
   * @param inputs Input arrays
   * @param axes Axes along which to vectorize
   * @return Pair of output arrays and corresponding axes
   */
  std::pair<std::vector<mx::array>, std::vector<int>> vmap(
      const std::vector<mx::array>& inputs,
      const std::vector<int>& axes) override;

  /**
   * @brief Validates inputs and extracts initial parameters for the attention operation.
   *
   * @param inputs Vector of input arrays
   * @param params Out-parameter to populate core dimensions
   * @param primitive_num_q_heads_member Value from this->num_q_heads_
   * @param primitive_head_dim_member Value from this->head_dim_
   * @param primitive_tokens_per_page_member Value from this->tokens_per_page_
   */
  static inline void validate_and_extract_initial_params(
      const std::vector<mx::array>& inputs,
      PagedAttentionParams& params,
      int primitive_num_q_heads_member,
      int primitive_head_dim_member,
      int primitive_tokens_per_page_member) {
    // Validate input count
    if (inputs.size() != 7) {
      throw std::invalid_argument(
          "[PagedAttentionPrimitive::eval_gpu] Expected 7 inputs (Q, K_pool, "
          "V_pool, Page_table, SeqLens, QToSeqMap, QOffsets), received " +
          std::to_string(inputs.size()));
    }

    // Extract inputs
    const auto& q = inputs[0];
    const auto& k_pool = inputs[1];
    const auto& v_pool = inputs[2];
    const auto& page_table = inputs[3];
    const auto& sequence_lengths = inputs[4];
    const auto& query_to_seq_map = inputs[5];
    const auto& query_token_offset = inputs[6];

    // Type checks for inputs
    if (q.dtype() != mx::float16 || k_pool.dtype() != mx::float16 ||
        v_pool.dtype() != mx::float16) {
      throw std::invalid_argument(
          "[PagedAttentionPrimitive] Q, K, and V inputs must be of type float16.");
    }
    if (page_table.dtype() != mx::uint32) {
      throw std::invalid_argument(
          "[PagedAttentionPrimitive] page_table input must be of type uint32.");
    }
    if (sequence_lengths.dtype() != mx::int32 ||
        query_to_seq_map.dtype() != mx::int32 ||
        query_token_offset.dtype() != mx::int32) {
      throw std::invalid_argument(
          "[PagedAttentionPrimitive] sequence_lengths, query_to_seq_map, and query_token_offset inputs must be of type int32.");
    }

    // Validate K/V pool geometry
    if (k_pool.ndim() != 4) {  // Expecting [NumPhysPages, TokensPerPage, NumKVHeads, HeadDim]
      throw std::invalid_argument("[PagedAttentionPrimitive] k_pool must be 4D.");
    }

    // Extract and validate tokens_per_page from KV pool
    int tokens_per_page_from_k_pool = k_pool.shape(1);
    if (primitive_tokens_per_page_member > 0 &&
        primitive_tokens_per_page_member != tokens_per_page_from_k_pool) {
      std::string error_msg =
          "[PagedAttentionPrimitive] Mismatch: tokens_per_page at construction (" +
          std::to_string(primitive_tokens_per_page_member) +
          ") does not match k_pool.shape(1) (" +
          std::to_string(tokens_per_page_from_k_pool) + ")";
      throw std::invalid_argument(error_msg);
    }

    // Extract parameters from KV pool shape
    params.tokens_per_page = tokens_per_page_from_k_pool;
    params.num_kv_heads = k_pool.shape(2);
    params.head_dim = k_pool.shape(3);

    // Validate head_dim
    if (params.head_dim == 0) {
      throw std::invalid_argument(
          "[PagedAttentionPrimitive] head_dim cannot be 0.");
    }
    if (params.head_dim % 4 != 0) {
      throw std::invalid_argument(
          "[PagedAttentionPrimitive] head_dim (" +
          std::to_string(params.head_dim) +
          ") must be a multiple of 4 for vectorized kernel execution.");
    }
    if (params.head_dim > kMaxHeadDimMetalInKernel) {
      throw std::invalid_argument(
          "[PagedAttentionPrimitive] params.head_dim (" +
          std::to_string(params.head_dim) +
          ") exceeds kernel's internal processing limit kMaxHeadDimMetal (" +
          std::to_string(kMaxHeadDimMetalInKernel) + ").");
    }
    if (params.head_dim > kMaxAccumulationTile * 1024) {
      throw std::invalid_argument(
          "[PagedAttentionPrimitive] params.head_dim (" +
          std::to_string(params.head_dim) +
          ") is excessively large for the tiled kernel approach.");
    }

    // Validate query dimensions
    if (q.ndim() < 1) {
      throw std::invalid_argument(
          "Queries 'q' must have at least 1 dimension.");
    }

    // Check query format and set num_q_heads
    if (q.ndim() == 3) {
      if (q.shape(2) != params.head_dim) {
        throw std::invalid_argument(
            "[PagedAttentionPrimitive] For 3D query input [NumTokens, NumQHeads, "
            "HeadDim], the HeadDim must match K/V head_dim.");
      }
      params.num_q_heads = q.shape(1);
    } else if (q.ndim() == 2) {
      if (q.shape(1) != params.head_dim) {
        throw std::invalid_argument(
            "[PagedAttentionPrimitive] For 2D query input [NumDispatchThreads, "
            "HeadDim], the HeadDim must match K/V head_dim.");
      }
      params.num_q_heads = 1;
    } else if (q.ndim() == 1) {
      if (params.head_dim != 1) {
        throw std::invalid_argument(
            "[PagedAttentionPrimitive] For 1D query input (interpreted as scalar "
            "items), the K/V head_dim (params.head_dim = " +
            std::to_string(params.head_dim) +
            ") must also be 1. The kernel will attempt to read head_dim elements "
            "for Q.");
      }
      params.num_q_heads = 1;  // Each item is effectively its own "Q-head" of size 1.
    } else {
      throw std::invalid_argument(
          "[PagedAttentionPrimitive] Query 'q' ndim not supported.");
    }

    // Validate page table and extract logical blocks parameters
    if (page_table.ndim() != 2) {
      throw std::invalid_argument(
          "[PagedAttentionPrimitive] page_table must be 2D [NumBatchSeq, "
          "MaxLogBlocksPerSeq].");
    }
    params.max_logical_blocks_per_seq = page_table.shape(1);

    // Set pool geometry parameters
    params.num_physical_pages_in_pool = k_pool.shape(0);
    params.num_sequences_in_batch = page_table.shape(0);

    // Validate Grouped Query Attention (GQA) parameters
    if (params.num_q_heads > params.num_kv_heads) {  // GQA case
      if (params.num_kv_heads == 0) {                // Avoid division by zero
        throw std::invalid_argument(
            "[PagedAttentionPrimitive] num_kv_heads cannot be 0 if num_q_heads > "
            "0 for GQA.");
      }
      if (params.num_q_heads % params.num_kv_heads != 0) {
        throw std::invalid_argument(
            "[PagedAttentionPrimitive] For GQA (num_q_heads > num_kv_heads), "
            "num_q_heads must be an integer multiple of num_kv_heads.");
      }
    }
  }

  /**
   * @brief Populates runtime-dependent attention parameters.
   *
   * @param params In/Out parameter struct to populate
   * @param inputs Vector of input arrays to access k_pool, page_table, etc.
   * @param current_threads_per_group Calculated in eval_gpu
   * @param mtl_device Metal device pointer for maxThreadgroupMemoryLength
   */
  static inline void populate_runtime_attention_params(
      PagedAttentionParams& params,
      const std::vector<mx::array>& inputs,
      size_t current_threads_per_group,
      MTL::Device* mtl_device);

  /**
   * @brief Calculates the total threadgroup memory required.
   *
   * @param params The fully populated parameters struct
   * @param current_threads_per_group Number of threads per threadgroup
   * @return Size in bytes needed for all threadgroup memory
   */
  static inline size_t calculate_total_threadgroup_memory(
      const PagedAttentionParams& params,
      size_t current_threads_per_group);

  /**
   * @brief Configures and dispatches the Metal compute kernel.
   *
   * @param compute_encoder Metal compute command encoder
   * @param kernel_pso Metal compute pipeline state
   * @param kernel_inputs Vector of input arrays (Q, K_pool, ...)
   * @param kernel_out_array Output array
   * @param kernel_params Parameters struct for the kernel
   * @param total_tg_memory_bytes Total threadgroup memory size in bytes
   * @param items_to_process_count Number of items to process
   * @param threads_per_group_count Number of threads per threadgroup
   */
  static inline void dispatch_metal_kernel(
      mlx::core::metal::CommandEncoder& compute_encoder,
      MTL::ComputePipelineState* kernel_pso,
      const std::vector<mx::array>& kernel_inputs,
      mx::array& kernel_out_array,
      const PagedAttentionParams& kernel_params,
      size_t total_tg_memory_bytes,
      size_t items_to_process_count,
      size_t threads_per_group_count);
};

}  // namespace pal::cpp
