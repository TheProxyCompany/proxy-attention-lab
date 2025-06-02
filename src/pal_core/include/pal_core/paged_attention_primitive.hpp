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
#include <vector>
#include "kernels/paged_attention_types.h"

namespace mx = mlx::core;

namespace pal::cpp {

static constexpr size_t MIN_NEW_TOKENS_FOR_FUSED = 64; // wip

static constexpr size_t FUSED_SIMD_GROUPS_PER_THREADGROUP = 2; // wip
static constexpr uint32_t PASS1_SIMD_GROUPS_PER_GQA_GROUP = 6; // hand tuned; 4-6 seems to be the sweet spot
static constexpr uint32_t PASS2_SIMD_GROUPS_PER_THREADGROUP = 8; // hand tuned; 8 is the sweet spot

static constexpr float kLogFp16DenormMinVal = -88.0f;
static constexpr uint32_t PASS2_TOKEN_BLOCK_SIZE = 16;
static constexpr uint32_t PASS2_QHEAD_BLOCK_SIZE = 8;
static constexpr uint32_t MAX_TILE_SIZE_PRACTICAL = 256;

static constexpr size_t kMemoryAlignmentBytes = 64;
static constexpr size_t kMemoryAlignmentMask = kMemoryAlignmentBytes - 1;
static constexpr size_t kFinalMemoryPaddingGuardBytes = 32;

struct CoreDims {
    uint32_t head_dim{0};
    uint32_t num_q_heads{0};
    uint32_t tokens_per_page{0};
    uint32_t num_kv_heads{0};
    size_t num_items_to_process{0};
    size_t query_token_count{0};
};

struct AttentionMemoryLayout {
    size_t total_bytes{0};
    size_t q_shmem_bytes{0};
    size_t partial_reduce_scratch_bytes{0};
    size_t simd_reduced_maxes_bytes{0};
    size_t simd_reduced_adjusted_sum_exps_bytes{0};
    size_t global_stats_bytes{0};
    size_t s_global_compensation_bytes{0};
    size_t simd_v_chunk_sums_bytes{0};
    size_t k_tile_bytes{0};
    size_t v_tile_bytes{0};
    size_t page_table_slice_bytes{0};
    size_t final_guard_bytes{0};

    static size_t align_size(size_t size) {
        return (size + kMemoryAlignmentMask) & ~kMemoryAlignmentMask;
    }

    static AttentionMemoryLayout calculate_attention_memory_layout(
      const PagedAttentionParams& params,
      size_t threads_per_group,
      size_t actual_simd_lanes_per_group,
      bool use_2pass_kernel
    );
};

static CoreDims extract_dims(const std::vector<mx::array>& inputs);

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
   * @param use_fused_kernel Whether to use the fused kernel
   */
  explicit PagedAttentionPrimitive(
    mx::StreamOrDevice stream,
    int num_q_heads = 0,
    int num_kv_heads = 0,
    int head_dim = 0,
    int tokens_per_page = 0,
    bool use_fused_kernel = false
  );

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
  bool is_equivalent(const mx::Primitive& other) const override {
    // Check if the other primitive is the same type
    if (typeid(*this) != typeid(other)) {
      return false;
    }

    // Cast and compare stored parameters
    const PagedAttentionPrimitive& other_pa =
        static_cast<const PagedAttentionPrimitive&>(other);
    return (this->num_q_heads_ == other_pa.num_q_heads_ &&
            this->num_kv_heads_ == other_pa.num_kv_heads_ &&
            this->head_dim_ == other_pa.head_dim_ &&
            this->tokens_per_page_ == other_pa.tokens_per_page_);
  }

  /**
   * @brief Calculates output shapes based on input shapes.
   *
   * @param inputs Vector of input arrays
   * @return Vector of output shapes
   */
  std::vector<mx::Shape> output_shapes(const std::vector<mx::array>& inputs) override;

  /**
   * @brief Calculates the optimal tile size and thread information for the paged attention kernel.
   *
   * @param head_dimension The dimension of the head
   * @param num_query_heads The number of query heads
   * @param num_kv_heads The number of key/value heads
   * @param stream_or_device The stream or device to execute on
   * @param pipeline_state The pipeline state to use for the kernel
   * @return A tuple containing the optimal tile size, the number of threads per threadgroup, and the actual SIMD width
   */
  static std::tuple<uint32_t, uint32_t, uint32_t> get_optimal_tile_size_and_thread_info(
    uint32_t head_dimension,
    uint32_t num_query_heads,
    uint32_t num_kv_heads,
    mx::StreamOrDevice stream_or_device = {},
    std::optional<MTL::ComputePipelineState*> pipeline_state = std::nullopt
  );

 private:
  // Parameters that define kernel behavior
  int num_q_heads_;
  int num_kv_heads_;
  int head_dim_;
  int tokens_per_page_;
  bool use_fused_kernel_;

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

  // Helper methods for decode and prefill paths
  void _eval_gpu_fused(
    const mx::Stream& stream,
    mlx::core::metal::Device& device,
    const std::vector<mx::array>& inputs,
    mx::array& out,
    const CoreDims& core_dims,
    PagedAttentionParams& params
  );

  void _eval_gpu_2pass(
    const mx::Stream& stream,
    mlx::core::metal::Device& device,
    const std::vector<mx::array>& inputs,
    mx::array& out,
    const CoreDims& core_dims,
    PagedAttentionParams& params
  );

  static uint32_t calculate_symmetric_tile_depth(
    uint32_t head_dimension,
    uint32_t num_query_heads,
    uint32_t num_kv_heads,
    size_t max_threadgroup_memory_bytes,
    size_t per_gqa_group_compute_scratch_bytes
  );

  static size_t calculate_per_gqa_group_compute_scratch(
    uint32_t head_dimension,
    uint32_t number_of_simd_groups,
    uint32_t threads_per_group
  );

  static bool should_use_fused_kernel(
    const CoreDims& core_dims,
    const std::vector<mx::array>& inputs
  );

};

}  // namespace pal::cpp
