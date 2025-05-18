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
};

// Expected size for PagedAttentionParams: 10 uint32_t (40 bytes) + 2 float (8 bytes) = 48 bytes.
// alignas(16) means total size is 48, as it's padded to multiple of 16.
// Note: We use 64-byte alignment for threadgroup memory, but the struct itself remains 16-byte aligned.
constexpr size_t kExpectedPagedAttentionParamsSize = 48;
static_assert(
    sizeof(PagedAttentionParams) == kExpectedPagedAttentionParamsSize,
    "sizeof(PagedAttentionParams) mismatch between C++ and expected size (48 bytes). "
    "Check paged_attention_types.h, members, and padding.");

// Constants for memory padding and alignment
constexpr size_t kFinalTgMemoryPaddingGuardBytes = 32;
constexpr size_t kAlignmentBytes = 64;
constexpr size_t kAlignmentMask = kAlignmentBytes - 1;

// Constants for head_dim validation and processing
constexpr uint32_t kMaxHeadDimMetalInKernel = 256;


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
  std::vector<mx::Shape> output_shapes(const std::vector<mx::array>& inputs) override;

 private:
  // Parameters that define kernel behavior
  int num_q_heads_;
  int num_kv_heads_;
  int head_dim_;
  int tokens_per_page_;
  MTL::ComputePipelineState* kernel_state_{nullptr};

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
