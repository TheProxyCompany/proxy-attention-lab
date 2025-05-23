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
#include "shaders/paged_attention_types.h"

namespace mx = mlx::core;

namespace pal::cpp {

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
   * @param is_prefill Whether to perform prefill or decoding
   */
  explicit PagedAttentionPrimitive(
    mx::StreamOrDevice stream,
    int num_q_heads = 0,
    int num_kv_heads = 0,
    int head_dim = 0,
    int tokens_per_page = 0,
    bool is_prefill = true
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
  bool is_prefill_;

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

};

}  // namespace pal::cpp
