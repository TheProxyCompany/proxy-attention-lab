#pragma once
// fill_kv_pages_primitive.hpp
// Defines the FillKVPagesPrimitive class for MLX KV cache page filling operations.
//
// Copyright 2025 The Proxy Company. All Rights Reserved.
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

static constexpr size_t SIMD_GROUPS_PER_THREADGROUP = 2; // wip

static constexpr int KVPairsPerThreadgroup = 8;

/**
 * @brief Custom primitive implementation for filling KV cache pages.
 *
 * This class implements a primitive for filling key and value cache pages in the MLX
 * framework. It manages both CPU and GPU implementations, handling the lifecycle
 * and execution of the operation within the computational graph.
 */
class FillKVPagesPrimitive : public mx::Primitive {
 public:
  /**
   * @brief Constructs a FillKVPagesPrimitive with specified parameters.
   *
   * @param stream_or_device The MLX stream or device to execute on
   * @param num_kv_heads Number of key/value heads
   * @param head_dim Hidden dimension size per attention head
   * @param tokens_per_page Number of tokens stored in each memory page
   */
  explicit FillKVPagesPrimitive(
    mx::StreamOrDevice stream_or_device,
    int num_kv_heads = 0,
    int head_dim = 0,
    int tokens_per_page = 0
  );

  /**
   * @brief Evaluates the primitive on CPU.
   *
   * @param inputs Vector of input arrays
   * @param out Output array to store the result (may be unused)
   */
  void eval_cpu(const std::vector<mx::array>& inputs, std::vector<mx::array>& outputs) override;

  /**
   * @brief Evaluates the primitive on GPU.
   *
   * Launches the Metal kernel to perform the KV page filling.
   *
   * @param inputs Vector of input arrays
   * @param out Output array to store the result (may be unused)
   */
  void eval_gpu(const std::vector<mx::array>& inputs, std::vector<mx::array>& outputs) override;

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
    if (typeid(*this) != typeid(other)) {
      return false;
    }
    const FillKVPagesPrimitive& other_fkv =
        static_cast<const FillKVPagesPrimitive&>(other);
    return (this->num_kv_heads_ == other_fkv.num_kv_heads_ &&
            this->head_dim_ == other_fkv.head_dim_ &&
            this->tokens_per_page_ == other_fkv.tokens_per_page_);
  }

  /**
   * @brief Calculates output shapes based on input shapes.
   *
   * @param inputs Vector of input arrays
   * @return Vector of output shapes
   */
  std::vector<mx::Shape> output_shapes(const std::vector<mx::array>& inputs) override;

 private:
  // Parameters that define kernel behavior
  int num_kv_heads_;
  int head_dim_;
  int tokens_per_page_;
};

}  // namespace pal::cpp
