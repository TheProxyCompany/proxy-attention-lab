// ops.cpp
// Implementation of PAL core operations for MLX integration.
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

#include "pal_core/ops.hpp"

#include <string>

#include "mlx/backend/common/utils.h"
#include "mlx/array.h"
#include "mlx/utils.h"

#include "pal_core/metal/metal_loader.hpp"
#include "pal_core/fill_kv_pages_primitive.hpp"
#include "pal_core/paged_attention_primitive.hpp"

#include <spdlog/spdlog.h>

namespace pal::cpp {

mx::array paged_attention(
    const mx::array& queries,
    const mx::array& k_cache_pool,
    const mx::array& v_cache_pool,
    const mx::array& page_table,
    const mx::array& sequence_lengths,
    const mx::array& query_to_seq_map,
    const mx::array& query_token_offset,
    bool use_fused_kernel,
    mx::StreamOrDevice stream_or_device
  ) {
  spdlog::debug("[PAL Ops] pal::cpp::paged_attention C++ operation called.");

  // Ensure Metal library is loaded and registered
  pal::cpp::MetalLibRegistrar::ensure_pal_metallib_registered(stream_or_device);

  // Extract key parameters from input arrays to pass to the primitive
  int num_q_heads = 1;  // Default for 1D/2D queries
  int head_dim = 0;
  int tokens_per_page = 0;
  int num_kv_heads = 0;

  // Extract head_dim and tokens_per_page from K cache pool
  if (k_cache_pool.ndim() == 4) {
    num_kv_heads = k_cache_pool.shape(1);
    tokens_per_page = k_cache_pool.shape(2);
    head_dim = k_cache_pool.shape(3);
  }

  // For 3D queries, num_q_heads comes from the second dimension
  if (queries.ndim() == 3) {
    num_q_heads = queries.shape(1);
  }

  spdlog::debug(
      "[PAL Ops] Creating primitive with extracted params: num_q_heads={}, "
      "num_kv_heads={}, head_dim={}, tokens_per_page={}",
      num_q_heads, num_kv_heads, head_dim, tokens_per_page);

  // TODO: tune chunk size
  int chunk_size = 512;

  // Create the primitive instance with the extracted parameters
  auto primitive = std::make_shared<PagedAttentionPrimitive>(
      stream_or_device,
      num_q_heads,
      num_kv_heads,
      head_dim,
      tokens_per_page,
      !use_fused_kernel, // todo: change to use_two_pass
      chunk_size
    );

  spdlog::debug("[PAL Ops] PagedAttentionPrimitive instance created.");

  // Use the primitive's output_shapes method to determine the correct output shape
  auto output_shapes = primitive->output_shapes(
      {queries, k_cache_pool, v_cache_pool, page_table, sequence_lengths});

  if (output_shapes.empty()) {
    spdlog::error("[PAL Ops] PagedAttentionPrimitive returned empty output_shapes");
    throw std::runtime_error(
        "[PAL Ops] PagedAttentionPrimitive returned empty output_shapes");
  }

  auto out_shape = output_shapes[0];
  auto out_dtype = queries.dtype();

  // Construct the output MLX array, adding the operation in the graph
  return mx::array(
      out_shape, out_dtype, primitive,
      {
        queries,
        k_cache_pool,
        v_cache_pool,
        page_table,
        sequence_lengths
      }
    );
}

std::tuple<mx::array, mx::array> fill_kv_pages(
    const mx::array& new_keys,
    const mx::array& new_values,
    const mx::array& global_key_pool,
    const mx::array& global_value_pool,
    const mx::array& page_table,
    const mx::array& current_token_write_positions,
    const mx::array& query_to_seq_map,
    mx::StreamOrDevice stream_or_device
  ) {
  spdlog::debug("[PAL Ops] pal::cpp::fill_kv_pages C++ operation called.");

  // Ensure Metal library is loaded and registered
  pal::cpp::MetalLibRegistrar::ensure_pal_metallib_registered(stream_or_device);

  if (global_key_pool.ndim() != 4 || global_value_pool.ndim() != 4) {
      throw std::invalid_argument("[fill_kv_pages] global_key_pool and global_value_pool must be 4D.");
  }

  // Extract key parameters from input arrays to pass to the primitive
  int num_kv_heads = global_key_pool.shape(1);
  int tokens_per_page = global_key_pool.shape(2);
  int head_dim = global_key_pool.shape(3);

  // Create the primitive instance with the extracted parameters
  auto primitive = std::make_shared<FillKVPagesPrimitive>(
    stream_or_device,
    num_kv_heads,
    head_dim,
    tokens_per_page
  );
  spdlog::debug("[PAL Ops] FillKVPagesPrimitive instance created.");

  auto primitive_inputs = {
    new_keys, new_values,
    global_key_pool, global_value_pool, // These are the arrays to be modified
    page_table, current_token_write_positions, query_to_seq_map
  };

  std::vector<mx::Shape> result_shapes = primitive->output_shapes(primitive_inputs);
  std::vector<mx::Dtype> result_dtypes = {global_key_pool.dtype(), global_value_pool.dtype()};

  auto outputs = mx::array::make_arrays(result_shapes, result_dtypes, primitive, primitive_inputs);

  return std::make_tuple(outputs[0], outputs[1]);
}

}  // namespace pal::cpp
