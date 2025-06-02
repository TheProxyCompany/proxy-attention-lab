// bindings.cpp
// Python bindings for PAL core C++ implementations.
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

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>

#include <mlx/array.h>
#include <mlx/mlx.h>

#include "pal_core/ops.hpp"
#include "pal_core/paged_attention_primitive.hpp"

namespace nb = nanobind;
using namespace nb::literals;

/**
 * @brief PAL core Python module definition
 *
 * Defines the Python bindings for the Proxy Attention Lab core functionality.
 * Exposes the paged_attention operation to Python with full documentation
 * and type annotations.
 */
NB_MODULE(pal_core, m) {
  m.doc() = "PAL C++ bindings: Paged Attention Operation";

  m.def(
    "get_optimal_tile_size",
    []
    (uint32_t head_dimension,
    uint32_t num_query_heads,
    uint32_t num_kv_heads,
    std::optional<mx::StreamOrDevice> stream_or_device) {
      auto tile_info = pal::cpp::PagedAttentionPrimitive::get_optimal_tile_size_and_thread_info(
        head_dimension,
        num_query_heads,
        num_kv_heads,
        stream_or_device.value_or(mx::StreamOrDevice{}));
      auto tile_size = std::get<0>(tile_info);
      return tile_size;
    },
      "head_dimension"_a,
      "num_query_heads"_a,
      "num_kv_heads"_a,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig("def get_optimal_tile_size(head_dimension: int, num_query_heads: int, num_kv_heads: int, *, stream: mlx.core.Stream | mlx.core.Device | None = None) -> int"),
      R"doc(
        Calculates the optimal tile size for the paged attention kernel.

        Args:
            head_dimension (int): The dimension of the head
            num_query_heads (int): The number of query heads
            num_kv_heads (int): The number of key/value heads
            stream (mlx.core.Stream | mlx.core.Device | None, optional): Stream or device
                                                                        for the operation.
        Returns:
            int: The optimal tile size
      )doc");

  m.def(
      "paged_attention",
      [](const mx::array& queries,
         const mx::array& k_cache_pool,
         const mx::array& v_cache_pool,
         const mx::array& page_table,
         const mx::array& sequence_lengths,
         const mx::array& query_to_seq_map,
         const mx::array& query_token_offset,
         std::optional<mx::StreamOrDevice> stream_or_device) {
        return pal::cpp::paged_attention(
            queries,
            k_cache_pool,
            v_cache_pool,
            page_table,
            sequence_lengths,
            query_to_seq_map,
            query_token_offset,
            stream_or_device.value_or(mx::StreamOrDevice{}));
      },
      // Arguments for Python
      "queries"_a,
      "k_cache_pool"_a,
      "v_cache_pool"_a,
      "page_table"_a,
      "sequence_lengths"_a,
      "query_to_seq_map"_a,
      "query_token_offset"_a,
      nb::kw_only(),  // stream is keyword-only argument
      "stream"_a = nb::none(),
      nb::sig("def paged_attention(queries: mlx.core.array, "
              "k_cache_pool: mlx.core.array, v_cache_pool: mlx.core.array, "
              "page_table: mlx.core.array, sequence_lengths: mlx.core.array, "
              "query_to_seq_map: mlx.core.array, query_token_offset: mlx.core.array, "
              "*, stream: mlx.core.Stream | mlx.core.Device | None = None) -> "
              "mlx.core.array"),
      R"doc(
        Performs paged attention using a custom primitive.

        Args:
            queries (mlx.core.array): Queries array. May be 1D, 2D, or 3D:
                - 1D: [NumItems] with HeadDim=1
                - 2D: [NumItems, HeadDim] (NumQHeads implicitly 1)
                - 3D: [NumTokens, NumQHeads, HeadDim]
            k_cache_pool (mlx.core.array): Global K cache data pool with shape
                                          [NumTotalPages, TokensPerPage, NumKVHeads, HeadDim].
            v_cache_pool (mlx.core.array): Global V cache data pool with shape
                                          [NumTotalPages, TokensPerPage, NumKVHeads, HeadDim].
            page_table (mlx.core.array): Page table array mapping logical blocks to physical
                                        page IDs. Shape [NumSequencesInBatch, MaxLogicalBlocksPerSeq].
            sequence_lengths (mlx.core.array): Array of actual lengths for each sequence
                                              in the batch.
            query_to_seq_map (mlx.core.array): Array mapping each query token to its
                                              sequence index in the batch.
            query_token_offset (mlx.core.array): Array of logical offsets for each query
                                                token within its sequence.
            stream (mlx.core.Stream | mlx.core.Device | None, optional): Stream or device
                                                                        for the operation.
        Returns:
            mlx.core.array: The result of the paged attention operation:
                - If queries are 3D [NumTokens, NumQHeads, HeadDim], output is [NumTokens*NumQHeads, HeadDim]
                - If queries are 2D [NumItems, HeadDim], output is [NumItems, HeadDim]
                - If queries are 1D [NumItems], output is [NumItems, HeadDim]

        Note:
            The output HeadDim is always taken from the KV cache head dimension, regardless of query dimensions.
      )doc");

  // Version information
  m.attr("__version__") = "0.1.0";
}
