// bindings.cpp
// Python bindings for PAL core C++ implementations.
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

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/tuple.h>

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
    "get_optimal_page_size",
    pal::cpp::PagedAttentionPrimitive::get_optimal_page_size,
    nb::sig("def get_optimal_page_size() -> int"),
    R"doc(
        Calculates the optimal page size for the paged attention kernel.
    )doc");
  m.def(
      "paged_attention",
      [](const mx::array& queries,
         const mx::array& k_cache_pool,
         const mx::array& v_cache_pool,
         const mx::array& page_table,
         const mx::array& sequence_lengths,
         std::optional<mx::StreamOrDevice> stream_or_device) {
        return pal::cpp::paged_attention(
            queries,
            k_cache_pool,
            v_cache_pool,
            page_table,
            sequence_lengths,
            stream_or_device.value_or(mx::StreamOrDevice{}));
      },
      // Arguments for Python
      "queries"_a,
      "k_cache_pool"_a,
      "v_cache_pool"_a,
      "page_table"_a,
      "sequence_lengths"_a,
      nb::kw_only(),  // stream is keyword-only argument
      "stream"_a = nb::none(),
      nb::sig("def paged_attention(queries: mlx.core.array, "
              "k_cache_pool: mlx.core.array, v_cache_pool: mlx.core.array, "
              "page_table: mlx.core.array, sequence_lengths: mlx.core.array, "
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
                                          [NumTotalPages, NumKVHeads, TokensPerPage, HeadDim].
            v_cache_pool (mlx.core.array): Global V cache data pool with shape
                                          [NumTotalPages, NumKVHeads, TokensPerPage, HeadDim].
            page_table (mlx.core.array): Page table array mapping logical blocks to physical
                                        page IDs. Shape [NumSequencesInBatch, MaxLogicalBlocksPerSeq].
            sequence_lengths (mlx.core.array): Array of actual lengths for each sequence
                                              in the batch.
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

  m.def(
      "fill_kv_pages",
      [](const mx::array& new_keys,
         const mx::array& new_values,
         const mx::array& global_key_pool,
         const mx::array& global_value_pool,
         const mx::array& page_table,
         const mx::array& current_token_write_positions,
         const mx::array& query_to_seq_map,
         std::optional<mx::StreamOrDevice> stream_or_device) {
          return pal::cpp::fill_kv_pages(
              new_keys,
              new_values,
              global_key_pool,
              global_value_pool,
              page_table,
              current_token_write_positions,
              query_to_seq_map,
              stream_or_device.value_or(mx::StreamOrDevice{})
          );
      },
      "new_keys"_a,
      "new_values"_a,
      "global_key_pool"_a,
      "global_value_pool"_a,
      "page_table"_a,
      "current_token_write_positions"_a,
      "query_to_seq_map"_a,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig("def fill_kv_pages(new_keys: mlx.core.array, "
              "new_values: mlx.core.array, "
              "global_key_pool: mlx.core.array, "
              "global_value_pool: mlx.core.array, "
              "page_table: mlx.core.array, "
              "current_token_write_positions: mlx.core.array, "
              "query_to_seq_map: mlx.core.array, "
              "*, stream: mlx.core.Stream | mlx.core.Device | None = None) -> tuple[mlx.core.array, mlx.core.array]"),
      R"doc(
        Fills the KV cache pages with the new keys and values.
      )doc");

  // Version information
  m.attr("__version__") = "0.1.0";
}
