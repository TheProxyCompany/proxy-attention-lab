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
      "paged_attention",
      [](const mx::array& queries,
         const mx::array& k_cache_pool,
         const mx::array& v_cache_pool,
         const mx::array& page_table,
         const mx::array& sequence_lengths,
         const mx::array& query_to_seq_map,
         const mx::array& query_token_offset,
         bool is_prefill,
         std::optional<mx::StreamOrDevice> stream_or_device) {
        return pal::cpp::paged_attention(
            queries,
            k_cache_pool,
            v_cache_pool,
            page_table,
            sequence_lengths,
            query_to_seq_map,
            query_token_offset,
            is_prefill,
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
      nb::kw_only(),  // is_prefill and stream are keyword-only arguments
      "is_prefill"_a = true,  // default to prefill mode
      "stream"_a = nb::none(),
      nb::sig("def paged_attention(queries: mlx.core.array, "
              "k_cache_pool: mlx.core.array, v_cache_pool: mlx.core.array, "
              "page_table: mlx.core.array, sequence_lengths: mlx.core.array, "
              "query_to_seq_map: mlx.core.array, query_token_offset: mlx.core.array, "
              "*, is_prefill: bool = True, stream: mlx.core.Stream | mlx.core.Device | None = None) -> "
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
            is_prefill (bool, optional): Whether to perform prefill or decoding. Defaults to True.
                - When True (prefill mode): Processes full sequence with one threadgroup per query token
                - When False (decode mode): Processes single token per sequence with one threadgroup
                                           per query-token-head pair.
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
