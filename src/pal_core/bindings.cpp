#include <nanobind/nanobind.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/vector.h>

#include <mlx/mlx.h>
#include <mlx/array.h>
#include "pal_core/ops.hpp"

namespace nb = nanobind;
using namespace nb::literals;

// --- Nanobind Module Definition ---
NB_MODULE(pal_core, m) {
    m.doc() = "PAL C++ bindings: Paged Attention Operation";

    m.def("paged_attention",
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
                stream_or_device.value_or(mx::StreamOrDevice{})
            );
          },
          // Arguments for Python
          "queries"_a,
          "k_cache_pool"_a,
          "v_cache_pool"_a,
          "page_table"_a,
          "sequence_lengths"_a,
          "query_to_seq_map"_a,
          "query_token_offset"_a,
          nb::kw_only(), // stream is a keyword-only argument
          "stream"_a = nb::none(),
          nb::sig("def paged_attention(queries: mlx.core.array, "
                    "k_cache_pool: mlx.core.array, v_cache_pool: mlx.core.array, "
                    "page_table: mlx.core.array, sequence_lengths: mlx.core.array, "
                    "query_to_seq_map: mlx.core.array, query_token_offset: mlx.core.array, "
                    "*, stream: mlx.core.Stream | mlx.core.Device | None = None) -> mlx.core.array"),
          R"doc(
            Performs paged attention using a custom primitive.

            Args:
                queries (mlx.core.array): Queries array.
                k_cache_pool (mlx.core.array): Global K cache data pool.
                v_cache_pool (mlx.core.array): Global V cache data pool.
                page_table (mlx.core.array): Page table array containing physical page IDs.
                sequence_lengths (mlx.core.array): Array of actual lengths for each sequence in the batch.
                query_to_seq_map (mlx.core.array): Array mapping each query token to its sequence index in the batch.
                query_token_offset (mlx.core.array): Array of logical offsets for each query token within its sequence.
                stream (mlx.core.Stream | mlx.core.Device | None, optional): Stream or device for the operation.
            Returns:
                mlx.core.array: The result of the paged attention operation.
          )doc"
    );

    // Version information
    m.attr("__version__") = "0.1.0";
}
