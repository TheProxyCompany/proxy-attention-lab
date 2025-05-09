#include <nanobind/nanobind.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/optional.h>

#include <mlx/mlx.h>
#include <mlx/array.h>
#include "ops.hpp"

namespace nb = nanobind;
using namespace nb::literals;

// --- Nanobind Module Definition ---
NB_MODULE(pal_core, m) {
    m.doc() = "PAL C++ bindings: Paged Attention Operation";

    m.def("paged_attention",
          [](const mlx::core::array& queries,
             const mlx::core::array& kv_cache,
             const mlx::core::array& page_table,
             std::optional<mlx::core::StreamOrDevice> stream_or_device) {
            // Resolve optional stream to default if None is passed from Python
            auto resolved_stream_or_device = stream_or_device.value_or(mx::StreamOrDevice{});
            return pal::cpp::paged_attention(queries, kv_cache, page_table, resolved_stream_or_device);
          },
          "queries"_a,
          "kv_cache"_a,
          "page_table"_a,
          nb::kw_only(),
          "stream"_a = nb::none(),
          nb::sig("def paged_attention(queries: mlx.core.array, kv_cache: mlx.core.array, page_table: mlx.core.array, stream: mlx.core.Stream | mlx.core.Device | None = None) -> mlx.core.array"),
          R"doc(
            Performs paged attention using a custom primitive.

            Args:
                queries (mlx.core.array): Queries array.
                kv_cache (mlx.core.array): KV cache array.
                page_table (mlx.core.array): Page table array.
                stream (mlx.core.Stream | mlx.core.Device | None): Stream or device to use for the operation.
            Returns:
                mlx.core.array: The result of the paged attention operation.
          )doc"
    );

    // Version information
    m.attr("__version__") = "0.1.0";
}
