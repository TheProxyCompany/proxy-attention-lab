#include <mlx/mlx.h>
#include <mlx/array.h>
#include <nanobind/nanobind.h>
#include <mlx/core.h>
#include "ops/ops.hpp"

namespace nb = nanobind;
using namespace nb::literals;

// --- Nanobind Module Definition ---
NB_MODULE(pal_core, m) {
    m.doc() = "PAL C++ bindings: Paged Attention Operation";

    // Function overload that directly takes the arguments with explicit stream handling
    m.def("paged_attention",
          &pal::cpp::paged_attention,
          "queries"_a,
          "kv_cache"_a,
          "page_table"_a,
        //   nb::kw_only(),
        //   "stream"_a = nb::none(),
          nb::sig("def paged_attention(queries: mlx.core.array, kv_cache: mlx.core.array, page_table: mlx.core.array) -> mlx.core.array"),
          R"doc(
            Performs paged attention using a custom primitive.

            Args:
                queries (mlx.core.array): Queries array.
                kv_cache (mlx.core.array): KV cache buffer array.
                page_table (mlx.core.array): Page table mapping logical to physical blocks.
            Returns:
                mlx.core.array: The attention output array.
          )doc"
    );

    // Version information
    m.attr("__version__") = "0.1.0";
}
