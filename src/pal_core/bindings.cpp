#include <mlx/mlx.h>
#include <mlx/array.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/variant.h>
#include "ops.hpp"

namespace nb = nanobind;
using namespace nb::literals;

// --- Nanobind Module Definition ---
NB_MODULE(pal_core, m) {
    m.doc() = "PAL C++ bindings: Paged Attention Operation";

    // Function overload that directly takes the arguments with explicit stream handling
    m.def("paged_attention",
          &pal::cpp::paged_attention,
          "queries"_a,
          nb::sig("def paged_attention(queries: mlx.core.array) -> int"),
          R"doc(
            Performs paged attention using a custom primitive.

            Args:
                queries (mlx.core.array): Queries array.
            Returns:
                int: The size of the queries array.
          )doc"
    );

    // Version information
    m.attr("__version__") = "0.1.0";
}
