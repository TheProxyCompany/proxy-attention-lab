#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>
#include <mlx/mlx.h>
#include "ops/ops.hpp"

namespace nb = nanobind;
using namespace nb::literals;

// --- Nanobind Module Definition ---
NB_MODULE(_pal_cpp_binding, m) {
    m.doc() = "PAL C++ bindings: Paged Attention Operation";

    // Add the binding for the C++ operation function pal::cpp::paged_attention
    m.def("paged_attention",             // Python function name
          &pal::cpp::paged_attention,    // Pointer to the C++ function
          "q"_a,                         // Argument 'q' of type mx::array
          "kv_cache"_a,                  // Argument 'kv_cache' of type mx::array
          "page_table"_a,                // Argument 'page_table' of type mx::array
          "stream"_a = nb::none(),       // Optional 'stream' argument (maps to StreamOrDevice)
          R"doc(
            Performs paged attention using a custom primitive.

            Args:
                q (mlx.core.array): Queries array.
                kv_cache (mlx.core.array): KV cache buffer array.
                page_table (mlx.core.array): Page table mapping logical to physical blocks.
                stream (mlx.core.Stream | mlx.core.Device | None): Optional stream or device.

            Returns:
                mlx.core.array: The attention output array.
          )doc"
    );

    // Version information
    m.attr("__version__") = "0.1.0";
}
