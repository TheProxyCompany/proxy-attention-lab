# pal_core

The `pal_core` directory holds the C++ library that wraps Metal kernels and provides Nanobind bindings.

- `include/pal_core/` – public C++ headers for primitives
- `include/kernels/` – header files shared between C++ and Metal
- `src/` – primitive implementations
- `kernels/` – Metal shader source files
- `bindings.cpp` – Nanobind module exposing primitives
- `CMakeLists.txt` – builds `pal_core_lib` and the metal library so it can be imported from Python or linked from other C++ projects
