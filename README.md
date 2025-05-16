# Proxy Attention Lab (PAL)

**A laboratory for developing and benchmarking high-performance, custom paged attention kernels for MLX on Apple Silicon.**

## Overview

The Proxy Attention Lab (PAL) is a C++/Metal-based library designed to provide custom, high-performance MLX (https://github.com/ml-explore/mlx) operators, with an initial focus on paged attention mechanisms. Paged attention is a key technique used in modern Large Language Model (LLM) inference engines to efficiently manage and access the Key-Value (KV) cache, enabling larger batch sizes and longer sequence lengths.

This project aims to explore and implement production-quality paged attention kernels specifically optimized for Apple Silicon (M-series GPUs) using the Metal Shading Language, while integrating seamlessly with the MLX framework.

## Prerequisites

*   macOS (for Metal development)
*   Xcode Command Line Tools (includes Clang and Metal compilers)
*   CMake (>= 3.20)
*   Python (>= 3.11 recommended)
*   `uv` (for Python package management, as used in `run.sh`) or `pip`
*   MLX (PAL links against an existing MLX installation or can be adapted to fetch it)
*   Nanobind

## Building and Testing

The primary way to build PAL and run its tests is using the provided shell script:

```bash
./run.sh
```

This script will:
1.  Set up a Python virtual environment (if not already configured as per script).
2.  Install/update Python dependencies using `uv pip install .`. This step also triggers the CMake build process for the C++ extension via `py-build-cmake`.
3.  Run `pytest` on the `tests/` directory.

To force a clean build:
```bash
CLEAN_BUILD=true ./run.sh
```

## Next Steps & Future Work (Tentative)

*   **Backward Pass Implementation:** Develop and test the backward pass for the paged attention kernel.
*   **PIE Integration:** Fully integrate and validate PAL within the Proxy Inference Engine (PIE), ensuring correct data flow from PIE's scheduler and KV cache manager to PAL's paged attention operator.
