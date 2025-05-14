# Proxy Attention Lab (PAL)

**A laboratory for developing and benchmarking high-performance, custom paged attention kernels for MLX on Apple Silicon.**

## Overview

The Proxy Attention Lab (PAL) is a C++/Metal-based library designed to provide custom, high-performance MLX (https://github.com/ml-explore/mlx) operators, with an initial focus on paged attention mechanisms. Paged attention is a key technique used in modern Large Language Model (LLM) inference engines to efficiently manage and access the Key-Value (KV) cache, enabling larger batch sizes and longer sequence lengths.

This project aims to explore and implement production-quality paged attention kernels specifically optimized for Apple Silicon (M-series GPUs) using the Metal Shading Language, while integrating seamlessly with the MLX framework.

## Current Status (Forward Pass - Functionally Complete & Robust)

As of the latest iteration (post TDD-7.6):

*   **Functionally Complete Forward Pass:** The core `paged_attn_kernel` (Metal) and its C++ MLX primitive (`PagedAttentionPrimitive`) successfully compute the full forward pass for paged attention. This includes:
    *   Numerically stable calculation of attention scores (max score, sum-exp).
    *   Softmax probability computation.
    *   Weighted aggregation of Value (V) vectors.
*   **Dynamic V-Accumulator Tiling:** The kernel employs a hybrid tiled stack strategy for V-accumulation, using a fixed-size thread-local tile and an outer loop to process `head_dim` in chunks. This correctly handles large head dimensions without overflowing thread stack memory or exceeding `threadgroup` memory limits.
*   **Robustness:**
    *   Handles GQA/MQA configurations.
    *   Explicitly outputs zeros for zero-history scenarios.
    *   Includes runtime alignment checks for K/V cache pointers with scalar fallbacks.
    *   Parameter marshalling between C++ and Metal is hardened with `static_asserts` and internal `scale` recalculation.
*   **Logging:** Uses `spdlog` for standardized C++ logging, integrated via CMake.
*   **Code Quality:** The C++, Metal, and Python codebase has undergone a "Production-Ready Code Specification" cleanup, focusing on Google C++ Style (C++/Metal) and PEP 8/Black (Python), Doxygen/docstring documentation, type hinting, and descriptive naming.
*   **Testing:** A suite of Python unit tests (using `pytest`) covers various scenarios, including core functionality, GQA/MQA, error handling, boundary conditions, and full V-aggregation. All tests are currently passing.

## Key Architectural Features

*   **MLX Integration:** Implemented as a custom MLX C++ primitive (`PagedAttentionPrimitive`).
*   **Metal Kernel (`paged_attn_kernel`):**
    *   **Dispatch:** One Metal threadgroup per query-item (token-head pair).
    *   **Q-Vector Staging:** Queries are pre-scaled and cooperatively loaded into `threadgroup` memory.
    *   **Paged K/V Access:** Threads within a group scan distinct history chunks, fetching K/V from paged pools via a page table.
    *   **Fused Two-Pass Softmax & V-Aggregation:** A single kernel launch performs:
        1.  **Statistics Pass:** Online log-sum-exp for local max/sum-exp, followed by threadgroup reduction for global statistics.
        2.  **Probability & V-Aggregation Pass:** Re-scores, calculates softmax probabilities using global stats, fetches V, and accumulates weighted V into thread-local tiles.
    *   **Tiled V-Reduction:** Reduced V-vector tiles are written directly to the output buffer.
*   **Parameterization:** Kernel behavior is controlled via a `PagedAttentionParams` struct passed from C++.
*   **Build System:** CMake for C++/Metal components, Python bindings via `nanobind`, and integration with `py-build-cmake` for the Python package.

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
