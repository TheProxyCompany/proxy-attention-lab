# Proxy Attention Lab (PAL)

**A workbench for designing, iteratively testing, and benchmarking custom Metal kernels for the MLX framework.**

## Overview

Proxy Attention Lab (PAL) provides a lightweight environment for experimenting with new GPU operators on Apple Silicon. While the initial kernel is a paged attention implementation, the project is structured to support any custom Metal kernel. PAL combines a C++/Metal core with Nanobind bindings so that each operator can be exercised from Python. Pytest drives unit and equivalency tests, with `pytest-benchmark` and Google Benchmark used for performance analysis. Results can be aggregated and visualised through the built in analysis scripts.

## Philosophy

* **Modular design** – new kernels can be added without touching unrelated components.
* **Rapid iteration** – Nanobind exposes the Metal primitives to Python, enabling test‑driven development.
* **Comprehensive benchmarking** – Python and C++ benchmarks share a common results format that feeds into a report generator.

## Directory Layout

```
src/
  pal_core/              # C++ primitives, bindings and Metal sources
  proxy_attention_lab/   # Python package exposing operations
tests/                   # Unit tests and benchmarks
scripts/                 # Helper scripts including benchmark runner and analyzer
```

## Usage

1. Ensure macOS with Xcode command line tools and Python 3.11+ are installed.
2. Create a virtual environment and install PAL with its dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
uv pip install --no-deps git+https://github.com/TheProxyCompany/mlx.git nanobind==2.5.0
uv pip install . --force-reinstall --no-build-isolation --no-cache-dir
```

3. Run tests:

```bash
pytest tests/
```

See `scripts/benchmarks.sh --help` for running and analysing benchmarks.

