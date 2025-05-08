# Proxy Attention Lab

Playground for designing, profiling, and testing Metal attention kernels with MLX integration.

## What's here

| Folder | Purpose |
|--------|---------|
| `src/cpp/` | C++ primitives and Metal kernel integration with MLX |
| `src/python/` | Python API wrapping the C++ implementation |
| `tests/` | `pytest` unit + perf checks |
| `scripts/` | helper scripts (`build_metal.sh`, optional `watch.sh`) |
| `build/` | Auto-generated build artifacts (`.metallib`, `.so`, etc.) |

## Prerequisites

```bash
xcode-select --install
uv venv .venv && source .venv/bin/activate
uv pip install -e -U "."        
```

## Quick Start

Build and test:
```bash
# Build the C++ extension and Metal library
CMAKE_BUILD_TYPE=Release pip install -e .

# Run tests
pytest -n auto
```

Hot reload during development:
```bash
brew install fswatch
./scripts/watch.sh
```

## Architecture

The project implements paged attention using MLX primitives:

1. Metal Kernel (`src/cpp/paged_attention.metal`): Implements the core computation
2. C++ Primitive (`src/cpp/paged_attention_primitive.{hpp,cpp}`): MLX integration
3. Python API (`src/python/ops.py`): High-level interface for Python users

This approach allows for high-performance execution while maintaining a clean Python API.