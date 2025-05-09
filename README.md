# Proxy Attention Lab

Playground for designing, profiling, and testing Metal attention kernels with MLX integration.

## What's here

| Folder | Purpose |
|--------|---------|
| `src/pal_core/` | C++ primitives and Metal kernel integration with MLX |
| `src/proxy_attention_lab/` | Python API wrapping the C++ implementation |
| `tests/` | `pytest` unit + perf checks |

## Prerequisites

```bash
xcode-select --install
uv venv .venv && source .venv/bin/activate
uv pip install -e -U "."
```

## Quick Start

Build and test:
```bash
./run.sh
```
## Architecture

The project implements paged attention using MLX primitives:

1. Metal Kernel (`src/pal_core/paged_attention.metal`): Implements the core computation
2. C++ Primitive (`src/pal_core/paged_attention_primitive.{hpp,cpp}`): MLX integration
3. Python API (`src/proxy_attention_lab/ops.py`): High-level interface for Python users

This approach allows for high-performance execution while maintaining a clean Python API.
