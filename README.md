# Proxy Attention Lab

Playground for designing, profiling, and testing Metal attention kernels.

## What’s here

| Folder | Purpose |
|--------|---------|
| `kernels/` | `.metal` source files (e.g. `paged_attention.metal`) |
| `lab/` | Python harness that registers the kernels with **MLX** |
| `tests/` | `pytest` unit + perf checks |
| `scripts/` | helper scripts (`build_metal.sh`, optional `watch.sh`) |
| `build/` | Auto-generated `.air` + `.metallib` binaries |

## Prerequisites

```bash
xcode-select --install          # Metal toolchain
uv venv .venv && source .venv/bin/activate
uv pip install -e -U "."         # installs mlx, pytest, xdist
```

Quick start
```bash
./scripts/build_metal.sh
pytest -n auto
```

Hot reload during dev:
```bash
brew install fswatch
./scripts/watch.sh
```
