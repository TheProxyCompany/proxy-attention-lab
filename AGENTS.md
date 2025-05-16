# Proxy Attention Lab Agent Guide

This guide explains how to set up your environment, run tests, and navigate the key files when contributing to the Proxy Attention Lab (PAL).

## Project Overview

PAL is a development playground for high-performance Metal kernels integrated with the MLX framework. It contains:

- **`pal_core` C++ library** that loads and executes Metal shaders as MLX custom primitives.
- **Metal kernels** such as `paged_attention.metal` implementing GPU logic.
- **Python bindings** built with Nanobind exposing the kernels to Python.
- **Pytest suite** for unit and integration tests.

While the current focus is a paged attention kernel, the framework is designed for experimenting with additional Metal kernels in the future.

## Development Environment Setup

1. **Prerequisites**
   - macOS with Xcode Command Line Tools
   - Python 3.11+
   - `uv` for dependency installation
2. **Create and activate a virtual environment**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
3. **Install dependencies and build**
   PAL relies on MLX and Nanobind. Run the following from the project root:
   ```bash
   uv pip install --no-deps git+https://github.com/TheProxyCompany/mlx.git nanobind==2.5.0
   uv pip install . --force-reinstall --no-build-isolation --no-cache-dir
   ```
   This installs Python requirements from `pyproject.toml` and triggers the CMake build for the C++ extension via `py-build-cmake`.

## Running Tests

Activate the virtual environment then execute:
```bash
pytest tests/
```
The tests cover core functionality, error handling and numerical equivalence for the paged attention kernel. All tests must pass before submitting changes.

## Code Structure / Key Files & Directories

- `pyproject.toml` – project metadata and build configuration
- `CMakeLists.txt` – top-level CMake configuration
- `run.sh` – script that installs dependencies, builds PAL and runs tests
- `src/proxy_attention_lab/ops.py` – Python op definitions
- `src/pal_core/` – C++ source for the extension
  - `CMakeLists.txt` – build file for the C++ library
  - `bindings.cpp` – Nanobind bindings
  - `kernels/` – Metal shader sources
  - `include/shaders/` – headers shared with Metal
  - `src/` – C++ primitive implementations
- `tests/` – Pytest suite

## Contribution & Development Guidelines

- **Code style**: follow existing style; Python code should comply with PEP 8 and include type hints and docstrings when possible.
- **Commit messages**: use clear descriptive messages (e.g., `feat:`, `fix:`, `docs:`).
- **Testing**: ensure `pytest tests/` passes; add tests for new functionality or bug fixes.
- **Workflow**: work on feature branches and keep pull requests focused.
