"""Utilities for loading benchmark JSON files."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from benchmarks.analyzer import config


def detect_format(json_file: Path) -> str:
    """Detect the benchmark JSON format."""
    with open(json_file) as f:
        data = json.load(f)
    if "context" in data and "benchmarks" in data:
        return "google"
    if "machine_info" in data and "benchmarks" in data:
        return "pytest"
    raise ValueError(f"Unknown benchmark format for {json_file}")


def extract_kernel_name(filename_stem: str, bench_name: str | None = None) -> str:
    """
    Extract kernel name from filename or benchmark name.

    Args:
        filename_stem: The stem of the filename (without extension)
        bench_name: Optional benchmark name to extract kernel from

    Returns:
        Extracted kernel name (e.g., "paged_attention", "sdpa")
    """
    # First try to extract from benchmark name if provided
    if bench_name:
        # For BM_PAL_ or BM_SDPA_ format
        if "BM_PAL_" in bench_name:
            return "paged_attention"
        elif "BM_SDPA_" in bench_name:
            return "sdpa"
        # For test_pal_ or test_sdpa_ format
        elif "test_pal_" in bench_name:
            return "paged_attention"
        elif "test_sdpa_" in bench_name:
            return "sdpa"

    # Then try to extract from filename patterns
    # cpp_[kernel]_benchmarks_cpp_* or py_[kernel]_*
    kernel_match = re.search(r"(?:cpp|py)_([a-zA-Z0-9_]+)_", filename_stem)
    if kernel_match:
        kernel = kernel_match.group(1)
        # Map standard kernel names
        if kernel == "pal":
            return "paged_attention"
        elif kernel == "sdpa":
            return "sdpa"
        # For other kernels, return as-is
        return kernel

    # Fallback to basic detection
    if "paged_attention" in filename_stem or ("pal" in filename_stem.lower() and "sdpa" not in filename_stem.lower()):
        return "paged_attention"
    elif "sdpa" in filename_stem.lower():
        return "sdpa"

    # Last resort
    return "unknown_kernel"


def parse_google_benchmark(json_file: Path) -> list[dict]:
    """Parse a Google Benchmark JSON file."""
    with open(json_file) as f:
        data = json.load(f)
    rows = []

    # Determine a fallback source based on filename. This is only used when we
    # cannot infer the source from the benchmark name itself.
    filename_stem = json_file.stem
    default_source = None
    if filename_stem.startswith("cpp_pal_"):
        default_source = "cpp_pal"
    elif filename_stem.startswith("cpp_sdpa_"):
        default_source = "cpp_sdpa"
    elif "pal" in filename_stem.lower() and "sdpa" not in filename_stem.lower():
        default_source = "cpp_pal"
    elif "sdpa" in filename_stem.lower():
        default_source = "cpp_sdpa"

    for bench in data.get("benchmarks", []):
        if bench.get("run_type") != "aggregate" or bench.get("aggregate_name") != "mean":
            continue
        name = bench.get("name", "")

        # Determine source for each individual benchmark based on its name
        # This ensures correct source assignment even if a single JSON contains both PAL and SDPA benchmarks
        if "BM_PAL_" in name:
            source = "cpp_pal"
        elif "BM_SDPA_" in name:
            source = "cpp_sdpa"
        elif default_source:
            # Use the default source if we can't determine from benchmark name
            source = default_source
        else:
            # Last resort fallback - should rarely happen
            source = "cpp_pal" if "pal" in name.lower() and "sdpa" not in name.lower() else "cpp_sdpa"

        # Extract kernel name from benchmark name or filename
        kernel_name = extract_kernel_name(filename_stem, name)

        base_match = re.match(r"(?P<base>[^/]+)", name)
        base_name = base_match.group("base") if base_match else name
        params_str = name[len(base_name) :].lstrip("/")

        # Extract model config name if this is a model config benchmark
        model_config_name = None
        if "ModelConfig" in base_name:
            # For model config benchmarks, the config name might be in the base name
            model_match = re.search(r"ModelConfig_([^/]+)", name)
            if model_match:
                model_config_name = model_match.group(1)

        row = {
            config.COL_BENCHMARK_NAME_BASE: base_name,
            "full_name": name,
            config.COL_SOURCE: source,
            config.COL_KERNEL_NAME: kernel_name,  # Add the extracted kernel name
            config.COL_MEAN_LATENCY: bench.get("real_time", 0) / 1_000_000.0,
            config.COL_THROUGHPUT: bench.get("items_per_second"),
            config.COL_PARAMS_STR: params_str,
        }

        # Add model config information if available
        if model_config_name:
            row["model_config_name"] = model_config_name

        rows.append(row)
    return rows


def parse_pytest_benchmark(json_file: Path) -> list[dict]:
    """Parse a pytest-benchmark JSON file."""
    with open(json_file) as f:
        data = json.load(f)
    rows = []

    # Determine a fallback source from the filename. Individual benchmark entries
    # may override this based on their name.
    filename_stem = json_file.stem
    default_source = None
    if filename_stem.startswith("py_pal_") or filename_stem.startswith("python_pal_"):
        default_source = "python_pal"
    elif filename_stem.startswith("py_sdpa_") or filename_stem.startswith("python_sdpa_"):
        default_source = "python_sdpa"
    elif "pal" in filename_stem.lower() and "sdpa" not in filename_stem.lower():
        default_source = "python_pal"
    elif "sdpa" in filename_stem.lower():
        default_source = "python_sdpa"

    for bench in data.get("benchmarks", []):
        name = bench.get("name", "")

        # Determine source for this benchmark based on its name
        if "test_pal_" in name:
            bench_source = "python_pal"
        elif "test_sdpa_" in name:
            bench_source = "python_sdpa"
        else:
            bench_source = default_source or "python_pal"

        # Extract kernel name from benchmark name or filename
        kernel_name = extract_kernel_name(filename_stem, name)

        group = bench.get("group", "")
        base_name = group if group else name.split("[")[0]

        # Extract parameter string from pytest name format
        params_str = ""
        if "[" in name and "]" in name:
            params_str = name.split("[")[1].split("]")[0]

        stats = bench.get("stats", {})

        # Extract model config information
        model_config_name = None
        model_params_raw = None

        # Check if this is a model config benchmark
        if "model_configs" in base_name and "[" in name and "]" in name:
            # For pytest, the model config and params might be in the params_str
            # Format could be like: test_pal_latency_model_configs[Llama3_70B_Sim-{...}]
            param_parts = params_str.split("-", 1)
            if param_parts:
                model_config_name = param_parts[0]
                # Save model params if available
                if len(param_parts) > 1 and param_parts[1].startswith("{") and param_parts[1].endswith("}"):
                    model_params_raw = param_parts[1]

        row = {
            config.COL_BENCHMARK_NAME_BASE: base_name,
            "full_name": name,
            config.COL_SOURCE: bench_source,
            config.COL_KERNEL_NAME: kernel_name,  # Add the extracted kernel name
            config.COL_MEAN_LATENCY: stats.get("mean", 0) * 1000.0,  # Convert to ms
            config.COL_PARAMS_STR: params_str,
            "rounds": stats.get("rounds"),
        }

        # Add any extracted model config information
        if model_config_name:
            row["model_config_name"] = model_config_name
        if model_params_raw:
            row["model_params_raw"] = model_params_raw

        # Calculate throughput for Python benchmarks if not already present
        # This will be refined in the transformers module after we extract all parameters

        rows.append(row)
    return rows


def load_all_results(json_files: list[Path]) -> pd.DataFrame:
    """Load all benchmark results into a DataFrame."""
    frames = []
    for jf in json_files:
        fmt = detect_format(jf)
        rows = parse_google_benchmark(jf) if fmt == "google" else parse_pytest_benchmark(jf)
        frames.append(pd.DataFrame(rows))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)
