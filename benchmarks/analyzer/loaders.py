"""Utilities for loading benchmark JSON files."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import pandas as pd

from benchmarks.analyzer import config

# Set up logging
logger = logging.getLogger(__name__)


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
    """Return the kernel name inferred from a filename or benchmark name."""

    patterns = {
        "paged_attention": re.compile(r"(?:pal|paged_attention)", re.IGNORECASE),
        "sdpa": re.compile(r"sdpa", re.IGNORECASE),
    }

    # Prefer the benchmark name when available
    if bench_name:
        for kernel, pattern in patterns.items():
            if pattern.search(bench_name):
                return kernel

    # Next try the filename stem
    for kernel, pattern in patterns.items():
        if pattern.search(filename_stem):
            return kernel

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

    # Track benchmark names for debugging
    raw_benchmark_names = []

    for bench in data.get("benchmarks", []):
        if bench.get("run_type") == "aggregate" and bench.get("aggregate_name") != "mean":
            # Skip non-mean aggregates
            continue

        name = bench.get("name", "")
        raw_benchmark_names.append(name)

        # Determine kernel_name and source directly from benchmark name for C++
        # This ensures more accurate and consistent classification
        kernel_name = None
        source = None

        # Check for specific benchmark prefixes first
        if "BM_PAL_" in name:
            kernel_name = "paged_attention"
            source = "cpp_pal"
        elif "BM_SDPA_" in name:
            kernel_name = "sdpa"
            source = "cpp_sdpa"
        elif default_source:
            # Use the default source from filename
            source = default_source
            # Extract kernel name from filename or benchmark name
            kernel_name = extract_kernel_name(filename_stem, name)
        else:
            # Last resort fallback - should rarely happen
            if "pal" in name.lower() and "sdpa" not in name.lower():
                source = "cpp_pal"
                kernel_name = "paged_attention"
            elif "sdpa" in name.lower():
                source = "cpp_sdpa"
                kernel_name = "sdpa"
            else:
                # Truly unknown, use filename for inference
                source = "cpp_unknown"
                kernel_name = extract_kernel_name(filename_stem, name)

        # Make absolutely sure we have a kernel name
        if not kernel_name:
            logger.warning(f"Failed to extract kernel name for benchmark: {name}")
            kernel_name = "unknown_kernel"

        logger.debug(f"C++ benchmark: {name} → source={source}, kernel={kernel_name}")

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
                logger.debug(f"Extracted C++ model config: {model_config_name} from {name}")

        # Convert the reported real_time to milliseconds. Google Benchmark may
        # emit timings in various units, so interpret the provided time_unit
        # field when available.
        time_unit = bench.get("time_unit", "ns")
        real_time = bench.get("real_time", 0)
        if time_unit == "ns":
            mean_latency_ms = real_time / 1_000_000.0
        elif time_unit == "us":
            mean_latency_ms = real_time / 1_000.0
        elif time_unit == "ms":
            mean_latency_ms = real_time
        elif time_unit == "s":
            mean_latency_ms = real_time * 1000.0
        else:
            logger.warning(f"Unexpected time unit '{time_unit}' for benchmark {name}; assuming ns")
            mean_latency_ms = real_time / 1_000_000.0

        row = {
            config.COL_BENCHMARK_NAME_BASE: base_name,
            "full_name": name,
            config.COL_SOURCE: source,
            config.COL_KERNEL_NAME: kernel_name,  # Add the extracted kernel name
            config.COL_MEAN_LATENCY: mean_latency_ms,
            config.COL_THROUGHPUT: bench.get("items_per_second", None),
            config.COL_PARAMS_STR: params_str,
        }

        # Add model config information if available
        if model_config_name:
            row["model_config_name"] = model_config_name

        rows.append(row)

    if not rows and raw_benchmark_names:
        logger.warning(f"No rows extracted from {len(raw_benchmark_names)} benchmarks in {json_file.name}")
        logger.debug(
            f"Raw benchmark names: {raw_benchmark_names[:5]}..."
            if len(raw_benchmark_names) > 5
            else raw_benchmark_names
        )

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

    # Track benchmark names for debugging
    raw_benchmark_names = []

    for bench in data.get("benchmarks", []):
        name = bench.get("name", "")
        if "cold_start" in name:
            continue

        raw_benchmark_names.append(name)

        # Determine kernel_name and source directly from benchmark name for Python
        kernel_name = None
        bench_source = None

        # First try to identify by test name pattern
        if "test_pal_" in name:
            kernel_name = "paged_attention"
            bench_source = "python_pal"
        elif "test_sdpa_" in name:
            kernel_name = "sdpa"
            bench_source = "python_sdpa"
        elif default_source:
            # Use default source from filename
            bench_source = default_source
            # Extract kernel name from filename or benchmark name
            kernel_name = extract_kernel_name(filename_stem, name)
        else:
            # Last resort fallback based on name contents
            if "pal" in name.lower() and "sdpa" not in name.lower():
                bench_source = "python_pal"
                kernel_name = "paged_attention"
            elif "sdpa" in name.lower():
                bench_source = "python_sdpa"
                kernel_name = "sdpa"
            else:
                # Truly unknown
                bench_source = "python_unknown"
                kernel_name = extract_kernel_name(filename_stem, name)

        # Make absolutely sure we have a kernel name
        if not kernel_name:
            logger.warning(f"Failed to extract kernel name for benchmark: {name}")
            kernel_name = "unknown_kernel"

        logger.debug(f"Python benchmark: {name} → source={bench_source}, kernel={kernel_name}")

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

        rows.append(row)

    if not rows and raw_benchmark_names:
        logger.warning(f"No rows extracted from {len(raw_benchmark_names)} benchmarks in {json_file.name}")
        logger.debug(
            f"Raw benchmark names: {raw_benchmark_names[:5]}..."
            if len(raw_benchmark_names) > 5
            else raw_benchmark_names
        )

    return rows


def load_all_results(json_files: list[Path]) -> pd.DataFrame:
    """Load all benchmark results into a DataFrame."""
    frames = []
    for jf in json_files:
        try:
            fmt = detect_format(jf)
            rows = parse_google_benchmark(jf) if fmt == "google" else parse_pytest_benchmark(jf)

            # Skip empty result sets
            if not rows:
                logger.warning(f"No benchmark data extracted from {jf.name}")
                continue

            # Create a DataFrame and ensure it has the kernel_name column
            df_chunk = pd.DataFrame(rows)

            # Check if kernel_name column is missing
            if config.COL_KERNEL_NAME not in df_chunk.columns:
                logger.warning(f"Missing kernel_name column in {jf.name} - inferring from filename")
                # Provide a fallback kernel_name based on the filename
                kernel_name = extract_kernel_name(jf.stem)
                df_chunk[config.COL_KERNEL_NAME] = kernel_name
                logger.info(f"Added kernel_name column with value '{kernel_name}' to data from {jf.name}")

            # Verify that all rows have a valid kernel_name
            missing_kernel_names = df_chunk[config.COL_KERNEL_NAME].isna().sum()
            if missing_kernel_names > 0:
                logger.warning(f"{missing_kernel_names} rows from {jf.name} have missing kernel_name values")
                # Extract a kernel name from filename as fallback
                default_kernel = extract_kernel_name(jf.stem)
                df_chunk[config.COL_KERNEL_NAME] = df_chunk[config.COL_KERNEL_NAME].fillna(default_kernel)
                logger.info(f"Filled missing kernel_name values with '{default_kernel}'")

            frames.append(df_chunk)
        except Exception as e:
            logger.error(f"Error processing benchmark file {jf.name}: {e}")

    if not frames:
        logger.error("No valid benchmark data frames were created from any input files")
        return pd.DataFrame()

    # Debug: log frame info before concatenation
    for i, frame in enumerate(frames):
        logger.debug(f"Frame {i} from file: columns={frame.columns.tolist()}, rows={len(frame)}")
        if config.COL_KERNEL_NAME not in frame.columns:
            logger.error(f"Frame {i} is still missing {config.COL_KERNEL_NAME} column!")
            # Add the column to prevent concat failure
            frame[config.COL_KERNEL_NAME] = "unknown_kernel"

    # Concatenate all frames
    df = pd.concat(frames, ignore_index=True)

    # Final validation after concatenation
    if config.COL_KERNEL_NAME not in df.columns:
        logger.error("CRITICAL: No kernel_name column in final concatenated DataFrame")
        # Add the column as last resort
        df[config.COL_KERNEL_NAME] = "unknown_kernel"
    elif df[config.COL_KERNEL_NAME].isna().any():
        missing_count = df[config.COL_KERNEL_NAME].isna().sum()
        logger.warning(f"{missing_count} rows have missing kernel_name values after concatenation")
        # Fill missing values with default
        df[config.COL_KERNEL_NAME] = df[config.COL_KERNEL_NAME].fillna("unknown_kernel")

    # Log kernel name distribution for debugging
    if not df.empty:
        kernel_counts = df[config.COL_KERNEL_NAME].value_counts().to_dict()
        logger.info(f"Kernel name distribution in loaded data: {kernel_counts}")

    return df
