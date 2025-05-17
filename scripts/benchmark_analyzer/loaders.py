"""Utilities for loading benchmark JSON files."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from . import config


def _parse_source_info(json_file: Path) -> tuple[str, str]:
    """Return (language, kernel_name) parsed from filename."""
    parts = json_file.stem.split("_")
    language = "python" if parts and parts[0] == "py" else "cpp"
    kernel = parts[1] if len(parts) > 1 else "unknown"
    return language, kernel


def detect_format(json_file: Path) -> str:
    """Detect the benchmark JSON format."""
    with open(json_file) as f:
        data = json.load(f)
    if "context" in data and "benchmarks" in data:
        return "google"
    if "machine_info" in data and "benchmarks" in data:
        return "pytest"
    raise ValueError(f"Unknown benchmark format for {json_file}")


def parse_google_benchmark(json_file: Path) -> list[dict]:
    """Parse a Google Benchmark JSON file."""
    with open(json_file) as f:
        data = json.load(f)
    rows = []
    language, kernel = _parse_source_info(json_file)
    source = f"{language}_{kernel}"
    for bench in data.get("benchmarks", []):
        if bench.get("run_type") != "aggregate" or bench.get("aggregate_name") != "mean":
            continue
        name = bench.get("name", "")
        base_match = re.match(r"(?P<base>[^/]+)", name)
        base_name = base_match.group("base") if base_match else name
        params_str = name[len(base_name) :].lstrip("/")
        rows.append(
            {
                config.COL_BENCHMARK_NAME_BASE: base_name,
                "full_name": name,
                config.COL_SOURCE: source,
                config.COL_LANGUAGE: language,
                config.COL_KERNEL_TESTED: kernel,
                config.COL_MEAN_LATENCY: bench.get("real_time", 0) / 1_000_000.0,
                config.COL_THROUGHPUT: bench.get("items_per_second"),
                config.COL_PARAMS_STR: params_str,
            }
        )
    return rows


def parse_pytest_benchmark(json_file: Path) -> list[dict]:
    """Parse a pytest-benchmark JSON file."""
    with open(json_file) as f:
        data = json.load(f)
    rows = []
    language, kernel = _parse_source_info(json_file)
    source = f"{language}_{kernel}"
    for bench in data.get("benchmarks", []):
        name = bench.get("name", "")
        group = bench.get("group", "")
        base_name = group if group else name.split("[")[0]
        params_str = ""
        if "[" in name and "]" in name:
            params_str = name.split("[")[1].split("]")[0]
        stats = bench.get("stats", {})
        rows.append(
            {
                config.COL_BENCHMARK_NAME_BASE: base_name,
                "full_name": name,
                config.COL_SOURCE: source,
                config.COL_LANGUAGE: language,
                config.COL_KERNEL_TESTED: kernel,
                config.COL_MEAN_LATENCY: stats.get("mean", 0) * 1000.0,
                config.COL_PARAMS_STR: params_str,
                "rounds": stats.get("rounds"),
            }
        )
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
