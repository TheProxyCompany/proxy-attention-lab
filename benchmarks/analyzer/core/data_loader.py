"""Data loading and processing for benchmark results."""

import json
import logging
from pathlib import Path
from typing import ClassVar

import pandas as pd

from benchmarks.analyzer.core.types import BenchmarkData

logger = logging.getLogger(__name__)


class DataLoader:
    """Handles loading and processing of benchmark data from various formats."""

    # Mapping of benchmark parameters to standardized column names
    DEFAULT_COLUMN_MAPPINGS: ClassVar[dict[str, str]] = {
        # Google Benchmark (C++)
        "param": "sequence_length",
        "real_time": "mean_latency",
        "cpu_time": "cpu_time",
        # pytest-benchmark (Python)
        "mean": "mean_latency",
        "stddev": "std_latency",
        "min": "min_latency",
        "max": "max_latency",
    }

    # Kernel type detection patterns
    KERNEL_PATTERNS: ClassVar[dict[str, list[str]]] = {
        "fused": ["decode", "fused", "batchdecode", "batchlatency"],
        "two_pass": ["prefill", "two_pass", "2pass", "prefillbatchlatency", "batchprefill"],
    }

    def __init__(self, column_mappings: dict[str, str] | None = None):
        self.column_mappings = column_mappings or self.DEFAULT_COLUMN_MAPPINGS

    def load_directory(self, directory: Path) -> BenchmarkData:
        """Load all benchmark JSON files from a directory."""
        # Skip results.json and other analyzer output files
        json_files = [f for f in directory.glob("*.json") if f.name not in ["results.json"]]
        if not json_files:
            raise ValueError(f"No JSON files found in {directory}")

        all_data = []
        source_files = []

        for json_file in json_files:
            logger.info(f"Processing {json_file}")
            try:
                df = self._load_file(json_file)
                if df is not None and not df.empty:
                    all_data.append(df)
                    source_files.append(json_file)
            except Exception as e:
                logger.error(f"Failed to process {json_file}: {e}")

        if not all_data:
            raise ValueError("No valid benchmark data found")

        # Combine all DataFrames
        combined_df = pd.concat(all_data, ignore_index=True)

        # Extract kernel types
        kernel_types = self._extract_kernel_types(combined_df)

        # Build metadata
        metadata = {
            "total_benchmarks": len(combined_df),
            "source_file_count": len(source_files),
            "kernel_types_found": list(set(kernel_types.values())),
        }

        return BenchmarkData(df=combined_df, source_files=source_files, kernel_types=kernel_types, metadata=metadata)

    def _load_file(self, file_path: Path) -> pd.DataFrame | None:
        """Load a single benchmark JSON file."""
        with open(file_path) as f:
            data = json.load(f)

        # Determine format and process accordingly
        if self._is_google_benchmark(data):
            return self._process_google_benchmark(data, file_path)
        elif self._is_pytest_benchmark(data):
            return self._process_pytest_benchmark(data, file_path)
        else:
            logger.warning(f"Unknown benchmark format in {file_path}")
            return None

    def _is_google_benchmark(self, data: dict) -> bool:
        """Check if data is in Google Benchmark format."""
        return "benchmarks" in data and "context" in data

    def _is_pytest_benchmark(self, data: dict) -> bool:
        """Check if data is in pytest-benchmark format."""
        return "benchmarks" in data and "machine_info" in data

    def _process_google_benchmark(self, data: dict, file_path: Path) -> pd.DataFrame:
        """Process Google Benchmark JSON data."""
        benchmarks = data.get("benchmarks", [])
        if not benchmarks:
            return pd.DataFrame()

        rows = []
        for benchmark in benchmarks:
            # Skip non-aggregate results for C++ benchmarks
            if benchmark.get("run_type") == "iteration":
                continue
            # Only process median aggregate results
            if benchmark.get("aggregate_name") != "median":
                continue

            row = {
                "name": benchmark.get("name", ""),
                "source": "cpp",
                "file": file_path.name,
            }

            # Extract timing information
            for key, mapped_key in self.column_mappings.items():
                if key in benchmark:
                    # Google Benchmark times are in nanoseconds, convert to milliseconds
                    row[mapped_key] = benchmark[key] / 1_000_000  # Convert ns to ms

            # Extract parameters based on benchmark type
            name = benchmark.get("name", "")
            parts = name.split("/")

            if "param" in benchmark:
                # Legacy: if param field exists, assume it's sequence_length
                row["sequence_length"] = int(benchmark["param"])
            elif len(parts) >= 2 and parts[1].isdigit():
                param_value = int(parts[1])

                # Determine parameter type from benchmark name
                # Check more specific patterns first
                if "PrefillBatchLatencyVsSeqLen" in name:
                    # First param is seq_len, second is num_sequences
                    row["sequence_length"] = param_value
                    if len(parts) >= 3 and parts[2].isdigit():
                        row["num_sequences"] = int(parts[2])
                elif "DecodeBatchLatencyVsHistoryLength" in name:
                    # First param is history_length, second is num_sequences
                    row["num_sequences"] = param_value
                    if len(parts) >= 3 and parts[2].isdigit():
                        row["history_length"] = int(parts[2])
                elif "PrefillLatencyVsSeqLen" in name:
                    row["sequence_length"] = param_value
                elif "DecodeLatencyVsHistoryLen" in name:
                    row["sequence_length"] = param_value  # For compatibility
                else:
                    row["sequence_length"] = param_value
            else:
                logger.debug(f"Could not extract parameters from benchmark: {name}")

            # Add group label
            row["group"] = self._create_group_label(row["name"], "cpp")
            rows.append(row)

        df = pd.DataFrame(rows)
        if df.empty:
            logger.warning(f"No valid benchmarks extracted from {file_path}")
        else:
            logger.info(f"Extracted {len(df)} benchmarks from {file_path}")
        return df

    def _process_pytest_benchmark(self, data: dict, file_path: Path) -> pd.DataFrame:
        """Process pytest-benchmark JSON data."""
        benchmarks = data.get("benchmarks", [])
        if not benchmarks:
            return pd.DataFrame()

        rows = []
        for benchmark in benchmarks:
            # Skip non-benchmark entries (like the machine info)
            if not isinstance(benchmark, dict) or "name" not in benchmark:
                continue

            row = {
                "name": benchmark.get("name", ""),
                "source": "python",
                "file": file_path.name,
            }

            # Extract timing stats
            stats = benchmark.get("stats", {})
            for key, mapped_key in self.column_mappings.items():
                if key in stats:
                    # Python benchmark times are in seconds, convert to milliseconds
                    row[mapped_key] = stats[key] * 1000  # Convert to milliseconds

            # Extract parameters
            if "params" in benchmark:
                params = benchmark["params"]
                if "sequence_length" in params:
                    row["sequence_length"] = int(params["sequence_length"])
                elif "seq_len_val" in params:
                    row["sequence_length"] = int(params["seq_len_val"])
                elif "history_len_val" in params:
                    row["sequence_length"] = int(params["history_len_val"])
            elif "param" in benchmark:
                row["sequence_length"] = int(benchmark["param"])

            # Add group label
            row["group"] = self._create_group_label(row["name"], "python")

            rows.append(row)

        df = pd.DataFrame(rows)
        if df.empty:
            logger.warning(f"No valid benchmarks extracted from {file_path}")
        else:
            logger.info(f"Extracted {len(df)} benchmarks from {file_path}")
        return df

    def _create_group_label(self, benchmark_name: str, source: str) -> str:
        """Create a group label for a benchmark."""
        # Simple group labels for better compatibility with plotters
        name_lower = benchmark_name.lower()

        # Check if it's a two-pass benchmark
        is_two_pass = "twopass" in name_lower or "two_pass" in name_lower

        if "pal" in name_lower:
            if is_two_pass:
                return f"{source}_pal_two_pass"
            else:
                return f"{source}_pal"
        elif "mlx" in name_lower:
            return f"{source}_mlx"
        else:
            return f"{source}_unknown"

    def _detect_kernel_type(self, benchmark_name: str) -> str:
        """Detect kernel type from benchmark name."""
        name_lower = benchmark_name.lower()

        for kernel_type, patterns in self.KERNEL_PATTERNS.items():
            for pattern in patterns:
                if pattern in name_lower:
                    return kernel_type

        # Default to two_pass if no pattern matches
        return "two_pass"

    def _extract_kernel_types(self, df: pd.DataFrame) -> dict[str, str]:
        """Extract kernel types for all benchmarks in the DataFrame."""
        kernel_types = {}
        for name in df["name"].unique():
            kernel_types[name] = self._detect_kernel_type(name)
        return kernel_types
