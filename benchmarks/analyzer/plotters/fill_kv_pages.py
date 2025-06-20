"""Plot fill KV pages benchmark results."""

from __future__ import annotations

import json
import logging
from collections import OrderedDict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from benchmarks.analyzer.core import BenchmarkData, plot_styles
from benchmarks.analyzer.plotters.base import BasePlotter

logger = logging.getLogger(__name__)

STYLES = plot_styles.get_plot_styles()


class FillKVPagesPlotter(BasePlotter):
    """Plotter for fill KV pages benchmark visualization."""

    def get_name(self) -> str:
        return "fill_kv_pages"

    def get_required_fields(self) -> list[str]:
        return ["name"]

    def plot(self, data: BenchmarkData, output_dir: Path, **kwargs) -> dict[str, Any]:
        """
        Generate fill KV pages benchmark plots.

        Creates multiple subplots for different benchmark scenarios:
        - Multiple tokens single sequence
        - Single token multiple sequences
        - Varying dimensions
        - Varying page sizes

        Args:
            data: BenchmarkData with benchmark results.
            output_dir: Output directory for the generated plots.
            **kwargs: Additional options (e.g., styles).

        Returns:
            Dictionary with plot metadata and results.
        """
        self.ensure_output_dir(output_dir)
        df = data.df
        kwargs.get("styles", STYLES)

        # Filter for fill KV pages benchmarks
        fill_benchmarks = df[df["name"].str.contains("fill", case=False, na=False)]

        if fill_benchmarks.empty:
            logger.warning("No fill KV pages benchmarks found in the data")
            return {}

        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()

        # Parse benchmark data and organize by type
        multi_token_data = {}
        multi_seq_data = {}
        dim_data = {}
        page_size_data = {}

        for _, row in fill_benchmarks.iterrows():
            name = row["name"]
            # Get mean time from mean_latency field (already in milliseconds)
            if "mean_latency" not in row or pd.isna(row["mean_latency"]):
                continue
            mean_time = row["mean_latency"]

            # Parse parameters from benchmark name
            if "test_fill_multiple_tokens_single_sequence" in name:
                # Extract number of tokens and dtype from parameter string like [dtype0-128]
                try:
                    param_str = name.split("[")[-1].split("]")[0]
                    parts = param_str.split("-")
                    if len(parts) >= 2:
                        dtype_str = parts[0]  # e.g., 'dtype0' or 'dtype1'
                        num_tokens = int(parts[1])

                        # Determine dtype based on dtype0/dtype1
                        key = "float16" if dtype_str == "dtype0" else "bfloat16"

                        if key not in multi_token_data:
                            multi_token_data[key] = []
                        multi_token_data[key].append((num_tokens, mean_time))
                except (ValueError, IndexError):
                    continue

            elif "test_fill_single_token_multiple_sequences" in name:
                # Extract number of sequences and dtype from parameter string
                try:
                    param_str = name.split("[")[-1].split("]")[0]
                    parts = param_str.split("-")
                    if len(parts) >= 2:
                        dtype_str = parts[0]
                        num_sequences = int(parts[1])

                        key = "float16" if dtype_str == "dtype0" else "bfloat16"

                        if key not in multi_seq_data:
                            multi_seq_data[key] = []
                        multi_seq_data[key].append((num_sequences, mean_time))
                except (ValueError, IndexError):
                    continue

            elif "test_fill_varying_dimensions" in name:
                # Extract head_dim and num_heads from parameter string
                try:
                    param_str = name.split("[")[-1].split("]")[0]
                    parts = param_str.split("-")
                    if len(parts) >= 3:
                        # Format: dtype0-num_heads-head_dim
                        num_heads = int(parts[1])
                        head_dim = int(parts[2])
                        key = f"{head_dim}x{num_heads}"
                        dim_data[key] = mean_time
                except (ValueError, IndexError):
                    continue

            elif "test_fill_varying_page_size" in name:
                # Extract page size from parameter string
                try:
                    param_str = name.split("[")[-1].split("]")[0]
                    parts = param_str.split("-")
                    if len(parts) >= 2:
                        # Format: dtype0-page_size
                        page_size = int(parts[1])
                        page_size_data[page_size] = mean_time
                except (ValueError, IndexError):
                    continue

        # Plot 1: Multiple tokens single sequence
        ax = axes[0]
        if multi_token_data:
            for dtype, data_points in multi_token_data.items():
                data_points.sort(key=lambda x: x[0])
                tokens, times = zip(*data_points, strict=False)
                color = STYLES["PAL_COLOR"] if dtype == "float16" else STYLES["MLX_COLOR"]
                ax.plot(
                    tokens, times, marker="o", linewidth=2, markersize=8, label=f"Fill KV Pages ({dtype})", color=color
                )

            ax.set_xlabel("Number of Tokens", fontsize=12)
            ax.set_ylabel("Mean Latency (ms)", fontsize=12)
            ax.set_title("Fill Multiple Tokens - Single Sequence", fontsize=14, fontweight="bold")
            ax.set_xscale("log", base=2)
            ax.set_yscale("log")
            ax.grid(True, alpha=0.3, linestyle="--")
            ax.legend()

        # Plot 2: Single token multiple sequences
        ax = axes[1]
        if multi_seq_data:
            for dtype, data_points in multi_seq_data.items():
                data_points.sort(key=lambda x: x[0])
                sequences, times = zip(*data_points, strict=False)
                color = STYLES["PAL_COLOR"] if dtype == "float16" else STYLES["MLX_COLOR"]
                ax.plot(
                    sequences,
                    times,
                    marker="s",
                    linewidth=2,
                    markersize=8,
                    label=f"Fill KV Pages ({dtype})",
                    color=color,
                )

            ax.set_xlabel("Number of Sequences", fontsize=12)
            ax.set_ylabel("Mean Latency (ms)", fontsize=12)
            ax.set_title("Fill Single Token - Multiple Sequences (Batched Decode)", fontsize=14, fontweight="bold")
            ax.set_xscale("log", base=2)
            ax.set_yscale("log")
            ax.grid(True, alpha=0.3, linestyle="--")
            ax.legend()

        # Plot 3: Varying dimensions
        ax = axes[2]
        if dim_data:
            labels = list(dim_data.keys())
            values = list(dim_data.values())
            x_pos = np.arange(len(labels))

            bars = ax.bar(x_pos, values, color=STYLES["PAL_COLOR"], alpha=0.8, edgecolor="black")
            ax.set_xlabel("Head Dimension x Number of Heads", fontsize=12)
            ax.set_ylabel("Mean Latency (ms)", fontsize=12)
            ax.set_title("Fill Performance by Model Dimensions", fontsize=14, fontweight="bold")
            ax.set_xticks(x_pos)
            ax.set_xticklabels(labels, rotation=45, ha="right")
            ax.grid(True, alpha=0.3, linestyle="--", axis="y")

            # Add value labels on bars
            for bar, value in zip(bars, values, strict=False):
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0, height, f"{value:.2f}", ha="center", va="bottom", fontsize=9
                )

        # Plot 4: Varying page sizes
        ax = axes[3]
        if page_size_data:
            page_sizes = sorted(page_size_data.keys())
            times = [page_size_data[ps] for ps in page_sizes]

            ax.plot(
                page_sizes,
                times,
                marker="D",
                linewidth=2,
                markersize=8,
                color=STYLES["PAL_COLOR"],
                label="Fill KV Pages",
            )
            ax.set_xlabel("Tokens per Page", fontsize=12)
            ax.set_ylabel("Mean Latency (ms)", fontsize=12)
            ax.set_title("Fill Performance by Page Size", fontsize=14, fontweight="bold")
            ax.set_xscale("log", base=2)
            ax.grid(True, alpha=0.3, linestyle="--")
            ax.legend()

        # Adjust layout and save
        plt.tight_layout()
        filename = "fill_kv_pages_benchmarks.png"
        fig.savefig(output_dir / filename, dpi=300, bbox_inches="tight")
        plt.close(fig)

        # Save results as JSON
        results = {
            "fill_kv_pages": {
                "multiple_tokens_single_sequence": {
                    dtype: OrderedDict(sorted(data_points, key=lambda x: x[0]))
                    for dtype, data_points in multi_token_data.items()
                },
                "single_token_multiple_sequences": {
                    dtype: OrderedDict(sorted(data_points, key=lambda x: x[0]))
                    for dtype, data_points in multi_seq_data.items()
                },
                "varying_dimensions": dim_data,
                "varying_page_sizes": OrderedDict(sorted(page_size_data.items())),
            }
        }

        # Update the main results.json
        main_results_file = output_dir / "results.json"
        if main_results_file.exists():
            with open(main_results_file) as f:
                all_results = json.load(f)
        else:
            all_results = {}

        all_results.update(results)

        with open(main_results_file, "w") as f:
            json.dump(all_results, f, indent=4)

        return {
            "filename": filename,
            "main_results_json": main_results_file,
            "total_benchmarks": len(fill_benchmarks),
            "benchmark_type": "fill_kv_pages",
        }
