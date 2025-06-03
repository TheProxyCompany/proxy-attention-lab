"""Plot batch decode latency and throughput metrics."""

from __future__ import annotations

import json
import logging
from collections import OrderedDict
from pathlib import Path
from typing import Any

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from benchmarks.analyzer.core import BenchmarkData, plot_styles
from benchmarks.analyzer.plotters.base import BasePlotter

logger = logging.getLogger(__name__)

STYLES = plot_styles.get_plot_styles()


class DecodeBatchLatencyPlotter(BasePlotter):
    """Plotter for batch decode latency and throughput visualization."""

    def get_name(self) -> str:
        return "batch_decode_latency"

    def get_required_fields(self) -> list[str]:
        return ["mean_latency", "group", "name"]

    def plot(self, data: BenchmarkData, output_dir: Path, **kwargs) -> dict[str, Any]:
        """
        Generate batch decode latency and throughput plots.

        Creates a 2x2 grid with:
        - Top left: Batch Latency vs History Length
        - Top right: Effective Decode Tokens/Sec vs History Length

        Args:
            data: BenchmarkData with benchmark results.
            output_dir: Output directory for the generated plots.
            **kwargs: Additional options (e.g., styles).

        Returns:
            Dictionary with plot metadata and results.
        """
        self.ensure_output_dir(output_dir)
        df = data.df
        styles = kwargs.get("styles", STYLES)

        # Filter for batch decode benchmarks
        batch_benchmarks = df[df["name"].str.contains("BatchLatency", case=False)]

        if batch_benchmarks.empty:
            logger.warning("No batch decode benchmarks found")
            return {"error": "No batch decode benchmarks found"}

        # Separate benchmarks by type
        vs_history = batch_benchmarks[batch_benchmarks["name"].str.contains("VsHistoryLength")]

        # Create 2x2 subplot grid
        fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(15, 6))

        # Define styles for PAL and MLX following the pattern from latency_vs_seq_len
        impl_styles = {
            "pal": {
                "color": styles["PAL_COLOR"],
                "outline_color": styles["PAL_OUTLINE_COLOR"],
                "linestyle": styles["PAL_STYLE"],
                "linewidth": styles["PAL_LINEWIDTH"],
                "marker": styles["PAL_MARKER"],
                "label": "PAL",
            },
            "mlx": {
                "color": styles["MLX_COLOR"],
                "outline_color": styles["MLX_OUTLINE_COLOR"],
                "linestyle": styles["MLX_STYLE"],
                "linewidth": styles["MLX_LINEWIDTH"],
                "marker": styles["MLX_MARKER"],
                "label": "MLX SDPA",
            },
        }

        # Plot 2: Batch Latency vs History Length
        if "history_length" in vs_history.columns:
            num_sequences = sorted(vs_history["num_sequences"].unique())

            # Group by implementation (PAL vs MLX)
            for group_name, group_data in vs_history.groupby("group"):
                # Extract implementation from simplified group names (e.g., "cpp_pal", "cpp_mlx")
                if "_pal" in str(group_name):
                    impl_name = "pal"
                elif "_mlx" in str(group_name):
                    impl_name = "mlx"
                else:
                    logger.warning(f"Unknown group name: {group_name}")
                    continue
                impl_style = impl_styles[impl_name]

                for idx, num_seq in enumerate(num_sequences):
                    subset = group_data[group_data["num_sequences"] == num_seq]
                    if subset.empty:
                        continue
                    sorted_subset = subset.sort_values("history_length")

                    # Use different line styles for different batch sizes
                    linestyle = ["-", "--", ":", "-."][idx % 4]

                    (line,) = ax1.plot(
                        sorted_subset["history_length"],
                        sorted_subset["mean_latency"],
                        marker=impl_style["marker"],
                        linewidth=impl_style["linewidth"],
                        linestyle=linestyle,
                        markersize=8,
                        color=impl_style["color"],
                        label=f"{impl_style['label']} (N={num_seq})",
                    )

                    # Apply outline effect
                    if "outline_color" in impl_style and isinstance(line, Line2D):
                        line.set_path_effects(
                            [
                                pe.Stroke(
                                    linewidth=impl_style["linewidth"] * 3,
                                    foreground=impl_style["outline_color"],
                                ),
                                pe.Normal(),
                            ]
                        )

                    # Calculate throughput
                    throughput = num_seq / (sorted_subset["mean_latency"] / 1000)  # Convert ms to seconds

                    (line,) = ax2.plot(
                        sorted_subset["history_length"],
                        throughput,
                        marker=impl_style["marker"],
                        linewidth=impl_style["linewidth"],
                        linestyle=linestyle,
                        markersize=8,
                        color=impl_style["color"],
                        label=f"{impl_style['label']} (N={num_seq})",
                    )

                    # Apply outline effect
                    if "outline_color" in impl_style and isinstance(line, Line2D):
                        line.set_path_effects(
                            [
                                pe.Stroke(
                                    linewidth=impl_style["linewidth"] * 3,
                                    foreground=impl_style["outline_color"],
                                ),
                                pe.Normal(),
                            ]
                        )

        # Configure plot 2: Batch Latency vs History Length
        ax1.set_xlabel("History Length (H)", fontsize=12)
        ax1.set_ylabel("Batch Latency (ms)", fontsize=12)
        ax1.set_title("Fused Kernel: Batch Latency vs. History Length", fontsize=14, fontweight="bold")
        ax1.set_xscale("log")
        ax1.set_yscale("log")
        ax1.grid(True, alpha=0.3, which="both")
        ax1.legend(title="Num Sequences", fontsize=10)
        ax1.minorticks_on()
        # Configure plot 4: Effective Decode Tokens/Sec vs History Length
        ax2.set_xlabel("History Length (H)", fontsize=12)
        ax2.set_ylabel("Effective Decode Tokens/Sec", fontsize=12)
        ax2.set_title("Fused Kernel: Effective Decode Tokens/Sec vs. History Length", fontsize=14, fontweight="bold")
        ax2.set_xscale("log")
        ax2.set_yscale("log")
        ax2.grid(True, alpha=0.3, which="both")
        ax2.legend(title="Num Sequences", fontsize=10)
        ax2.minorticks_on()

        # ── Deduplicate legend entries and keep things compact ──
        for ax in (ax1, ax2):
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                uniq = dict(zip(labels, handles, strict=True))
                ax.legend(
                    uniq.values(),
                    uniq.keys(),
                    fontsize=10,
                    ncol=2,
                    title=ax.get_legend().get_title().get_text() if ax.get_legend() else None,
                )

        # Adjust layout
        plt.tight_layout()

        # Save figure
        filename = "batch_decode_latency.png"
        fig.savefig(output_dir / filename, dpi=300)
        plt.close(fig)

        # Prepare results for JSON output following latency_vs_seq_len pattern
        results = {
            "batch_decode_latency": {
                "_description": "Batch decode performance metrics for PAL and MLX SDPA",
                "_units": {
                    "latency": "milliseconds (ms)",
                    "throughput": "tokens per second",
                    "history_length": "number of tokens in KV cache",
                },
                "latency_vs_history_length": {},
                "throughput_vs_history_length": {},
            }
        }

        # Process data by group (implementation) following latency_vs_seq_len pattern
        for group, group_df in batch_benchmarks.groupby("group"):
            # Use the group name directly as it's already simplified (e.g., "cpp_pal", "cpp_mlx")
            impl_name = str(group)
            if not ("_pal" in impl_name or "_mlx" in impl_name):
                logger.warning(f"Skipping unknown group: {group}")
                continue

            # ---- Latency & throughput vs history length for this impl ----
            vs_history_impl = group_df[group_df["name"].str.contains("VsHistoryLength")]
            if not vs_history_impl.empty and "history_length" in vs_history_impl.columns:
                for num_seq in sorted(vs_history_impl["num_sequences"].unique()):
                    subset = vs_history_impl[vs_history_impl["num_sequences"] == num_seq]
                    key = f"num_sequences_{num_seq}"
                    if key not in results["batch_decode_latency"]["latency_vs_history_length"]:
                        results["batch_decode_latency"]["latency_vs_history_length"][key] = {}
                        results["batch_decode_latency"]["throughput_vs_history_length"][key] = {}

                    # Store latency data (sorted by history_length)
                    latency_pairs = [
                        (float(hist), float(latency))
                        for hist, latency in zip(subset["history_length"], subset["mean_latency"], strict=True)
                    ]
                    latency_map = OrderedDict(sorted(latency_pairs, key=lambda x: x[0]))
                    results["batch_decode_latency"]["latency_vs_history_length"][key][impl_name] = latency_map

                    # Calculate and store throughput (sorted by history_length)
                    throughput_pairs = [
                        (float(hist), float(num_seq / (latency / 1000)))
                        for hist, latency in zip(subset["history_length"], subset["mean_latency"], strict=True)
                    ]
                    throughput_map = OrderedDict(sorted(throughput_pairs, key=lambda x: x[0]))
                    results["batch_decode_latency"]["throughput_vs_history_length"][key][impl_name] = throughput_map

        # Update the main results.json by merging with existing data
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
            "plots_created": [
                "latency_vs_history_length",
                "throughput_vs_history_length",
            ],
            "total_benchmarks": len(batch_benchmarks),
            "benchmark_type": "batch_decode_latency",
        }
