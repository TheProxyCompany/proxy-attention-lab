"""Plot latency as a function of sequence length."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from benchmarks.analyzer.core import BenchmarkData, plot_styles, register_plotter
from benchmarks.analyzer.plotters.base import BasePlotter

logger = logging.getLogger(__name__)

STYLES = plot_styles.get_plot_styles()

# Plot configuration constants
ERROR_BAR_CAPSIZE = 5
ERROR_BAR_CAPTHICK = 1
ERROR_BAR_ALPHA = 0.9
MLX_ERROR_COLOR = "#808080"  # Gray for visibility against dark background
REF_LINE_LABEL_OFFSET = 1.05
N2_LABEL_POSITION = 0.7
NLOGN_LABEL_POSITION = 0.45
N_LABEL_POSITION = 0.7


def calculate_log_log_slope(x_data: np.ndarray, y_data: np.ndarray) -> tuple[float, float] | None:
    """
    Calculate the slope and intercept in log-log space using linear regression.

    Returns:
        tuple of (slope, intercept) or None if calculation fails
    """
    try:
        # Remove any invalid values
        mask = (x_data > 0) & (y_data > 0) & np.isfinite(x_data) & np.isfinite(y_data)
        x_clean = x_data[mask]
        y_clean = y_data[mask]

        if len(x_clean) < 2:
            return None

        # Calculate slope in log-log space
        log_x = np.log(x_clean)
        log_y = np.log(y_clean)

        # Simple linear regression: y = mx + b
        A = np.vstack([log_x, np.ones(len(log_x))]).T
        slope, intercept = np.linalg.lstsq(A, log_y, rcond=None)[0]

        return slope, intercept
    except Exception as e:
        logger.warning(f"Failed to calculate slope: {e}")
        return None


def add_complexity_guide(ax, plot_type: str = "prefill") -> None:
    """Add an interpretation guide for log-log plots below the title."""
    if plot_type == "prefill":
        text = "Log-log: Slope ≈ 1 (linear), ≈ 2 (quadratic); lower = better scaling."
    else:  # decode
        text = "Log-log: Slope ≈ 1 (linear); lower = better scaling."

    # # Get current title and reposition it higher with padding
    current_title = ax.get_title()
    ax.set_title(current_title, fontsize=14, fontweight="bold", y=1.06)

    # Add guide text below the title
    ax.text(
        0.5,
        1.025,
        text,
        transform=ax.transAxes,
        fontsize=8,
        verticalalignment="bottom",
        horizontalalignment="center",
        color="gray",
        style="italic",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="lightgray", alpha=0.8, linewidth=1.0),
    )


@register_plotter
class LatencyVsSeqLenPlotter(BasePlotter):
    """Plotter for latency vs sequence length visualization."""

    def get_name(self) -> str:
        return "latency_vs_seq_len"

    def get_required_fields(self) -> list[str]:
        return ["sequence_length", "mean_latency", "group", "name"]

    def plot(self, data: BenchmarkData, output_dir: Path, **kwargs) -> dict[str, Any]:
        """
        Generate latency vs sequence length plots.

        Creates either:
        - Two side-by-side plots if both two_pass and fused data exist
        - Single plot if only one type of data exists

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

        source_styles = {
            "pal": {
                "short_name": "PAL",
                "name": "Proxy, Paged Attention",
                "color": styles["PAL_COLOR"],
                "outline_color": styles["PAL_OUTLINE_COLOR"],
                "linestyle": styles["PAL_STYLE"],
                "linewidth": styles["PAL_LINEWIDTH"],
                "marker": styles["PAL_MARKER"],
                "label": styles["PAL_LABEL"],
            },
            "mlx": {
                "short_name": "MLX",
                "name": "MLX, Scaled Dot Product Attention",
                "color": styles["MLX_COLOR"],
                "outline_color": styles["MLX_OUTLINE_COLOR"],
                "linestyle": styles["MLX_STYLE"],
                "linewidth": styles["MLX_LINEWIDTH"],
                "marker": styles["MLX_MARKER"],
                "label": styles["MLX_LABEL"],
            },
            "default": {
                "short_name": "Default",
                "name": "Default",
                "color": "gray",
                "outline_color": "black",
                "linestyle": "-",
                "linewidth": 2,
                "marker": "o",
                "label": "default",
            },
        }

        if "name" not in df.columns:
            raise ValueError("'name' column not found in DataFrame")

        # Split data into two_pass and fused benchmarks based on kernel types
        two_pass_names = [name for name, ktype in data.kernel_types.items() if ktype == "two_pass"]
        fused_names = [name for name, ktype in data.kernel_types.items() if ktype == "fused"]

        two_pass_df = df[df["name"].isin(two_pass_names)]
        fused_df = df[df["name"].isin(fused_names)]

        # Determine plot layout based on available data
        has_two_pass = not two_pass_df.empty
        has_fused = not fused_df.empty

        if has_two_pass and has_fused:
            fig, (ax_two_pass, ax_fused) = plt.subplots(1, 2, figsize=(15, 6))
        elif has_two_pass:
            fig, ax_two_pass = plt.subplots(1, 1, figsize=(8, 6))
            ax_fused = None
        elif has_fused:
            fig, ax_fused = plt.subplots(1, 1, figsize=(8, 6))
            ax_two_pass = None
        else:
            raise ValueError("No data available for plotting")

        # Process two_pass data
        if has_two_pass and ax_two_pass is not None:
            for src, group_data in two_pass_df.groupby("group"):
                sorted_group = group_data.sort_values(by="sequence_length")  # type: ignore[arg-type]

                if "pal" in str(src).lower():
                    source_style = source_styles["pal"]
                elif "mlx" in str(src).lower():
                    source_style = source_styles["mlx"]
                else:
                    source_style = source_styles["default"]

                # Plot with or without error bars
                if "std_latency" in sorted_group.columns and sorted_group["std_latency"].notna().any():
                    error_color = MLX_ERROR_COLOR if "mlx" in str(src).lower() else source_style["color"]
                    errorbar_container = ax_two_pass.errorbar(
                        sorted_group["sequence_length"],
                        sorted_group["mean_latency"],
                        yerr=sorted_group["std_latency"],
                        color=source_style["color"],
                        ecolor=error_color,
                        linestyle=source_style["linestyle"],
                        linewidth=source_style["linewidth"],
                        marker=source_style["marker"],
                        label=source_style["label"],
                        capsize=ERROR_BAR_CAPSIZE,
                        capthick=ERROR_BAR_CAPTHICK,
                        alpha=ERROR_BAR_ALPHA,
                    )
                    line = errorbar_container[0]
                else:
                    (line,) = ax_two_pass.plot(
                        sorted_group["sequence_length"],
                        sorted_group["mean_latency"],
                        color=source_style["color"],
                        linestyle=source_style["linestyle"],
                        linewidth=source_style["linewidth"],
                        marker=source_style["marker"],
                        label=source_style["label"],
                    )

                if "outline_color" in source_style and isinstance(line, Line2D):
                    line.set_path_effects(
                        [
                            pe.Stroke(
                                linewidth=source_style["linewidth"] * 3,
                                foreground=source_style["outline_color"],
                            ),
                            pe.Normal(),
                        ]
                    )

                x_data = sorted_group["sequence_length"].values
                y_data = sorted_group["mean_latency"].values
                assert isinstance(x_data, np.ndarray)
                assert isinstance(y_data, np.ndarray)

                # Calculate and display slope
                result = calculate_log_log_slope(x_data, y_data)
                if result is not None:
                    slope, intercept = result
                    label = source_style["label"].replace(r"$\mathbf{", "").replace(r"}$", "").replace("\\", "")
                    logger.info(f"Prefill {label}: slope = {slope:.2f} (O(n^{slope:.2f}))")

                    ax_two_pass.text(
                        x_data[-1] * 0.97,
                        y_data[-1] * 1.03,
                        f"Slope ≈ {slope:.2f}",
                        color="red",
                        fontsize=10,
                        fontweight="bold",
                        horizontalalignment="right",
                        verticalalignment="bottom",
                        bbox=dict(
                            boxstyle="round,pad=0.5",
                            facecolor=source_style["color"],
                            edgecolor=source_style["outline_color"],
                            alpha=0.5,
                            linewidth=1.0,
                        ),
                    )

            # Set prefill plot aesthetics
            title = "Prefill Latency vs Sequence Length"
            plot_styles.apply_common_plot_aesthetics(
                ax_two_pass,
                title,
                "Sequence Length (tokens)",
                "Mean Latency (ms)",
                styles,
                x_scale="log",
                y_scale="log",
                include_legend=True,
            )
            plt.subplots_adjust(top=0.9)
            add_complexity_guide(ax_two_pass, "prefill")

        # Process fused data
        if has_fused and ax_fused is not None:
            for src, group_data in fused_df.groupby("group"):
                src_label = str(src)
                if "pal" in src_label.lower():
                    src_label = "PAL"
                    source_style = source_styles["pal"].copy()
                elif "mlx" in src_label.lower():
                    src_label = "MLX"
                    source_style = source_styles["mlx"].copy()
                else:
                    source_style = source_styles["default"].copy()

                sorted_group = group_data.sort_values(by="sequence_length")  # type: ignore[arg-type]

                # Plot with or without error bars
                if "std_latency" in sorted_group.columns and sorted_group["std_latency"].notna().any():
                    error_color = MLX_ERROR_COLOR if "mlx" in str(src).lower() else source_style["color"]
                    errorbar_container = ax_fused.errorbar(
                        sorted_group["sequence_length"],
                        sorted_group["mean_latency"],
                        yerr=sorted_group["std_latency"],
                        color=source_style["color"],
                        ecolor=error_color,
                        linestyle=source_style["linestyle"],
                        linewidth=source_style["linewidth"],
                        marker=source_style["marker"],
                        label=source_style["label"],
                        capsize=ERROR_BAR_CAPSIZE,
                        capthick=ERROR_BAR_CAPTHICK,
                        alpha=ERROR_BAR_ALPHA,
                    )
                    line = errorbar_container[0]
                else:
                    (line,) = ax_fused.plot(
                        sorted_group["sequence_length"],
                        sorted_group["mean_latency"],
                        color=source_style["color"],
                        linestyle=source_style["linestyle"],
                        linewidth=source_style["linewidth"],
                        marker=source_style["marker"],
                        label=source_style["label"],
                    )

                if "outline_color" in source_style and isinstance(line, Line2D):
                    line.set_path_effects(
                        [
                            pe.Stroke(
                                linewidth=source_style["linewidth"] * 3,
                                foreground=source_style["outline_color"],
                            ),
                            pe.Normal(),
                        ]
                    )

                x_data = sorted_group["sequence_length"].values
                y_data = sorted_group["mean_latency"].values
                assert isinstance(x_data, np.ndarray)
                assert isinstance(y_data, np.ndarray)

                # Calculate and display slope
                result = calculate_log_log_slope(x_data, y_data)
                if result is not None:
                    slope, intercept = result
                    label = source_style["label"].replace(r"$\mathbf{", "").replace(r"}$", "").replace("\\", "")
                    logger.info(f"Decode {label}: slope = {slope:.2f} (O(n^{slope:.2f}))")

                ax_fused.text(
                    x_data[-2] * 0.92,
                    y_data[-2] * 1.05,
                    f"Slope ≈ {slope:.2f}",
                    color="red",
                    fontsize=10,
                    fontweight="bold",
                    horizontalalignment="right",
                    verticalalignment="bottom",
                    bbox=dict(
                        boxstyle="round,pad=0.5",
                        facecolor=source_style["color"],
                        edgecolor=source_style["outline_color"],
                        alpha=0.5,
                        linewidth=1.0,
                    ),
                )

            # Set decode plot aesthetics
            title = "Decode Latency vs History Length"
            plot_styles.apply_common_plot_aesthetics(
                ax_fused,
                title,
                "History Length (tokens)",
                "Mean Latency (ms)",
                styles,
                x_scale="log",
                y_scale="log",
                include_legend=True,
            )
            plt.subplots_adjust(top=0.9)
            add_complexity_guide(ax_fused, "decode")

        # Configure tick formatting
        if ax_two_pass is not None:
            ax_two_pass.minorticks_on()
            ax_two_pass.grid(linewidth=0.5, alpha=0.5)

        if ax_fused is not None:
            ax_fused.minorticks_on()
            ax_fused.grid(linewidth=0.5, alpha=0.5)

        # Save figure
        filename = "latency_vs_seq_len.png"
        fig.savefig(output_dir / filename, dpi=300)
        plt.close(fig)

        # Save results as JSON organized by benchmark type
        results = {"latency_vs_seq_len": {"prefill": {}, "decode": {}}}

        # Organize results by kernel type (prefill/decode) and implementation
        for group, group_df in df.groupby("group"):
            seq_lat_map = {
                float(seq_len): float(m_lat)
                for seq_len, m_lat in zip(group_df["sequence_length"], group_df["mean_latency"], strict=False)
            }

            # Determine if this is prefill (two_pass) or decode (fused)
            if "two_pass" in str(group).lower():
                benchmark_type = "prefill"
            elif "fused" in str(group).lower():
                benchmark_type = "decode"
            else:
                continue

            # Extract implementation name (e.g., "cpp_pal_paged_attention" or "cpp_mlx_sdpa")
            parts = str(group).split("_")
            if "pal" in str(group).lower():
                impl_name = f"{parts[0]}_pal"
            elif "mlx" in str(group).lower():
                impl_name = f"{parts[0]}_mlx"
            else:
                impl_name = group

            results["latency_vs_seq_len"][benchmark_type][impl_name] = seq_lat_map

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
            "kernel_types_plotted": list(set(data.kernel_types.values())),
            "total_groups": len(df["group"].unique()),
            "benchmark_type": "latency_vs_seq_len",
        }
