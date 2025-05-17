"""Plot latency for model configurations."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from benchmarks.analyzer import plot_utils
from benchmarks.analyzer.config import (
    COL_KERNEL_NAME,
    COL_MEAN_LATENCY,
    COL_SOURCE,
    COL_THROUGHPUT,
)

STYLES = plot_utils.get_plot_styles()


def plot(
    df: pd.DataFrame, output_dir: Path, styles: dict[str, str | float] | None = None, kernel_filter: str | None = None
) -> str:
    """
    Generate model configuration latency and throughput bar charts.

    Creates a grouped bar chart comparing model configurations across different sources
    (cpp_pal, python_pal, cpp_sdpa, python_sdpa) for both latency and throughput.

    Args:
        df: DataFrame with benchmark results.
        output_dir: Output directory for the generated plot.
        styles: Plot style dictionary.
        kernel_filter: Optional kernel name to filter by (e.g., "paged_attention", "sdpa").

    Returns:
        Filename of the generated plot.
    """
    styles = styles or STYLES

    # Filter out rows without model_config_name
    cfg_df = df.dropna(subset=["model_config_name"])

    # Filter by kernel if specified
    if kernel_filter and COL_KERNEL_NAME in cfg_df.columns:
        cfg_df = cfg_df[cfg_df[COL_KERNEL_NAME] == kernel_filter]

    if cfg_df.empty:
        return ""

    # Create two plots: one for latency, one for throughput
    fig, (ax_latency, ax_throughput) = plt.subplots(2, 1, figsize=(10, 12), sharex=True)

    # Get unique model config names and sources for grouping
    model_configs = cfg_df["model_config_name"].unique()
    sources = cfg_df[COL_SOURCE].unique()

    # Number of groups and bars
    n_configs = len(model_configs)
    n_sources = len(sources)

    # Set the width of a bar and positions of the bars
    bar_width = 0.8 / n_sources
    index = np.arange(n_configs)

    # Color mapping for sources
    color_map = {
        "cpp_pal": styles["PAL_CPP_COLOR"],
        "python_pal": styles["PAL_PY_COLOR"],
        "cpp_sdpa": styles["SDPA_CPP_COLOR"],
        "python_sdpa": styles["SDPA_PY_COLOR"],
    }

    # Label mapping for sources
    label_map = {
        "cpp_pal": styles["PAL_CPP_LABEL"],
        "python_pal": styles["PAL_PY_LABEL"],
        "cpp_sdpa": styles["SDPA_CPP_LABEL"],
        "python_sdpa": styles["SDPA_PY_LABEL"],
    }

    # Hatch pattern for visual differentiation
    hatch_map = {"cpp_pal": "//", "python_pal": "", "cpp_sdpa": "\\\\", "python_sdpa": "."}

    # Plot latency and throughput for each source
    for i, source in enumerate(sources):
        source_data = cfg_df[cfg_df[COL_SOURCE] == source]

        # Create mapping from model config to latency/throughput for this source
        latency_map = {row["model_config_name"]: row[COL_MEAN_LATENCY] for _, row in source_data.iterrows()}

        throughput_map = {
            row["model_config_name"]: row[COL_THROUGHPUT]
            for _, row in source_data.iterrows()
            if pd.notna(row[COL_THROUGHPUT])
        }

        # Plot latency bars
        latency_values = [latency_map.get(config, np.nan) for config in model_configs]
        bars = ax_latency.bar(
            index + i * bar_width,
            latency_values,
            bar_width,
            label=label_map.get(source, source),
            color=color_map.get(source, "gray"),
            hatch=hatch_map.get(source, ""),
            edgecolor="black",
            alpha=0.9,
        )

        # Add value labels on top of bars
        for _j, bar in enumerate(bars):
            if not np.isnan(bar.get_height()):
                ax_latency.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.1,
                    f"{bar.get_height():.1f}",
                    ha="center",
                    va="bottom",
                    rotation=90,
                    fontsize=styles.get("LEGEND_FONTSIZE", 8),
                )

        # Plot throughput bars if data exists
        if throughput_map:
            throughput_values = [throughput_map.get(config, np.nan) for config in model_configs]
            bars = ax_throughput.bar(
                index + i * bar_width,
                throughput_values,
                bar_width,
                label=label_map.get(source, source),
                color=color_map.get(source, "gray"),
                hatch=hatch_map.get(source, ""),
                edgecolor="black",
                alpha=0.9,
            )

            # Add value labels on top of bars
            for _j, bar in enumerate(bars):
                if not np.isnan(bar.get_height()):
                    ax_throughput.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.1,
                        f"{bar.get_height():.1f}",
                        ha="center",
                        va="bottom",
                        rotation=90,
                        fontsize=styles.get("LEGEND_FONTSIZE", 8),
                    )

    # Set title based on whether we're filtering by kernel
    title_prefix = ""
    if kernel_filter:
        # Format the kernel name for display
        kernel_display = kernel_filter.replace("_", " ").title()
        title_prefix = f"{kernel_display}: "

    # Set latency plot aesthetics
    plot_utils.apply_common_plot_aesthetics(
        ax_latency,
        f"{title_prefix}Model Configuration Latency",
        "",  # No x-label for latency (shared with throughput)
        "Mean Latency (ms)",
        styles,
        include_legend=True,
    )

    # Set throughput plot aesthetics
    plot_utils.apply_common_plot_aesthetics(
        ax_throughput,
        f"{title_prefix}Model Configuration Throughput",
        "Model Configuration",
        "Throughput (items/sec)",
        styles,
        include_legend=False,  # Already included in latency plot
    )

    # Set x-ticks and labels
    ax_throughput.set_xticks(index + (n_sources - 1) * bar_width / 2)
    ax_throughput.set_xticklabels(model_configs, rotation=45, ha="right")

    # Adjust layout
    plt.tight_layout()

    # Construct filename based on kernel filter
    filename = f"{kernel_filter}_model_configs_comparison.png" if kernel_filter else "model_configs_comparison.png"

    # Save figure
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / filename, dpi=300)
    plt.close(fig)

    return filename
