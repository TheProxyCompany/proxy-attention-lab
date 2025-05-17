"""Plot latency for model configurations."""

from __future__ import annotations

import logging
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

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        filtered_df = cfg_df[cfg_df[COL_KERNEL_NAME] == kernel_filter]
        if filtered_df.empty:
            logger.warning(f"No data found for kernel '{kernel_filter}' in model_configs_latency plot.")
            return ""
        cfg_df = filtered_df

    if cfg_df.empty:
        logger.warning("No data available for model_configs_latency plot.")
        return ""

    # Log what we're working with
    logger.debug(f"Model config benchmark sources: {cfg_df[COL_SOURCE].unique().tolist()}")
    logger.debug(f"Model config names: {cfg_df['model_config_name'].unique().tolist()}")

    # Check if we have enough data to plot
    if len(cfg_df["model_config_name"].unique()) == 0:
        logger.warning("No model configurations available to plot")
        return ""

    if len(cfg_df[COL_SOURCE].unique()) == 0:
        logger.warning("No sources available in model configurations data")
        return ""

    # Create two plots: one for latency, one for throughput
    fig, (ax_latency, ax_throughput) = plt.subplots(2, 1, figsize=(10, 12), sharex=True)

    # Get unique model config names and sources for grouping
    model_configs = cfg_df["model_config_name"].unique()
    sources = cfg_df[COL_SOURCE].unique()

    # Log what we're plotting
    logger.info(f"Plotting model configs: {model_configs.tolist()} for sources: {sources.tolist()}")

    # Number of groups and bars
    n_configs = len(model_configs)
    n_sources = len(sources)

    # Set the width of a bar and positions of the bars
    bar_width = 0.8 / max(1, n_sources)  # Avoid division by zero
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

        # Check if we have data for this source
        if source_data.empty or len(source_data) == 0:
            logger.warning(f"No data for source {source} in model configs")
            continue

        logger.debug(f"Processing model config data for source {source}: {len(source_data)} rows")

        # Create mapping from model config to latency/throughput for this source
        latency_map = {row["model_config_name"]: row[COL_MEAN_LATENCY] for _, row in source_data.iterrows()}

        throughput_map = {
            row["model_config_name"]: row[COL_THROUGHPUT]
            for _, row in source_data.iterrows()
            if pd.notna(row[COL_THROUGHPUT])
        }

        # Check if we have latency data to plot
        if not latency_map:
            logger.warning(f"No latency data for source {source} in model configs")
            continue

        # Log the data we're plotting
        logger.debug(f"Latency data for {source}: {latency_map}")
        logger.debug(f"Throughput data for {source}: {throughput_map}")

        # Plot latency bars
        latency_values = [latency_map.get(config, np.nan) for config in model_configs]

        # Skip if all values are NaN
        if all(np.isnan(v) for v in latency_values):
            logger.warning(f"All latency values are NaN for source {source}")
            continue

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
        for j, bar in enumerate(bars):
            if not np.isnan(bar.get_height()):
                config_name = model_configs[j] if j < len(model_configs) else "unknown"
                logger.debug(f"Bar for {source} - {config_name}: {bar.get_height():.1f}")
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

            # Skip if all values are NaN
            if all(np.isnan(v) for v in throughput_values):
                logger.warning(f"All throughput values are NaN for source {source}")
                continue

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
            for j, bar in enumerate(bars):
                if not np.isnan(bar.get_height()):
                    config_name = model_configs[j] if j < len(model_configs) else "unknown"
                    logger.debug(f"Throughput bar for {source} - {config_name}: {bar.get_height():.1f}")
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
