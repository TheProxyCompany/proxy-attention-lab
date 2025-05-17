"""Plot latency as a function of effective items."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from analyzer import plot_utils
from analyzer.config import COL_BENCHMARK_NAME_BASE, COL_MEAN_LATENCY, COL_SOURCE, COL_THROUGHPUT

STYLES = plot_utils.get_plot_styles()


def plot(df: pd.DataFrame, output_dir: Path, styles: dict[str, str | float] | None = None) -> str:
    """
    Generate latency vs effective items plots.

    Creates two plots:
    1. Latency vs effective items
    2. Throughput vs effective items (if throughput data available)

    Args:
        df: DataFrame with benchmark results.
        output_dir: Output directory for the generated plots.
        styles: Plot style dictionary.

    Returns:
        Filename of the generated latency plot.
    """
    styles = styles or STYLES

    # Filter data for this plot type
    benchmark_names = [
        "BM_PAL_LatencyVsNumItems",
        "test_pal_latency_vs_query_items",
        "BM_SDPA_LatencyVsNumItems",
        "test_sdpa_latency_vs_batch_size",
    ]
    plot_df = df[df[COL_SOURCE].notna() & df["effective_items"].notna() & df[COL_MEAN_LATENCY].notna()]

    # Further filter if we have benchmark_name_base information
    if COL_BENCHMARK_NAME_BASE in plot_df.columns and set(benchmark_names) & set(
        plot_df[COL_BENCHMARK_NAME_BASE].unique()
    ):
        plot_df = plot_df[plot_df[COL_BENCHMARK_NAME_BASE].isin(benchmark_names)]

    if plot_df.empty:
        return ""

    # Create figure with two subplots: latency and throughput
    fig, (ax_latency, ax_throughput) = plt.subplots(1, 2, figsize=(15, 6))

    # Style mapping for different sources
    source_styles = {
        "cpp_pal": {
            "color": styles["PAL_CPP_COLOR"],
            "linestyle": styles["PAL_CPP_STYLE"],
            "linewidth": styles["PAL_LINEWIDTH"],
            "marker": styles["PAL_CPP_MARKER"],
            "label": styles["PAL_CPP_LABEL"],
        },
        "python_pal": {
            "color": styles["PAL_PY_COLOR"],
            "linestyle": styles["PAL_PY_STYLE"],
            "linewidth": styles["PAL_LINEWIDTH"],
            "marker": styles["PAL_PY_MARKER"],
            "label": styles["PAL_PY_LABEL"],
        },
        "cpp_sdpa": {
            "color": styles["SDPA_CPP_COLOR"],
            "linestyle": styles["SDPA_CPP_STYLE"],
            "linewidth": styles["SDPA_LINEWIDTH"],
            "marker": styles["SDPA_CPP_MARKER"],
            "label": styles["SDPA_CPP_LABEL"],
        },
        "python_sdpa": {
            "color": styles["SDPA_PY_COLOR"],
            "linestyle": styles["SDPA_PY_STYLE"],
            "linewidth": styles["SDPA_LINEWIDTH"],
            "marker": styles["SDPA_PY_MARKER"],
            "label": styles["SDPA_PY_LABEL"],
        },
    }

    # Plot latency for each source
    for src, group in plot_df.groupby(COL_SOURCE):
        # Sort by effective items
        group = group.sort_values("effective_items")

        # Get style for this source

        source_style = source_styles.get(str(src), {})
        if not source_style:
            # Default style if source not in mapping
            source_style = {"color": "gray", "linestyle": "-", "linewidth": 2, "marker": "o", "label": src}

        # Plot latency
        ax_latency.plot(group["effective_items"], group[COL_MEAN_LATENCY], **source_style)

        # Plot throughput if available
        if COL_THROUGHPUT in group.columns and not group[COL_THROUGHPUT].isna().all():
            ax_throughput.plot(group["effective_items"], group[COL_THROUGHPUT], **source_style)

    # Add reference lines
    items = plot_df["effective_items"].unique()
    if len(items) > 1:
        min_items = min(items)
        max_items = max(items)

        # Linear reference line O(n)
        x_linear = np.array([min_items, max_items])
        scale_factor = min(plot_df[COL_MEAN_LATENCY]) / min_items
        y_linear = x_linear * scale_factor

        ax_latency.plot(
            x_linear,
            y_linear,
            color=styles["REF_LINE_COLOR"],
            linestyle=styles["REF_LINE_STYLE"],
            linewidth=styles["REF_LINE_WIDTH"],
            alpha=styles["REF_LINE_ALPHA"],
            label="O(n)",
        )

        # Constant throughput reference (y = x)
        if not plot_df[COL_THROUGHPUT].isna().all():
            plot_df[COL_THROUGHPUT].min()
            x_throughput = np.array([min_items, max_items])
            y_throughput = x_throughput

            ax_throughput.plot(
                x_throughput,
                y_throughput,
                color=styles["REF_LINE_COLOR"],
                linestyle=styles["REF_LINE_STYLE"],
                linewidth=styles["REF_LINE_WIDTH"],
                alpha=styles["REF_LINE_ALPHA"],
                label="Linear Scaling",
            )

    # Set latency plot aesthetics
    plot_utils.apply_common_plot_aesthetics(
        ax_latency,
        "Latency vs Effective Items",
        "Effective Items",
        "Mean Latency (ms)",
        styles,
        x_scale="log",
        y_scale="log",
        include_legend=True,
    )

    # Set throughput plot aesthetics
    plot_utils.apply_common_plot_aesthetics(
        ax_throughput,
        "Throughput vs Effective Items",
        "Effective Items",
        "Throughput (items/sec)",
        styles,
        x_scale="log",
        y_scale="log",
        include_legend=False,  # Legend already in latency plot
    )

    # Adjust layout
    plt.tight_layout()

    # Save figure
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = "latency_vs_effective_items.png"
    fig.savefig(output_dir / filename, dpi=300)
    plt.close(fig)

    return filename
