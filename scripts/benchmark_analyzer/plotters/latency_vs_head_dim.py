"""Plot latency as a function of head dimension."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .. import plot_utils
from ..config import COL_MEAN_LATENCY, COL_SOURCE, COL_THROUGHPUT

STYLES = plot_utils.get_plot_styles()


<<<<<<< HEAD
def plot(df: pd.DataFrame, output_dir: Path, styles: dict[str, str | float] | None = None) -> dict[str, str]:
    """Generate latency vs head dimension plots per kernel."""
    styles = styles or STYLES
    output_dir.mkdir(parents=True, exist_ok=True)
    filenames: dict[str, str] = {}
    for kernel, kdf in df.dropna(subset=["head_dim"]).groupby(COL_KERNEL_TESTED):
        fig, ax = plt.subplots(figsize=(8, 6))
        for lang, group in kdf.groupby(COL_LANGUAGE):
            ax.plot(group["head_dim"], group[COL_MEAN_LATENCY], label=lang, marker="o")
        plot_utils.apply_common_plot_aesthetics(
            ax,
            f"{kernel} Latency vs Head Dimension",
            "Head Dimension",
            "Mean Latency (ms)",
            styles,
        )
        filename = f"latency_vs_head_dim_{kernel}.png"
        fig.savefig(output_dir / filename, dpi=300)
        plt.close(fig)
        filenames[kernel] = filename
    return filenames
=======
def plot(df: pd.DataFrame, output_dir: Path, styles: dict[str, str | float] | None = None) -> str:
    """
    Generate latency vs head dimension plots.

    Creates two plots:
    1. Latency vs head dimension
    2. Throughput vs head dimension (if throughput data available)

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
        "BM_PAL_LatencyVsHeadDim",
        "test_pal_latency_vs_head_dim",
        "BM_SDPA_LatencyVsHeadDim",
        "test_sdpa_latency_vs_head_dim",
    ]
    plot_df = df[df[COL_SOURCE].notna() & df["head_dim"].notna() & df[COL_MEAN_LATENCY].notna()]

    # Further filter if we have benchmark_name_base information
    if plot_utils.COL_BENCHMARK_NAME_BASE in plot_df.columns:
        if set(benchmark_names).intersection(set(plot_df[plot_utils.COL_BENCHMARK_NAME_BASE].unique())):
            plot_df = plot_df[plot_df[plot_utils.COL_BENCHMARK_NAME_BASE].isin(benchmark_names)]

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
        # Sort by head dimension
        group = group.sort_values("head_dim")

        # Get style for this source
        source_style = source_styles.get(src, {})
        if not source_style:
            # Default style if source not in mapping
            source_style = {"color": "gray", "linestyle": "-", "linewidth": 2, "marker": "o", "label": src}

        # Plot latency
        ax_latency.plot(group["head_dim"], group[COL_MEAN_LATENCY], **source_style)

        # Plot throughput if available
        if COL_THROUGHPUT in group.columns and not group[COL_THROUGHPUT].isna().all():
            ax_throughput.plot(group["head_dim"], group[COL_THROUGHPUT], **source_style)

    # Add reference lines
    head_dims = plot_df["head_dim"].unique()
    if len(head_dims) > 1:
        min_dim = min(head_dims)
        max_dim = max(head_dims)

        # Linear reference line O(d)
        x_linear = np.array([min_dim, max_dim])
        scale_factor = min(plot_df[COL_MEAN_LATENCY]) / min_dim
        y_linear = x_linear * scale_factor

        ax_latency.plot(
            x_linear,
            y_linear,
            color=styles["REF_LINE_COLOR"],
            linestyle=styles["REF_LINE_STYLE"],
            linewidth=styles["REF_LINE_WIDTH"],
            alpha=styles["REF_LINE_ALPHA"],
            label="O(d)",
        )

        # Quadratic reference line O(d²)
        y_quadratic = x_linear**2 * (scale_factor / min_dim)

        ax_latency.plot(
            x_linear,
            y_quadratic,
            color=styles["REF_LINE_COLOR"],
            linestyle=":",
            linewidth=styles["REF_LINE_WIDTH"],
            alpha=styles["REF_LINE_ALPHA"],
            label="O(d²)",
        )

    # Set latency plot aesthetics
    plot_utils.apply_common_plot_aesthetics(
        ax_latency,
        "Latency vs Head Dimension",
        "Head Dimension",
        "Mean Latency (ms)",
        styles,
        x_scale="linear",
        y_scale="linear",
        include_legend=True,
    )

    # Set throughput plot aesthetics
    plot_utils.apply_common_plot_aesthetics(
        ax_throughput,
        "Throughput vs Head Dimension",
        "Head Dimension",
        "Throughput (items/sec)",
        styles,
        x_scale="linear",
        y_scale="linear",
        include_legend=False,  # Legend already in latency plot
    )

    # Adjust layout
    plt.tight_layout()

    # Save figure
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = "latency_vs_head_dim.png"
    fig.savefig(output_dir / filename, dpi=300)
    plt.close(fig)

    return filename
