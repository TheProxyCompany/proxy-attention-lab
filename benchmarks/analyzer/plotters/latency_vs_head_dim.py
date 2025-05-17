"""Plot latency as a function of head dimension."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from benchmarks.analyzer import plot_utils
from benchmarks.analyzer.config import (
    COL_BENCHMARK_NAME_BASE,
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
    Generate latency vs head dimension plots.

    Creates two plots:
    1. Latency vs head dimension
    2. Throughput vs head dimension (if throughput data available)

    Args:
        df: DataFrame with benchmark results.
        output_dir: Output directory for the generated plots.
        styles: Plot style dictionary.
        kernel_filter: Optional kernel name to filter by (e.g., "paged_attention", "sdpa").

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
    if COL_BENCHMARK_NAME_BASE in plot_df.columns and set(benchmark_names) & set(
        plot_df[COL_BENCHMARK_NAME_BASE].unique()
    ):
        plot_df = plot_df[plot_df[COL_BENCHMARK_NAME_BASE].isin(benchmark_names)]

    # Filter by kernel if specified
    if kernel_filter and COL_KERNEL_NAME in plot_df.columns:
        filtered_df = plot_df[plot_df[COL_KERNEL_NAME] == kernel_filter]
        if filtered_df.empty:
            logger.warning(f"No data found for kernel '{kernel_filter}' in latency_vs_head_dim plot.")
            return ""
        plot_df = filtered_df

    if plot_df.empty:
        logger.warning("No data available for latency_vs_head_dim plot.")
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
        if len(group) < 2:
            logger.warning(f"Not enough data points for source {src} in latency_vs_head_dim plot (found {len(group)})")
            continue

        # Sort by head dimension
        group = group.sort_values("head_dim")

        # Get style for this source
        source_style = source_styles.get(str(src), {})
        if not source_style:
            # Default style if source not in mapping
            source_style = {"color": "gray", "linestyle": "-", "linewidth": 2, "marker": "o", "label": src}

        # Plot latency (check for valid data)
        if not group[COL_MEAN_LATENCY].isna().all() and len(group["head_dim"].unique()) > 1:
            ax_latency.plot(group["head_dim"], group[COL_MEAN_LATENCY], **source_style)
        else:
            logger.warning(f"Missing latency data for source {src} in latency_vs_head_dim plot")

        # Plot throughput if available
        if (
            COL_THROUGHPUT in group.columns
            and not group[COL_THROUGHPUT].isna().all()
            and len(group["head_dim"].unique()) > 1
        ):
            ax_throughput.plot(group["head_dim"], group[COL_THROUGHPUT], **source_style)
        else:
            logger.debug(f"Missing throughput data for source {src} in latency_vs_head_dim plot")

    # Add reference lines if we have enough data points
    valid_data = plot_df[plot_df[COL_MEAN_LATENCY].notna()]
    head_dims = valid_data["head_dim"].unique()

    if len(head_dims) > 1 and not valid_data.empty:
        min_dim = min(head_dims)
        max_dim = max(head_dims)

        # Find a suitable reference point for scaling
        # Use the median point for more stability
        ref_idx = valid_data["head_dim"].searchsorted(min_dim)
        if ref_idx < len(valid_data):
            ref_point = valid_data.iloc[ref_idx]
            ref_latency = ref_point[COL_MEAN_LATENCY]
            ref_head_dim = ref_point["head_dim"]

            if ref_latency > 0 and ref_head_dim > 0:
                # Linear reference line O(d)
                x_linear = np.array([min_dim, max_dim])
                scale_factor = ref_latency / ref_head_dim
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
                y_quadratic = x_linear**2 * (scale_factor / ref_head_dim)

                ax_latency.plot(
                    x_linear,
                    y_quadratic,
                    color=styles["REF_LINE_COLOR"],
                    linestyle=":",
                    linewidth=styles["REF_LINE_WIDTH"],
                    alpha=styles["REF_LINE_ALPHA"],
                    label="O(d²)",
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
        f"{title_prefix}Latency vs Head Dimension",
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
        f"{title_prefix}Throughput vs Head Dimension",
        "Head Dimension",
        "Throughput (items/sec)",
        styles,
        x_scale="linear",
        y_scale="linear",
        include_legend=False,  # Legend already in latency plot
    )

    # Adjust layout
    plt.tight_layout()

    # Construct filename based on kernel filter
    filename = f"{kernel_filter}_latency_vs_head_dim.png" if kernel_filter else "latency_vs_head_dim.png"

    # Save figure
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / filename, dpi=300)
    plt.close(fig)

    return filename
