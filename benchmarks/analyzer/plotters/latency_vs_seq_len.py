"""Plot latency as a function of sequence length."""

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
    Generate latency vs sequence length plots.

    Creates two plots:
    1. Latency vs sequence length
    2. Throughput vs sequence length (if throughput data available)

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
        "BM_PAL_LatencyVsSeqLen",
        "test_pal_latency_vs_seq_len",
        "BM_SDPA_LatencyVsSeqLen",
        "test_sdpa_latency_vs_seq_len",
    ]
    plot_df = df[df[COL_SOURCE].notna() & df["seq_len"].notna() & df[COL_MEAN_LATENCY].notna()]

    # Further filter if we have benchmark_name_base information
    if COL_BENCHMARK_NAME_BASE in plot_df.columns and set(benchmark_names).intersection(
        set(plot_df[COL_BENCHMARK_NAME_BASE].unique())
    ):
        plot_df = plot_df[plot_df[COL_BENCHMARK_NAME_BASE].isin(benchmark_names)]

    # Filter by kernel if specified
    if kernel_filter and COL_KERNEL_NAME in plot_df.columns:
        filtered_df = plot_df[plot_df[COL_KERNEL_NAME] == kernel_filter]
        if filtered_df.empty:
            logger.warning(f"No data found for kernel '{kernel_filter}' in latency_vs_seq_len plot.")
            return ""
        plot_df = filtered_df

    if plot_df.empty:
        logger.warning("No data available for latency_vs_seq_len plot.")
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
            logger.warning(f"Not enough data points for source {src} in latency_vs_seq_len plot (found {len(group)})")
            continue

        # Sort by sequence length
        group = group.sort_values("seq_len")

        # Get style for this source
        source_style = source_styles.get(str(src), {})
        if not source_style:
            # Default style if source not in mapping
            source_style = {"color": "gray", "linestyle": "-", "linewidth": 2, "marker": "o", "label": src}

        # Plot latency (check for valid data)
        if not group[COL_MEAN_LATENCY].isna().all() and len(group["seq_len"].unique()) > 1:
            ax_latency.plot(group["seq_len"], group[COL_MEAN_LATENCY], **source_style)
        else:
            logger.warning(f"Missing latency data for source {src} in latency_vs_seq_len plot")

        # Plot throughput if available
        if (
            COL_THROUGHPUT in group.columns
            and not group[COL_THROUGHPUT].isna().all()
            and len(group["seq_len"].unique()) > 1
        ):
            ax_throughput.plot(group["seq_len"], group[COL_THROUGHPUT], **source_style)
        else:
            logger.debug(f"Missing throughput data for source {src} in latency_vs_seq_len plot")

    # Add reference lines if we have enough data points
    valid_data = plot_df[plot_df[COL_MEAN_LATENCY].notna()]
    seq_lens = valid_data["seq_len"].unique()

    if len(seq_lens) > 1 and not valid_data.empty:
        min_seq = min(seq_lens)
        max_seq = max(seq_lens)

        # Find a suitable reference point for scaling
        # Use the median point for more stability
        ref_idx = valid_data["seq_len"].searchsorted(min_seq)
        if ref_idx < len(valid_data):
            ref_point = valid_data.iloc[ref_idx]
            ref_latency = ref_point[COL_MEAN_LATENCY]
            ref_seq_len = ref_point["seq_len"]

            if ref_latency > 0 and ref_seq_len > 0:
                # Linear reference line O(n)
                x_linear = np.array([min_seq, max_seq])
                scale_factor = ref_latency / ref_seq_len
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

                # Quadratic reference line O(n²)
                y_quadratic = x_linear**2 * (scale_factor / ref_seq_len)

                ax_latency.plot(
                    x_linear,
                    y_quadratic,
                    color=styles["REF_LINE_COLOR"],
                    linestyle=":",
                    linewidth=styles["REF_LINE_WIDTH"],
                    alpha=styles["REF_LINE_ALPHA"],
                    label="O(n²)",
                )

                # Log-linear reference line O(n log n)
                y_nlogn = x_linear * np.log2(x_linear) * (scale_factor / (ref_seq_len * np.log2(ref_seq_len)))

                ax_latency.plot(
                    x_linear,
                    y_nlogn,
                    color=styles["REF_LINE_COLOR"],
                    linestyle="-.",
                    linewidth=styles["REF_LINE_WIDTH"],
                    alpha=styles["REF_LINE_ALPHA"],
                    label="O(n log n)",
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
        f"{title_prefix}Latency vs Sequence Length",
        "Sequence Length",
        "Mean Latency (ms)",
        styles,
        x_scale="log",
        y_scale="log",
        include_legend=True,
    )

    # Set throughput plot aesthetics
    plot_utils.apply_common_plot_aesthetics(
        ax_throughput,
        f"{title_prefix}Throughput vs Sequence Length",
        "Sequence Length",
        "Throughput (items/sec)",
        styles,
        x_scale="log",
        y_scale="log",
        include_legend=False,  # Legend already in latency plot
    )

    # Adjust layout
    plt.tight_layout()

    # Construct filename based on kernel filter
    filename = f"{kernel_filter}_latency_vs_seq_len.png" if kernel_filter else "latency_vs_seq_len.png"

    # Save figure
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / filename, dpi=300)
    plt.close(fig)

    return filename
