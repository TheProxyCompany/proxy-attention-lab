"""Plot latency as a function of sequence length."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import LogFormatter, LogLocator

from benchmarks.analyzer import plot_utils
from benchmarks.analyzer.config import (
    COL_BENCHMARK_NAME_BASE,
    COL_KERNEL_NAME,
    COL_MEAN_LATENCY,
    COL_SOURCE,
    COL_THROUGHPUT,
)

# Set up logging
logger = logging.getLogger(__name__)

STYLES = plot_utils.get_plot_styles()
# Define a distinct color for reference lines to avoid clash with grid/data
# Using a light, desaturated blue/purple for subtlety but distinctiveness
DISTINCT_REF_LINE_COLOR = "#B0C4DE"  # LightSteelBlue, can be adjusted


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
            "linestyle": "-",  # Solid line for clarity
            "linewidth": styles["PAL_LINEWIDTH"],
            "marker": styles["PAL_CPP_MARKER"],
            "label": styles["PAL_CPP_LABEL"],
            "zorder": 3,  # Ensure data lines are on top
        },
        "python_pal": {
            "color": styles["PAL_PY_COLOR"],
            "linestyle": "-",  # Solid line for clarity
            "linewidth": styles["PAL_LINEWIDTH"],
            "marker": styles["PAL_PY_MARKER"],
            "label": styles["PAL_PY_LABEL"],
            "zorder": 3,
        },
        "cpp_sdpa": {
            "color": styles["SDPA_CPP_COLOR"],
            "linestyle": "-",  # Solid line for clarity
            "linewidth": styles["SDPA_LINEWIDTH"],
            "marker": styles["SDPA_CPP_MARKER"],
            "label": styles["SDPA_CPP_LABEL"],
            "zorder": 3,
        },
        "python_sdpa": {
            "color": styles["SDPA_PY_COLOR"],
            "linestyle": "-",  # Solid line for clarity
            "linewidth": styles["SDPA_LINEWIDTH"],
            "marker": styles["SDPA_PY_MARKER"],
            "label": styles["SDPA_PY_LABEL"],
            "zorder": 3,
        },
    }

    # Track which sources we actually plotted
    plotted_sources = []

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
            plotted_sources.append(src)
            logger.info(f"Plotted latency data for source: {src}")
        else:
            logger.warning(f"Missing or insufficient latency data for source {src} in latency_vs_seq_len plot")

        # Plot throughput if available
        if (
            COL_THROUGHPUT in group.columns
            and not group[COL_THROUGHPUT].isna().all()
            and len(group["seq_len"].unique()) > 1
        ):
            ax_throughput.plot(group["seq_len"], group[COL_THROUGHPUT], **source_style)
            logger.info(f"Plotted throughput data for source: {src}")
        else:
            logger.debug(f"Missing throughput data for source {src} in latency_vs_seq_len plot")

    # Log summary of what was plotted
    logger.info(f"Successfully plotted data for {len(plotted_sources)} sources: {', '.join(plotted_sources)}")

    # Store current Y limits of ax_latency based on DATA ONLY, before reference lines
    data_y_min_latency, data_y_max_latency = None, None
    if ax_latency.get_lines():  # Check if any data was actually plotted
        data_y_min_latency, data_y_max_latency = ax_latency.get_ylim()

    # Add reference lines if we have enough data points
    valid_data = plot_df[plot_df[COL_MEAN_LATENCY].notna()]
    seq_lens = valid_data["seq_len"].unique()

    if len(seq_lens) > 1 and not valid_data.empty:
        try:
            min_seq = min(s for s in seq_lens if s > 0)  # Ensure min_seq is positive for logspace
            max_seq = max(seq_lens)

            # Skip reference lines if min/max are too close or invalid
            if min_seq <= 0 or max_seq <= 0 or max_seq <= min_seq:  # also check max_seq > 0
                logger.warning(f"Cannot add reference lines: invalid sequence length range [{min_seq}, {max_seq}]")
            else:
                # Find a representative point from actual data for scaling reference lines
                # Use the minimum latency point as reference for scaling
                ref_data_for_scaling = valid_data.loc[valid_data[COL_MEAN_LATENCY].idxmin()]

                if (
                    pd.notna(ref_data_for_scaling["seq_len"])
                    and ref_data_for_scaling["seq_len"] > 0
                    and pd.notna(ref_data_for_scaling[COL_MEAN_LATENCY])
                    and ref_data_for_scaling[COL_MEAN_LATENCY] > 0
                ):
                    ref_x_val = ref_data_for_scaling["seq_len"]
                    ref_y_val = ref_data_for_scaling[COL_MEAN_LATENCY]

                    logger.info(f"Using reference point for scaling: seq_len={ref_x_val}, latency={ref_y_val}")

                    # Create more x values for smoother reference lines
                    x_vals = np.logspace(np.log10(min_seq), np.log10(max_seq), 100)

                    # O(n log n) reference line
                    if (
                        ref_x_val > 1 and np.log2(ref_x_val) > 0
                    ):  # log2(1) is 0, causing division by zero if ref_x_val is 1
                        scale_nlogn = ref_y_val / (ref_x_val * np.log2(ref_x_val))
                        # Ensure x_vals for log are > 1 for log2
                        valid_x_for_log = x_vals[x_vals > 1]
                        if len(valid_x_for_log) > 0:
                            y_nlogn = scale_nlogn * valid_x_for_log * np.log2(valid_x_for_log)
                            ax_latency.plot(
                                valid_x_for_log,
                                y_nlogn,
                                color=DISTINCT_REF_LINE_COLOR,
                                linestyle="-.",
                                linewidth=styles["REF_LINE_WIDTH"],
                                alpha=float(styles["REF_LINE_ALPHA"]) * 0.5,
                                label="O(n log n)",
                                zorder=1,  # Draw behind data
                            )
                            logger.debug(f"Added O(n log n) reference line, scale_factor={scale_nlogn}")
                    else:
                        logger.warning(
                            f"Cannot add O(n log n) reference line: invalid reference x value {ref_x_val} or it's <= 1"
                        )

                    # O(n) reference line
                    if ref_x_val > 0:  # Avoid division by zero for ref_x_val
                        scale_n = ref_y_val / ref_x_val
                        y_n = scale_n * x_vals
                        ax_latency.plot(
                            x_vals,
                            y_n,
                            color=DISTINCT_REF_LINE_COLOR,
                            linestyle="-.",
                            linewidth=styles["REF_LINE_WIDTH"],
                            alpha=float(styles["REF_LINE_ALPHA"]) * 0.5,
                            label="O(n)",
                            zorder=1,
                        )
                        logger.debug(f"Added O(n) reference line, scale_factor={scale_n}")
                    else:
                        logger.warning(f"Cannot add O(n) reference line: invalid reference x value {ref_x_val}")
                else:
                    logger.warning(
                        "Could not determine valid reference point for scaling reference lines in Latency vs SeqLen."
                    )
        except Exception as e:
            logger.error(f"Error adding reference lines: {e}", exc_info=True)

    # Set title based on whether we're filtering by kernel
    title_prefix = ""
    if kernel_filter:
        # Format the kernel name for display
        kernel_display = kernel_filter.replace("_", " ").title()
        title_prefix = f"{kernel_display}: "

    # Set latency plot aesthetics
    plot_utils.apply_common_plot_aesthetics(
        ax_latency,
        f"{title_prefix}Latency (milliseconds) vs Sequence Length (tokens)",
        "Sequence Length (tokens)",
        "Mean Latency (ms)",
        styles,
        x_scale="log",
        y_scale="log",
        include_legend=True,
    )

    # Add reference lines for throughput if we have enough data points
    valid_data_throughput = plot_df[
        plot_df[COL_THROUGHPUT].notna() & plot_df["seq_len"].notna() & (plot_df[COL_THROUGHPUT] > 0)
    ]
    seq_lens_throughput = valid_data_throughput["seq_len"].unique()

    if len(seq_lens_throughput) > 1 and not valid_data_throughput.empty:
        try:
            min_seq_t = min(s for s in seq_lens_throughput if s > 0)
            max_seq_t = max(seq_lens_throughput)

            if min_seq_t <= 0 or max_seq_t <= 0 or max_seq_t <= min_seq_t:
                logger.warning(
                    f"Cannot add throughput reference lines: invalid sequence length range [{min_seq_t}, {max_seq_t}]"
                )
            else:
                # Use the point with maximum throughput as reference for scaling
                ref_data_t_scaling = valid_data_throughput.loc[valid_data_throughput[COL_THROUGHPUT].idxmax()]

                if (
                    pd.notna(ref_data_t_scaling["seq_len"])
                    and ref_data_t_scaling["seq_len"] > 0
                    and pd.notna(ref_data_t_scaling[COL_THROUGHPUT])
                    and ref_data_t_scaling[COL_THROUGHPUT] > 0
                ):
                    ref_s_val_t = ref_data_t_scaling["seq_len"]
                    ref_t_val = ref_data_t_scaling[COL_THROUGHPUT]

                    logger.info(
                        f"Using throughput reference point for scaling: seq_len={ref_s_val_t}, throughput={ref_t_val}"
                    )

                    x_vals_t = np.linspace(min_seq_t, max_seq_t, 100)
                    # Ensure x_vals_t are positive for calculations
                    x_vals_t = x_vals_t[x_vals_t > 0]
                    if len(x_vals_t) == 0:
                        raise ValueError("No valid x_vals_t > 0 for throughput reference lines.")

                    # O(1/n) reference line for throughput
                    if ref_s_val_t > 0:
                        scale_inv_n = ref_t_val * ref_s_val_t
                        y_inv_n = scale_inv_n / x_vals_t
                        ax_throughput.plot(
                            x_vals_t,
                            y_inv_n,
                            color=DISTINCT_REF_LINE_COLOR,
                            linestyle="-.",
                            linewidth=styles["REF_LINE_WIDTH"],
                            alpha=float(styles["REF_LINE_ALPHA"]) * 0.5,
                            label="O(1/n)",
                            zorder=1,
                        )
                        logger.debug(f"Added O(1/n) throughput reference line, scale_factor={scale_inv_n}")

                    # O(1/(n log n)) reference line for throughput
                    valid_x_for_log_t = x_vals_t[x_vals_t > 1]
                    if ref_s_val_t > 1 and np.log2(ref_s_val_t) > 0 and len(valid_x_for_log_t) > 0:
                        scale_inv_nlogn = ref_t_val * ref_s_val_t * np.log2(ref_s_val_t)
                        y_inv_nlogn = scale_inv_nlogn / (valid_x_for_log_t * np.log2(valid_x_for_log_t))
                        ax_throughput.plot(
                            valid_x_for_log_t,
                            y_inv_nlogn,
                            color=DISTINCT_REF_LINE_COLOR,
                            linestyle="-.",
                            linewidth=styles["REF_LINE_WIDTH"],
                            alpha=float(styles["REF_LINE_ALPHA"]) * 0.5,
                            label="O(1/(n log n))",
                            zorder=1,
                        )
                        logger.debug(f"Added O(1/(n log n)) throughput reference line, scale_factor={scale_inv_nlogn}")
                else:
                    logger.warning("Could not determine valid reference point for scaling throughput reference lines.")
        except Exception as e:
            logger.error(f"Error adding throughput reference lines: {e}", exc_info=True)

    # Set throughput plot aesthetics
    plot_utils.apply_common_plot_aesthetics(
        ax_throughput,
        f"{title_prefix}Throughput vs Sequence Length",
        "Sequence Length (tokens)",
        "Throughput (Query Vectors/sec)",
        styles,
        x_scale="log",
        y_scale="log",
        include_legend=True,
    )

    # Construct filename based on kernel filter
    filename = f"{kernel_filter}_latency_vs_seq_len.png" if kernel_filter else "latency_vs_seq_len.png"

    # Configure tick formatting before saving
    ax_latency.yaxis.set_major_locator(LogLocator(base=10))
    ax_latency.yaxis.set_major_formatter(LogFormatter(labelOnlyBase=False))
    ax_latency.minorticks_on()
    ax_latency.grid(which="minor", linestyle=":", linewidth=0.5, alpha=0.5)

    # Save figure with a flourish of directory creation
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / filename, dpi=300)
    plt.close(fig)

    return filename
