"""Plot latency as a function of sequence length."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from benchmarks.analyzer import plot_utils

# Set up logging
logger = logging.getLogger(__name__)

STYLES = plot_utils.get_plot_styles()


def plot(df: pd.DataFrame, output_dir: Path, styles: dict[str, str | float] | None = None) -> str:
    """
    Generate latency vs sequence length plots.

    Creates one plot:
    1. Latency vs sequence length

    Args:
        df: DataFrame with benchmark results.
        output_dir: Output directory for the generated plots.
        styles: Plot style dictionary.
        kernel_filter: Optional kernel name to filter by (e.g., "paged_attention", "sdpa").

    Returns:
        Filename of the generated latency plot.
    """
    styles = styles or STYLES
    fig, ax_latency = plt.subplots(1, 1, figsize=(8, 6))
    source_styles = {
        "pal": {
            "color": styles["PAL_COLOR"],
            "linestyle": styles["PAL_STYLE"],
            "linewidth": styles["PAL_LINEWIDTH"],
            "marker": styles["PAL_MARKER"],
            "label": styles["PAL_LABEL"],
        },
        "mlx": {
            "color": styles["MLX_COLOR"],
            "linestyle": styles["MLX_STYLE"],
            "linewidth": styles["MLX_LINEWIDTH"],
            "marker": styles["MLX_MARKER"],
            "label": styles["MLX_LABEL"],
        },
        "default": {
            "color": "gray",
            "linestyle": "-",
            "linewidth": 2,
            "marker": "o",
            "label": "default",
        },
    }

    # Plot latency for each source
    for src, group_data in df.groupby("group"):
        # Sort by sequence length to ensure correct plotting of lines and filled areas
        sorted_group = group_data.sort_values(by="sequence_length")
        if "pal" in str(src).lower():
            source_style = source_styles["pal"]
        elif "mlx" in str(src).lower():
            source_style = source_styles["mlx"]
        else:
            source_style = source_styles["default"]

        # Plot the mean latency line
        ax_latency.plot(
            sorted_group["sequence_length"],
            sorted_group["mean_latency"],
            color=source_style["color"],
            linestyle=source_style["linestyle"],
            linewidth=source_style["linewidth"],
            marker=source_style["marker"],
            label=source_style["label"],
        )

    # Add reference lines if we have enough data points
    x_range = df["sequence_length"].dropna()

    if not x_range.empty:
        x_max = x_range.max()
        x_vals = np.linspace(0, x_max, 200)
        # Find maximum latency value for scaling
        max_latency = df["mean_latency"].max()

        # Compute O(n^2) and O(n log n) reference lines, but scale so their max matches the data's max latency
        o_n2 = x_vals**2
        o_nlogn = x_vals * np.log(np.clip(x_vals, 1, None))  # avoid log(0)

        # Scale so that the max of each reference curve matches max_latency (but never exceeds it)
        o_n2_scaled = max_latency * (o_n2 / o_n2.max()) if np.any(o_n2 > 0) else o_n2
        o_nlogn_scaled = max_latency * (o_nlogn / o_nlogn.max()) if np.any(o_nlogn > 0) else o_nlogn

        # O(n^2) reference line
        ax_latency.plot(
            x_vals,
            o_n2_scaled,
            color=styles["REF_LINE_COLOR"],
            linestyle=styles["REF_LINE_STYLE"],
            linewidth=styles["REF_LINE_WIDTH"],
            alpha=float(styles["REF_LINE_ALPHA"]),
            label=r"$O(n^2)$",
            zorder=1,
        )
        n2_label_idx = int(0.7 * len(x_vals))
        ax_latency.text(
            x_vals[n2_label_idx] * 1.05,
            o_n2_scaled[n2_label_idx],
            r"$O(n^2)$",
            color=styles["REF_LINE_COLOR"],
            fontsize=styles["REF_LINE_FONTSIZE"],
            zorder=1,
            ha="left",
        )

        # O(n log n) reference line
        ax_latency.plot(
            x_vals,
            o_nlogn_scaled,
            color=styles["REF_LINE_COLOR"],
            linestyle=styles["REF_LINE_STYLE"],
            linewidth=styles["REF_LINE_WIDTH"],
            alpha=float(styles["REF_LINE_ALPHA"]),
            label=r"$O(n \log n)$",
            zorder=1,
        )
        nlogn_label_idx = int(0.45 * len(x_vals))
        ax_latency.text(
            x_vals[nlogn_label_idx] * 1.05,
            o_nlogn_scaled[nlogn_label_idx],
            r"$O(n \log n)$",
            color=styles["REF_LINE_COLOR"],
            fontsize=styles["REF_LINE_FONTSIZE"],
            zorder=1,
            ha="left",
        )

    # Set latency plot aesthetics
    plot_utils.apply_common_plot_aesthetics(
        ax_latency,
        "Latency (milliseconds) vs Sequence Length (tokens)",
        "Sequence Length (tokens)",
        "Mean Latency (ms)",
        styles,
        x_scale="linear",
        y_scale="linear",
        include_legend=True,
    )

    # Construct filename based on kernel filter
    filename = "latency_vs_seq_len.png"

    # Configure tick formatting before saving
    ax_latency.minorticks_on()
    ax_latency.grid(linewidth=0.5, alpha=0.5)

    # Save figure with a flourish of directory creation
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / filename, dpi=300)
    plt.close(fig)

    # save the dataframe to results.json in the output directory

    # Group the data as {group: {sequence_length: mean_latency, ...}, ...}
    output_dict = {}
    for group, group_df in df.groupby("group"):
        # Use float for sequence_length keys to avoid accidental stringification
        seq_lat_map = {float(row["sequence_length"]): float(row["mean_latency"]) for _, row in group_df.iterrows()}
        output_dict[group] = seq_lat_map

    with open(output_dir / "results.json", "w") as f:
        json.dump(output_dict, f, indent=4)

    return filename
