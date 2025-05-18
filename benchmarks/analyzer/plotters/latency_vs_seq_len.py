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
DISTINCT_REF_LINE_COLOR = "#024645"


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
    # Create figure with one subplot: latency
    fig, ax_latency = plt.subplots(1, 1, figsize=(8, 6))
    # Style mapping for different sources
    source_styles = {
        "python_pal": {
            "color": styles["PAL_PY_COLOR"],
            "linestyle": "-",  # Solid line for clarity
            "linewidth": styles["PAL_LINEWIDTH"],
            "marker": styles["PAL_PY_MARKER"],
            "label": styles["PAL_PY_LABEL"],
        },
        "python_sdpa": {
            "color": styles["SDPA_PY_COLOR"],
            "linestyle": "-",  # Solid line for clarity
            "linewidth": styles["SDPA_LINEWIDTH"],
            "marker": styles["SDPA_PY_MARKER"],
            "label": styles["SDPA_PY_LABEL"],
        },
    }
    df["sequence_length"] = df["param"].astype(float)
    df["mean_latency"] = df["stats"].apply(lambda x: x["mean"] * 1000)  # convert to milliseconds

    # Plot latency for each source
    for src, group_data in df.groupby("group"):
        # Sort by sequence length to ensure correct plotting of lines and filled areas
        sorted_group = group_data.sort_values(by="sequence_length")

        # Get style for this source
        source_style = source_styles.get(str(src), {})
        if not source_style:
            # Default style if source not in mapping
            logger.warning(f"No style defined for source: {src}. Using default style.")
            source_style = {"color": "gray", "linestyle": "-", "linewidth": 2, "marker": "o", "label": str(src)}

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
        x_vals = np.linspace(0, x_max)
        # Find maximum latency value for scaling
        max_latency = df["mean_latency"].max()
        y_scale = max_latency / x_max if x_max > 0 else 1

        # O(n) reference line
        ax_latency.plot(
            x_vals,
            y_scale * x_vals,
            color=DISTINCT_REF_LINE_COLOR,
            linestyle="--",
            linewidth=styles["REF_LINE_WIDTH"],
            alpha=float(styles["REF_LINE_ALPHA"]),
            label="O(n)",
            zorder=1,
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
