"""Plot latency as a function of sequence length."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib.patheffects as pe
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

    Creates two side-by-side plots:
    1. Prefill Latency vs sequence length
    2. Decode Latency vs history length

    Args:
        df: DataFrame with benchmark results.
        output_dir: Output directory for the generated plots.
        styles: Plot style dictionary.

    Returns:
        Filename of the generated latency plot.
    """
    styles = styles or STYLES

    # Create wider figure with two subplots
    fig, (ax_prefill, ax_decode) = plt.subplots(1, 2, figsize=(15, 6))

    source_styles = {
        "pal": {
            "color": styles["PAL_COLOR"],
            "outline_color": styles["PAL_OUTLINE_COLOR"],
            "linestyle": styles["PAL_STYLE"],
            "linewidth": styles["PAL_LINEWIDTH"],
            "marker": styles["PAL_MARKER"],
            "label": styles["PAL_LABEL"],
        },
        "mlx": {
            "color": styles["MLX_COLOR"],
            "outline_color": styles["MLX_OUTLINE_COLOR"],
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

    # Split data into prefill and decode benchmarks
    prefill_df = df[~df["name"].str.contains("decode", case=False)]
    decode_df = df[df["name"].str.contains("decode", case=False)]

    # Process prefill data
    # ------------------------------------------------------------------------------
    # Plot latency for each source (prefill)
    for src, group_data in prefill_df.groupby("group"):
        # Sort by sequence length to ensure correct plotting of lines and filled areas
        sorted_group = group_data.sort_values(by="sequence_length")  # type: ignore[arg-type]
        if "pal" in str(src).lower():
            source_style = source_styles["pal"]
        elif "mlx" in str(src).lower():
            source_style = source_styles["mlx"]
        else:
            source_style = source_styles["default"]

        # Plot the mean latency line for prefill, with optional outline for visual panache
        (line,) = ax_prefill.plot(
            sorted_group["sequence_length"],
            sorted_group["mean_latency"],
            color=source_style["color"],
            linestyle=source_style["linestyle"],
            linewidth=source_style["linewidth"],
            marker=source_style["marker"],
            label=source_style["label"],
        )
        if "outline_color" in source_style:
            line.set_path_effects(
                [
                    pe.Stroke(
                        linewidth=source_style["linewidth"] * 3,
                        foreground=source_style["outline_color"],
                    ),
                    pe.Normal(),
                ]
            )

    # Add reference lines for prefill if we have enough data points
    x_range_prefill = prefill_df["sequence_length"].dropna()

    if not x_range_prefill.empty:
        x_max = x_range_prefill.max()
        x_vals = np.linspace(0, x_max, 200)
        # Find maximum latency value for scaling
        max_latency = prefill_df["mean_latency"].max()

        # Compute O(n^2) and O(n log n) reference lines, but scale so their max matches the data's max latency
        o_n2 = x_vals**2
        o_nlogn = x_vals * np.log(np.clip(x_vals, 1, None))  # avoid log(0)

        # Scale so that the max of each reference curve matches max_latency (but never exceeds it)
        o_n2_scaled = max_latency * (o_n2 / o_n2.max()) if np.any(o_n2 > 0) else o_n2
        o_nlogn_scaled = max_latency * (o_nlogn / o_nlogn.max()) if np.any(o_nlogn > 0) else o_nlogn

        # O(n^2) reference line for prefill
        ax_prefill.plot(
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
        ax_prefill.text(
            x_vals[n2_label_idx] * 1.05,
            o_n2_scaled[n2_label_idx],
            r"$O(n^2)$",
            color=styles["REF_LINE_COLOR"],
            fontsize=styles["REF_LINE_FONTSIZE"],
            zorder=1,
            ha="left",
        )

        # O(n log n) reference line for prefill
        ax_prefill.plot(
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
        ax_prefill.text(
            x_vals[nlogn_label_idx] * 1.05,
            o_nlogn_scaled[nlogn_label_idx],
            r"$O(n \log n)$",
            color=styles["REF_LINE_COLOR"],
            fontsize=styles["REF_LINE_FONTSIZE"],
            zorder=1,
            ha="left",
        )

    # Set prefill plot aesthetics
    plot_utils.apply_common_plot_aesthetics(
        ax_prefill,
        "Prefill Latency vs Sequence Length",
        "Sequence Length (tokens)",
        "Mean Latency (ms)",
        styles,
        x_scale="log",
        y_scale="log",
        include_legend=True,
    )

    # Plot latency for each source (decode)
    for src, group_data in decode_df.groupby("group"):
        # Rename group labels to indicate decode
        src_label = str(src)
        if "pal" in src_label.lower():
            src_label = "PAL"
            source_style = source_styles["pal"].copy()
            source_style["label"] = source_style["label"] + " (Decode)"
        elif "mlx" in src_label.lower():
            src_label = "MLX"
            source_style = source_styles["mlx"].copy()
            source_style["label"] = source_style["label"] + " (Decode)"
        else:
            source_style = source_styles["default"].copy()
            source_style["label"] = source_style["label"] + " (Decode)"

        # Sort by sequence length to ensure correct plotting of lines and filled areas
        sorted_group = group_data.sort_values(by="sequence_length")  # type: ignore[arg-type]

        # Plot the mean latency line for decode
        (line,) = ax_decode.plot(
            sorted_group["sequence_length"],
            sorted_group["mean_latency"],
            color=source_style["color"],
            linestyle=source_style["linestyle"],
            linewidth=source_style["linewidth"],
            marker=source_style["marker"],
            label=source_style["label"],
        )
        if "outline_color" in source_style:
            line.set_path_effects(
                [
                    pe.Stroke(
                        linewidth=source_style["linewidth"] * 3,
                        foreground=source_style["outline_color"],
                    ),  # the outline
                    pe.Normal(),  # the original line
                ]
            )

    # Add reference lines for decode if we have enough data points
    x_range_decode = decode_df["sequence_length"].dropna()

    if not x_range_decode.empty:
        x_max = x_range_decode.max()
        x_vals = np.linspace(0, x_max, 200)
        # Find maximum latency value for scaling
        max_latency = decode_df["mean_latency"].max()

        # Compute O(n) reference line for decode (since decode should be more efficient)
        o_n = x_vals
        # Scale so that the max of each reference curve matches max_latency (but never exceeds it)
        o_n_scaled = max_latency * (o_n / o_n.max()) if np.any(o_n > 0) else o_n

        # O(n) reference line for decode
        ax_decode.plot(
            x_vals,
            o_n_scaled,
            color=styles["REF_LINE_COLOR"],
            linestyle=styles["REF_LINE_STYLE"],
            linewidth=styles["REF_LINE_WIDTH"],
            alpha=float(styles["REF_LINE_ALPHA"]),
            label=r"$O(n)$",
            zorder=1,
        )
        n_label_idx = int(0.7 * len(x_vals))
        ax_decode.text(
            x_vals[n_label_idx] * 1.05,
            o_n_scaled[n_label_idx],
            r"$O(n)$",
            color=styles["REF_LINE_COLOR"],
            fontsize=styles["REF_LINE_FONTSIZE"],
            zorder=1,
            ha="left",
        )

    # Set decode plot aesthetics
    plot_utils.apply_common_plot_aesthetics(
        ax_decode,
        "Decode Latency vs History Length",
        "History Length (tokens)",
        "Mean Latency (ms)",
        styles,
        x_scale="log",
        y_scale="log",
        include_legend=True,
    )

    ax_decode.legend(loc="best")

    # Configure tick formatting for both plots
    ax_prefill.minorticks_on()
    ax_prefill.grid(linewidth=0.5, alpha=0.5)
    ax_decode.minorticks_on()
    ax_decode.grid(linewidth=0.5, alpha=0.5)

    # Add more space between subplots
    plt.subplots_adjust(wspace=0.25)

    # Construct filename
    filename = "latency_vs_seq_len.png"

    # Save figure with a flourish of directory creation
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / filename, dpi=300)
    plt.close(fig)

    # save the dataframe to results.json in the output directory

    # Group the data as {group: {sequence_length: mean_latency, ...}, ...}
    output_dict = {}
    for group, group_df in df.groupby("group"):
        # Use float for sequence_length keys to avoid accidental stringification
        seq_lat_map = {
            float(seq_len): float(m_lat)
            for seq_len, m_lat in zip(group_df["sequence_length"], group_df["mean_latency"], strict=False)
        }
        output_dict[group] = seq_lat_map

    with open(output_dir / "results.json", "w") as f:
        json.dump(output_dict, f, indent=4)

    return filename
