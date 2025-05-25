"""Plot latency as a function of sequence length."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from benchmarks.analyzer import plot_utils

logger = logging.getLogger(__name__)

STYLES = plot_utils.get_plot_styles()

# Plot configuration constants
ERROR_BAR_CAPSIZE = 5
ERROR_BAR_CAPTHICK = 1
ERROR_BAR_ALPHA = 0.9
MLX_ERROR_COLOR = "#808080"  # Gray for visibility against dark background
REF_LINE_LABEL_OFFSET = 1.05
N2_LABEL_POSITION = 0.7
NLOGN_LABEL_POSITION = 0.45
N_LABEL_POSITION = 0.7


def calculate_log_log_slope(x_data: np.ndarray, y_data: np.ndarray) -> tuple[float, float] | None:
    """
    Calculate the slope and intercept in log-log space using linear regression.

    Returns:
        tuple of (slope, intercept) or None if calculation fails
    """
    try:
        # Remove any invalid values
        mask = (x_data > 0) & (y_data > 0) & np.isfinite(x_data) & np.isfinite(y_data)
        x_clean = x_data[mask]
        y_clean = y_data[mask]

        if len(x_clean) < 2:
            return None

        # Calculate slope in log-log space
        log_x = np.log(x_clean)
        log_y = np.log(y_clean)

        # Simple linear regression: y = mx + b
        A = np.vstack([log_x, np.ones(len(log_x))]).T
        slope, intercept = np.linalg.lstsq(A, log_y, rcond=None)[0]

        return slope, intercept
    except Exception as e:
        logger.warning(f"Failed to calculate slope: {e}")
        return None


def add_complexity_guide(ax, plot_type: str = "prefill") -> None:
    """Add an interpretation guide for log-log plots below the title."""
    if plot_type == "prefill":
        text = "Log-log: Slope ≈ 1 (linear), ≈ 2 (quadratic); lower = better scaling."
    else:  # decode
        text = "Log-log: Slope ≈ 1 (linear); lower = better scaling."

    # Get current title and reposition it higher with padding
    current_title = ax.get_title()
    ax.set_title(current_title, fontsize=14, fontweight="bold", pad=30, y=1.03)

    # Add guide text below the title
    ax.text(
        0.5,
        1.04,
        text,
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment="bottom",
        horizontalalignment="center",
        color="gray",
        style="italic",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="lightgray", alpha=0.8, linewidth=1.0),
    )


def plot(df: pd.DataFrame, output_dir: Path, styles: dict[str, str | float] | None = None) -> str:
    """
    Generate latency vs sequence length plots.

    Creates either:
    - Two side-by-side plots if both prefill and decode data exist
    - Single plot if only one type of data exists

    Args:
        df: DataFrame with benchmark results.
        output_dir: Output directory for the generated plots.
        styles: Plot style dictionary.

    Returns:
        Filename of the generated latency plot.
    """
    styles = styles or STYLES

    source_styles = {
        "pal": {
            "short_name": "PAL",
            "name": "Proxy, Paged Attention",
            "color": styles["PAL_COLOR"],
            "outline_color": styles["PAL_OUTLINE_COLOR"],
            "linestyle": styles["PAL_STYLE"],
            "linewidth": styles["PAL_LINEWIDTH"],
            "marker": styles["PAL_MARKER"],
            "label": styles["PAL_LABEL"],
        },
        "mlx": {
            "short_name": "MLX",
            "name": "MLX, Scaled Dot Product Attention",
            "color": styles["MLX_COLOR"],
            "outline_color": styles["MLX_OUTLINE_COLOR"],
            "linestyle": styles["MLX_STYLE"],
            "linewidth": styles["MLX_LINEWIDTH"],
            "marker": styles["MLX_MARKER"],
            "label": styles["MLX_LABEL"],
        },
        "default": {
            "short_name": "Default",
            "name": "Default",
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

    # Determine plot layout based on available data
    has_prefill = not prefill_df.empty
    has_decode = not decode_df.empty

    if has_prefill and has_decode:
        fig, (ax_prefill, ax_decode) = plt.subplots(1, 2, figsize=(15, 6))
    elif has_prefill:
        fig, ax_prefill = plt.subplots(1, 1, figsize=(8, 6))
        ax_decode = None
    elif has_decode:
        fig, ax_decode = plt.subplots(1, 1, figsize=(8, 6))
        ax_prefill = None
    else:
        raise ValueError("No data available for plotting")

    # Process prefill data
    if has_prefill and ax_prefill is not None:
        for src, group_data in prefill_df.groupby("group"):
            sorted_group = group_data.sort_values(by="sequence_length")  # type: ignore[arg-type]

            if "pal" in str(src).lower():
                source_style = source_styles["pal"]
            elif "mlx" in str(src).lower():
                source_style = source_styles["mlx"]
            else:
                source_style = source_styles["default"]

            # Plot with or without error bars
            if "std_latency" in sorted_group.columns and sorted_group["std_latency"].notna().any():
                error_color = MLX_ERROR_COLOR if "mlx" in str(src).lower() else source_style["color"]
                errorbar_container = ax_prefill.errorbar(
                    sorted_group["sequence_length"],
                    sorted_group["mean_latency"],
                    yerr=sorted_group["std_latency"],
                    color=source_style["color"],
                    ecolor=error_color,
                    linestyle=source_style["linestyle"],
                    linewidth=source_style["linewidth"],
                    marker=source_style["marker"],
                    label=source_style["label"],
                    capsize=ERROR_BAR_CAPSIZE,
                    capthick=ERROR_BAR_CAPTHICK,
                    alpha=ERROR_BAR_ALPHA,
                )
                line = errorbar_container[0]
            else:
                (line,) = ax_prefill.plot(
                    sorted_group["sequence_length"],
                    sorted_group["mean_latency"],
                    color=source_style["color"],
                    linestyle=source_style["linestyle"],
                    linewidth=source_style["linewidth"],
                    marker=source_style["marker"],
                    label=source_style["label"],
                )

            if "outline_color" in source_style and isinstance(line, Line2D):
                line.set_path_effects(
                    [
                        pe.Stroke(
                            linewidth=source_style["linewidth"] * 3,
                            foreground=source_style["outline_color"],
                        ),
                        pe.Normal(),
                    ]
                )

            x_data = sorted_group["sequence_length"].values
            y_data = sorted_group["mean_latency"].values
            assert isinstance(x_data, np.ndarray)
            assert isinstance(y_data, np.ndarray)

            # Calculate and display slope
            result = calculate_log_log_slope(x_data, y_data)
            if result is not None:
                slope, intercept = result
                label = source_style["label"].replace(r"$\mathbf{", "").replace(r"}$", "").replace("\\", "")
                logger.info(f"Prefill {label}: slope = {slope:.2f} (O(n^{slope:.1f}))")

                ax_prefill.text(
                    x_data[-1],
                    y_data[-1],
                    f"Slope ≈ {slope:.1f}",
                    color="red",
                    fontsize=12,
                    fontweight="bold",
                    horizontalalignment="right",
                    verticalalignment="bottom",
                )

        # Set prefill plot aesthetics
        title = "Prefill Latency vs Sequence Length"
        plot_utils.apply_common_plot_aesthetics(
            ax_prefill,
            title,
            "Sequence Length (tokens)",
            "Mean Latency (ms)",
            styles,
            x_scale="log",
            y_scale="log",
            include_legend=True,
        )
        add_complexity_guide(ax_prefill, "prefill")

    # Process decode data
    if has_decode and ax_decode is not None:
        for src, group_data in decode_df.groupby("group"):
            src_label = str(src)
            if "pal" in src_label.lower():
                src_label = "PAL"
                source_style = source_styles["pal"].copy()
            elif "mlx" in src_label.lower():
                src_label = "MLX"
                source_style = source_styles["mlx"].copy()
            else:
                source_style = source_styles["default"].copy()

            sorted_group = group_data.sort_values(by="sequence_length")  # type: ignore[arg-type]

            # Plot with or without error bars
            if "std_latency" in sorted_group.columns and sorted_group["std_latency"].notna().any():
                error_color = MLX_ERROR_COLOR if "mlx" in str(src).lower() else source_style["color"]
                errorbar_container = ax_decode.errorbar(
                    sorted_group["sequence_length"],
                    sorted_group["mean_latency"],
                    yerr=sorted_group["std_latency"],
                    color=source_style["color"],
                    ecolor=error_color,
                    linestyle=source_style["linestyle"],
                    linewidth=source_style["linewidth"],
                    marker=source_style["marker"],
                    label=source_style["label"],
                    capsize=ERROR_BAR_CAPSIZE,
                    capthick=ERROR_BAR_CAPTHICK,
                    alpha=ERROR_BAR_ALPHA,
                )
                line = errorbar_container[0]
            else:
                (line,) = ax_decode.plot(
                    sorted_group["sequence_length"],
                    sorted_group["mean_latency"],
                    color=source_style["color"],
                    linestyle=source_style["linestyle"],
                    linewidth=source_style["linewidth"],
                    marker=source_style["marker"],
                    label=source_style["label"],
                )

            if "outline_color" in source_style and isinstance(line, Line2D):
                line.set_path_effects(
                    [
                        pe.Stroke(
                            linewidth=source_style["linewidth"] * 3,
                            foreground=source_style["outline_color"],
                        ),
                        pe.Normal(),
                    ]
                )

            x_data = sorted_group["sequence_length"].values
            y_data = sorted_group["mean_latency"].values
            assert isinstance(x_data, np.ndarray)
            assert isinstance(y_data, np.ndarray)

            # Calculate and display slope
            result = calculate_log_log_slope(x_data, y_data)
            if result is not None:
                slope, intercept = result
                label = source_style["label"].replace(r"$\mathbf{", "").replace(r"}$", "").replace("\\", "")
                logger.info(f"Decode {label}: slope = {slope:.2f} (O(n^{slope:.1f}))")

                ax_decode.text(
                    x_data[-1],
                    y_data[-1],
                    f"Slope ≈ {slope:.1f}",
                    color="red",
                    fontsize=12,
                    fontweight="bold",
                    horizontalalignment="right",
                    verticalalignment="bottom",
                )

        # Set decode plot aesthetics
        title = "Decode Latency vs History Length"
        plot_utils.apply_common_plot_aesthetics(
            ax_decode,
            title,
            "History Length (tokens)",
            "Mean Latency (ms)",
            styles,
            x_scale="log",
            y_scale="log",
            include_legend=True,
        )
        add_complexity_guide(ax_decode, "decode")

    # Configure tick formatting
    if ax_prefill is not None:
        ax_prefill.minorticks_on()
        ax_prefill.grid(linewidth=0.5, alpha=0.5)

    if ax_decode is not None:
        ax_decode.minorticks_on()
        ax_decode.grid(linewidth=0.5, alpha=0.5)

    # Save figure
    filename = "latency_vs_seq_len.png"
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / filename, dpi=300)
    plt.close(fig)

    # Save results as JSON
    output_dict = {}
    for group, group_df in df.groupby("group"):
        seq_lat_map = {
            float(seq_len): float(m_lat)
            for seq_len, m_lat in zip(group_df["sequence_length"], group_df["mean_latency"], strict=False)
        }
        output_dict[group] = seq_lat_map

    with open(output_dir / "results.json", "w") as f:
        json.dump(output_dict, f, indent=4)

    return filename
