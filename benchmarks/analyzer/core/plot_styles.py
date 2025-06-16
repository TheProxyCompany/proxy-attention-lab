"""Shared plotting utilities."""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.axes import Axes


def get_plot_styles() -> dict[str, str | float]:
    """Return standardized plotting styles."""
    styles: dict[str, str | float] = {}
    styles["PAL_COLOR"] = "#DAD0AF"  # light gold
    styles["PAL_OUTLINE_COLOR"] = "#024645"  # green
    styles["MLX_COLOR"] = "#FFFFFF"  # white
    styles["MLX_OUTLINE_COLOR"] = "#000000"  # black
    styles["PAL_STYLE"] = "-"  # Solid
    styles["MLX_STYLE"] = "-"  # Dotted
    # Use consistent line widths for both kernels
    styles["PAL_LINEWIDTH"] = 1.5
    styles["MLX_LINEWIDTH"] = 1.5
    styles["PAL_MARKER"] = "o"
    styles["MLX_MARKER"] = "o"
    # Language-specific markers
    styles["CPP_MARKER"] = "o"  # Circle for C++
    styles["PYTHON_MARKER"] = "s"  # Square for Python
    styles["CPP_MARKERSIZE"] = 6
    styles["PYTHON_MARKERSIZE"] = 5
    styles["REF_LINE_COLOR"] = "red"
    styles["REF_LINE_STYLE"] = "-."
    styles["REF_LINE_WIDTH"] = 1.0
    styles["REF_LINE_ALPHA"] = 0.5
    styles["REF_LINE_FONTSIZE"] = 8
    styles["TITLE_FONTSIZE"] = 16
    styles["TITLE_FONTWEIGHT"] = "bold"
    styles["AXIS_LABEL_FONTSIZE"] = 12
    styles["TICK_LABEL_FONTSIZE"] = 10
    styles["LEGEND_FONTSIZE"] = 10
    styles["GRID_COLOR"] = "#D3D3D3"
    styles["GRID_ALPHA"] = 0.5
    styles["PAL_LABEL"] = r"$\mathbf{PAL\ PagedAttention}$"
    styles["MLX_LABEL"] = r"$\mathbf{MLX\ ScaledDotProductAttention}$"
    return styles


def apply_common_plot_aesthetics(
    ax: Axes,
    title: str,
    xlabel: str,
    ylabel: str,
    styles: dict[str, str | float],
    x_scale: str = "linear",
    y_scale: str = "linear",
    include_legend: bool = True,
) -> None:
    """Apply common aesthetics to a Matplotlib axis."""
    ax.set_title(title, fontsize=styles["TITLE_FONTSIZE"], fontweight=styles["TITLE_FONTWEIGHT"])
    ax.set_xlabel(xlabel, fontsize=styles["AXIS_LABEL_FONTSIZE"])
    ax.set_ylabel(ylabel, fontsize=styles["AXIS_LABEL_FONTSIZE"])
    ax.set_xscale(x_scale)
    ax.set_yscale(y_scale)
    ax.grid(True, which="both", alpha=styles["GRID_ALPHA"], color=styles["GRID_COLOR"])
    ax.tick_params(axis="both", which="both", labelsize=styles["TICK_LABEL_FONTSIZE"])
    if include_legend:
        ax.legend(loc="best", fontsize=styles["LEGEND_FONTSIZE"])
    plt.tight_layout()
