"""Shared plotting utilities."""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.axes import Axes


def get_plot_styles() -> dict[str, str | float]:
    """Return standardized plotting styles."""
    styles: dict[str, str | float] = {}
    styles["PAL_CPP_COLOR"] = "#024645"  # Teal
    styles["PAL_PY_COLOR"] = "#026664"  # Slightly lighter teal
    styles["SDPA_CPP_COLOR"] = "#000000"  # Black
    styles["SDPA_PY_COLOR"] = "#333333"  # Dark gray
    styles["PAL_CPP_STYLE"] = "--"  # Dashed
    styles["PAL_PY_STYLE"] = "-"  # Solid
    styles["SDPA_CPP_STYLE"] = "-."  # Dash-dot
    styles["SDPA_PY_STYLE"] = ":"  # Dotted
    styles["PAL_LINEWIDTH"] = 2.5
    styles["SDPA_LINEWIDTH"] = 1.5
    styles["PAL_CPP_MARKER"] = "o"
    styles["PAL_PY_MARKER"] = "s"  # Square
    styles["SDPA_CPP_MARKER"] = "D"  # Diamond
    styles["SDPA_PY_MARKER"] = "^"  # Triangle
    styles["REF_LINE_COLOR"] = "gray"
    styles["REF_LINE_STYLE"] = "--"
    styles["REF_LINE_WIDTH"] = 1.0
    styles["REF_LINE_ALPHA"] = 0.6
    styles["REF_LINE_FONTSIZE"] = 12
    styles["TITLE_FONTSIZE"] = 16
    styles["TITLE_FONTWEIGHT"] = "bold"
    styles["AXIS_LABEL_FONTSIZE"] = 12
    styles["TICK_LABEL_FONTSIZE"] = 10
    styles["LEGEND_FONTSIZE"] = 10
    styles["GRID_COLOR"] = "#D3D3D3"
    styles["GRID_ALPHA"] = 0.5
    styles["GRID_LINESTYLE"] = ":"
    styles["PAL_CPP_LABEL"] = r"$\mathbf{Paged\ Attention\ (C++)}$"
    styles["PAL_PY_LABEL"] = r"$\mathbf{Paged\ Attention\ (Python)}$"
    styles["SDPA_CPP_LABEL"] = "MLX SDPA (C++)"
    styles["SDPA_PY_LABEL"] = "MLX SDPA (Python)"
    styles["PAL_BAR_COLOR"] = "#024645"
    styles["SDPA_BAR_COLOR"] = "#000000"
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
    ax.grid(True, which="both", ls=styles["GRID_LINESTYLE"], alpha=styles["GRID_ALPHA"], color=styles["GRID_COLOR"])
    ax.tick_params(axis="both", which="major", labelsize=styles["TICK_LABEL_FONTSIZE"])
    if include_legend:
        ax.legend(loc="best", fontsize=styles["LEGEND_FONTSIZE"])
    plt.tight_layout()
