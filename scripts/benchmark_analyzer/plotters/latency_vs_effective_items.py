"""Plot latency vs effective items."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .. import plot_utils
from ..config import COL_MEAN_LATENCY, COL_SOURCE

STYLES = plot_utils.get_plot_styles()


def plot(df: pd.DataFrame, output_dir: Path, styles: dict[str, str | float] | None = None) -> str:
    """Generate latency vs effective items plot."""
    styles = styles or STYLES
    fig, ax = plt.subplots(figsize=(8, 6))
    for src, group in df.dropna(subset=["effective_items"]).groupby(COL_SOURCE):
        ax.plot(group["effective_items"], group[COL_MEAN_LATENCY], label=src, marker="o")
    plot_utils.apply_common_plot_aesthetics(
        ax,
        "Latency vs Effective Items",
        "Effective Items",
        "Mean Latency (ms)",
        styles,
        x_scale="log",
        y_scale="log",
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = "latency_vs_effective_items.png"
    fig.savefig(output_dir / filename, dpi=300)
    plt.close(fig)
    return filename
