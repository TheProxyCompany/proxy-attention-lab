"""Plot latency as a function of sequence length."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .. import plot_utils
from ..config import COL_MEAN_LATENCY, COL_SOURCE


STYLES = plot_utils.get_plot_styles()


def plot(df: pd.DataFrame, output_dir: Path, styles: dict[str, str | float] | None = None) -> str:
    """Generate latency vs sequence length plot."""
    styles = styles or STYLES
    fig, ax = plt.subplots(figsize=(8, 6))
    for src, group in df.dropna(subset=["seq_len"]).groupby(COL_SOURCE):
        ax.plot(
            group["seq_len"],
            group[COL_MEAN_LATENCY],
            label=src,
            marker="o",
        )
    plot_utils.apply_common_plot_aesthetics(
        ax,
        "Latency vs Sequence Length",
        "Sequence Length",
        "Mean Latency (ms)",
        styles,
        x_scale="log",
        y_scale="log",
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = "latency_vs_seq_len.png"
    fig.savefig(output_dir / filename, dpi=300)
    plt.close(fig)
    return filename
