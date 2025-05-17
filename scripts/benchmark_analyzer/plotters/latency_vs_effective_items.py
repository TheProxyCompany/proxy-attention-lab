"""Plot latency vs effective items."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .. import plot_utils
from ..config import COL_KERNEL_TESTED, COL_LANGUAGE, COL_MEAN_LATENCY

STYLES = plot_utils.get_plot_styles()


def plot(df: pd.DataFrame, output_dir: Path, styles: dict[str, str | float] | None = None) -> dict[str, str]:
    """Generate latency vs effective items plots per kernel."""
    styles = styles or STYLES
    output_dir.mkdir(parents=True, exist_ok=True)
    filenames: dict[str, str] = {}
    for kernel, kdf in df.dropna(subset=["effective_items"]).groupby(COL_KERNEL_TESTED):
        fig, ax = plt.subplots(figsize=(8, 6))
        for lang, group in kdf.groupby(COL_LANGUAGE):
            ax.plot(group["effective_items"], group[COL_MEAN_LATENCY], label=lang, marker="o")
        plot_utils.apply_common_plot_aesthetics(
            ax,
            f"{kernel} Latency vs Effective Items",
            "Effective Items",
            "Mean Latency (ms)",
            styles,
            x_scale="log",
            y_scale="log",
        )
        filename = f"latency_vs_effective_items_{kernel}.png"
        fig.savefig(output_dir / filename, dpi=300)
        plt.close(fig)
        filenames[kernel] = filename
    return filenames
