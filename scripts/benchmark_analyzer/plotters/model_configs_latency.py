"""Plot latency for model configurations."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .. import plot_utils
from ..config import COL_KERNEL_TESTED, COL_MEAN_LATENCY, COL_SOURCE

STYLES = plot_utils.get_plot_styles()


def plot(df: pd.DataFrame, output_dir: Path, styles: dict[str, str | float] | None = None) -> str:
    """Generate model configuration latency bar chart."""
    styles = styles or STYLES
    cfg_df = df.dropna(subset=["model_config_name"])
    if cfg_df.empty:
        return ""
    kernel = cfg_df[COL_KERNEL_TESTED].unique()[0] if not cfg_df.empty else "kernel"
    fig, ax = plt.subplots(figsize=(8, 6))
    pivot = cfg_df.pivot_table(index="model_config_name", columns=COL_SOURCE, values=COL_MEAN_LATENCY)
    pivot.plot.bar(ax=ax, color=[styles.get("PAL_BAR_COLOR", "#333333"), styles.get("SDPA_BAR_COLOR", "#777777")])
    plot_utils.apply_common_plot_aesthetics(
        ax,
        f"{kernel} Model Configuration Latency",
        "Model Config",
        "Mean Latency (ms)",
        styles,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = "model_configs_latency.png"
    fig.savefig(output_dir / filename, dpi=300)
    plt.close(fig)
    return filename
