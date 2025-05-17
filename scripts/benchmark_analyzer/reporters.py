"""Reporting utilities for benchmark analyzer."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .config import COL_KERNEL_TESTED, COL_MEAN_LATENCY


def generate_json_report(df: pd.DataFrame, output_dir: Path, plot_filenames: dict[str, str]) -> None:
    """Generate a simple JSON report grouped by kernel."""
    report = {"summary_metrics": {}, "plot_files": plot_filenames}
    for kernel, kdf in df.groupby(COL_KERNEL_TESTED):
        report["summary_metrics"][kernel] = {"mean_latency_ms": kdf[COL_MEAN_LATENCY].mean()}
    with open(output_dir / "results.json", "w") as f:
        json.dump(report, f, indent=2)
