"""Reporting utilities for benchmark analyzer."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .config import COL_MEAN_LATENCY, COL_SOURCE


def generate_json_report(df: pd.DataFrame, output_dir: Path, plot_filenames: dict[str, str]) -> None:
    """Generate a simple JSON report."""
    report = {"summary_metrics": {}, "plot_files": plot_filenames}
    if not df.empty:
        report["summary_metrics"]["rows"] = len(df)
    with open(output_dir / "results.json", "w") as f:
        json.dump(report, f, indent=2)
