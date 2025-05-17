"""CLI tool for analyzing PAL benchmark results for custom kernels."""

from __future__ import annotations

import argparse
from pathlib import Path

from benchmark_analyzer import loaders, plot_utils, reporters, transformers
from benchmark_analyzer.plotters import (
    latency_vs_effective_items,
    latency_vs_head_dim,
    latency_vs_seq_len,
    model_configs_latency,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze benchmark results")
    parser.add_argument("results_dir", type=Path, help="Directory with JSON results")
    parser.add_argument("output_dir", type=Path, help="Directory for generated artifacts")
    args = parser.parse_args()

    json_files = [p for p in args.results_dir.glob("*.json") if p.name != "results.json"]
    if not json_files:
        print("No JSON files found")
        return

    raw_df = loaders.load_all_results(json_files)
    if raw_df.empty:
        print("No data loaded")
        return

    df = transformers.extract_and_normalize_parameters(raw_df)
    styles = plot_utils.get_plot_styles()
    plot_filenames: dict[str, str] = {}
    for k, v in latency_vs_seq_len.plot(df, args.output_dir, styles).items():
        plot_filenames[f"latency_vs_seq_len_{k}"] = v
    for k, v in latency_vs_head_dim.plot(df, args.output_dir, styles).items():
        plot_filenames[f"latency_vs_head_dim_{k}"] = v
    for k, v in latency_vs_effective_items.plot(df, args.output_dir, styles).items():
        plot_filenames[f"latency_vs_effective_items_{k}"] = v
    model_plot = model_configs_latency.plot(df, args.output_dir, styles)
    if model_plot:
        plot_filenames["model_configs_latency"] = model_plot

    reporters.generate_json_report(df, args.output_dir, plot_filenames)
    print(f"Saved results to {args.output_dir}")


if __name__ == "__main__":
    main()
