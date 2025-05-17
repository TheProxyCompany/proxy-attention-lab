"""CLI tool for analyzing PAL benchmark results for custom kernels."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from benchmark_analyzer import config, loaders, plot_utils, reporters, transformers
from benchmark_analyzer.plotters import (
    latency_vs_effective_items,
    latency_vs_head_dim,
    latency_vs_seq_len,
    model_configs_latency,
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def main() -> None:
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Analyze benchmark results")
    parser.add_argument("results_dir", type=Path, help="Directory with JSON results")
    parser.add_argument("output_dir", type=Path, help="Directory for generated artifacts")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    args = parser.parse_args()

    # Set logging level based on verbosity
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Create output directory if it doesn't exist
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Find JSON files to analyze (excluding previously generated results.json)
    json_files = [p for p in args.results_dir.glob("*.json") if p.name != "results.json"]
    if not json_files:
        logger.error("No JSON files found in %s", args.results_dir)
        return

    logger.info("Found %d JSON files to analyze", len(json_files))

    # Load data from all JSON files
    logger.info("Loading benchmark data...")
    raw_df = loaders.load_all_results(json_files)
    if raw_df.empty:
        logger.error("No data loaded from JSON files")
        return

    logger.info("Loaded %d benchmark results", len(raw_df))

    # Extract and normalize parameters
    logger.info("Extracting and normalizing parameters...")
    df = transformers.extract_and_normalize_parameters(raw_df)

    # Debug information if verbose
    if args.verbose:
        print("\nDataFrame columns:")
        print(df.columns.tolist())
        print("\nDataFrame shape:", df.shape)
        print("\nSource distribution:")
        if config.COL_SOURCE in df.columns:
            print(df[config.COL_SOURCE].value_counts())
        print("\nFirst few rows:")
        print(df.head())

    # Get styles for plotting
    styles = plot_utils.get_plot_styles()

    # Generate plots and collect filenames
    logger.info("Generating plots...")
    plot_filenames: dict[str, str] = {}

    # Plot latency vs sequence length
    seq_len_plot = latency_vs_seq_len.plot(df, args.output_dir, styles)
    if seq_len_plot:
        plot_filenames["latency_vs_seq_len"] = seq_len_plot
        logger.info("Generated latency vs sequence length plot")

    # Plot latency vs head dimension
    head_dim_plot = latency_vs_head_dim.plot(df, args.output_dir, styles)
    if head_dim_plot:
        plot_filenames["latency_vs_head_dim"] = head_dim_plot
        logger.info("Generated latency vs head dimension plot")

    # Plot latency vs effective items
    effective_items_plot = latency_vs_effective_items.plot(df, args.output_dir, styles)
    if effective_items_plot:
        plot_filenames["latency_vs_effective_items"] = effective_items_plot
        logger.info("Generated latency vs effective items plot")

    # Plot model configurations latency
    model_plot = model_configs_latency.plot(df, args.output_dir, styles)
    if model_plot:
        plot_filenames["model_configs_latency"] = model_plot
        logger.info("Generated model configurations plot")

    # Generate JSON report with summary metrics
    logger.info("Generating JSON report...")
    reporters.generate_json_report(df, args.output_dir, plot_filenames)

    logger.info("Analysis complete. Results saved to %s", args.output_dir)


if __name__ == "__main__":
    main()
