"""CLI tool for analyzing PAL benchmark results for custom kernels."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from benchmarks.analyzer import config, loaders, plot_utils, reporters, transformers
from benchmarks.analyzer.plotters import latency_vs_seq_len

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
    parser.add_argument("--kernel", type=str, help="Filter results by kernel name (e.g., 'paged_attention', 'sdpa')")
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

    # Use kernel filter if provided
    kernel_filter = args.kernel
    if kernel_filter:
        logger.info(f"Filtering results for kernel: {kernel_filter}")
        # Verify kernel exists in data before proceeding
        if config.COL_KERNEL_NAME in df.columns:
            kernels = df[config.COL_KERNEL_NAME].unique()
            if kernel_filter not in kernels:
                logger.warning(
                    f"Warning: Selected kernel '{kernel_filter}' not found in data. Available kernels: {kernels.tolist()}"
                )
        else:
            logger.warning(
                f"Warning: No kernel_name column found in data, filter '{kernel_filter}' may not be applied correctly"
            )

    # Plot latency vs sequence length
    logger.info("Generating latency vs sequence length plot...")
    seq_len_plot = latency_vs_seq_len.plot(df, args.output_dir, styles, kernel_filter)
    if seq_len_plot:
        plot_filenames["latency_vs_seq_len"] = seq_len_plot
        logger.info(f"Successfully generated latency vs sequence length plot: {seq_len_plot}")
    else:
        logger.warning("Failed to generate latency vs sequence length plot - no suitable data found")

    # Generate JSON report with summary metrics
    logger.info("Generating JSON report...")
    reporters.generate_json_report(df, args.output_dir, plot_filenames, kernel_filter)

    # Log summary of what was generated
    num_plots = len(plot_filenames)
    kernel_info = f" for kernel '{kernel_filter}'" if kernel_filter else ""
    logger.info(f"Analysis complete. Generated {num_plots} plots{kernel_info}.")
    logger.info(f"Results saved to: {args.output_dir}/results.json")
    if plot_filenames:
        logger.info("Generated plots:")
        for category, filename in plot_filenames.items():
            logger.info(f"  - {category}: {filename}")
    else:
        logger.warning("No plots were generated - check logs for errors or data issues")


if __name__ == "__main__":
    main()
