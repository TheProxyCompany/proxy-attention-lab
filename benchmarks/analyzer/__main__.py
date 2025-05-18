"""CLI tool for analyzing PAL benchmark results for custom kernels."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

from benchmarks.analyzer import plot_utils
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
    with open(json_files[0]) as f:
        benchmarks_data = json.load(f)["benchmarks"]

    df = pd.DataFrame(benchmarks_data)
    df["group"] = df["name"].apply(lambda x: "python_pal" if "pal" in x.lower() else "python_sdpa")

    # Get styles for plotting
    styles = plot_utils.get_plot_styles()

    # Generate plots and collect filenames
    logger.info("Generating plots...")
    plot_filenames: dict[str, str] = {}

    # Plot latency vs sequence length
    logger.info("Generating latency vs sequence length plot...")
    seq_len_plot = latency_vs_seq_len.plot(df, args.output_dir, styles)
    if seq_len_plot:
        plot_filenames["latency_vs_seq_len"] = seq_len_plot
        logger.info(f"Successfully generated latency vs sequence length plot: {seq_len_plot}")

    logger.info(f"Results saved to: {args.output_dir}/results.json")
    if plot_filenames:
        logger.info("Generated plots:")
        for category, filename in plot_filenames.items():
            logger.info(f"  - {category}: {filename}")


if __name__ == "__main__":
    main()
