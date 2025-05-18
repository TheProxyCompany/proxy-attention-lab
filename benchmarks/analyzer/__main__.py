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


def process_cpp_file(file_data: dict) -> pd.DataFrame:
    benchmarks_data = file_data["benchmarks"]
    results_df = pd.DataFrame(benchmarks_data)
    results_df["group"] = results_df["name"].apply(
        lambda x: "cpp_pal_paged_attention" if "pal" in x.lower() else "cpp_mlx_sdpa"
    )
    if "param" not in results_df.columns:
        results_df["param"] = results_df["name"].apply(lambda x: x.split("/")[1])

    results_df["mean_latency"] = results_df["real_time"] / results_df["iterations"]
    # convert to milliseconds
    if "time_unit" in results_df.columns:
        match str(results_df["time_unit"].iloc[0]):
            case "ns":
                results_df["mean_latency"] = results_df["mean_latency"] / 1_000_000
            case "us":
                results_df["mean_latency"] = results_df["mean_latency"] / 1_000
            case "ms":
                pass

    results_df["sequence_length"] = results_df["param"].astype(float)

    return results_df


def process_python_file(file_data: dict) -> pd.DataFrame:
    benchmarks_data = file_data["benchmarks"]
    results_df = pd.DataFrame(benchmarks_data)
    results_df["group"] = results_df["name"].apply(
        lambda x: "python_pal_paged_attention" if "pal" in x.lower() else "python_mlx_sdpa"
    )

    results_df["sequence_length"] = results_df["param"].astype(float)
    results_df["mean_latency"] = results_df["stats"].apply(lambda x: x["mean"] * 1000)  # convert to milliseconds
    results_df["mean_latency"] = results_df["mean_latency"].astype(float)

    return results_df


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

    args.output_dir.mkdir(parents=True, exist_ok=True)

    results_df = pd.DataFrame()

    # Find JSON files to analyze (excluding previously generated results.json)
    json_files = [p for p in args.results_dir.glob("*.json") if p.name != "results.json"]
    if not json_files:
        logger.error("No JSON files found in %s", args.results_dir)
        return

    logger.info("Found %d JSON files to analyze", len(json_files))
    for json_file in json_files:
        with open(json_file) as f:
            file_data = json.load(f)
            if "python" in json_file.name:
                # assume any file with "python" in the name is a python benchmark file
                benchmarks_data = process_python_file(file_data)
            else:
                # assume any file without "python" in the name is a c++ benchmark file
                benchmarks_data = process_cpp_file(file_data)
            results_df = pd.concat([results_df, benchmarks_data])

    # Get styles for plotting
    styles = plot_utils.get_plot_styles()

    # Generate plots and collect filenames
    logger.info("Generating plots...")
    plot_filenames: dict[str, str] = {}

    # Plot latency vs sequence length
    logger.info("Generating latency vs sequence length plot...")
    seq_len_plot = latency_vs_seq_len.plot(df=results_df, output_dir=args.output_dir, styles=styles)

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
