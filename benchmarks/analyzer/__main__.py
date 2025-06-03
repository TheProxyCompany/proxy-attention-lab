"""CLI tool for analyzing PAL benchmark results for custom kernels."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from benchmarks.analyzer.core import DataLoader, get_registry
from benchmarks.analyzer.core.plot_styles import get_plot_styles

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

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load benchmark data using the new data loader
    logger.info("Loading benchmark data from %s", args.results_dir)
    data_loader = DataLoader()

    try:
        benchmark_data = data_loader.load_directory(args.results_dir)
    except ValueError as e:
        logger.error(f"Failed to load benchmark data: {e}")
        return

    logger.info(
        f"Loaded {benchmark_data.metadata['total_benchmarks']} benchmarks from {benchmark_data.metadata['source_file_count']} files"
    )
    logger.info(f"Kernel types found: {', '.join(benchmark_data.metadata['kernel_types_found'])}")

    # Get the plotter registry and auto-discover plotters
    registry = get_registry()
    plotters_dir = Path(__file__).parent / "plotters"
    registry.auto_discover(plotters_dir)

    logger.info(f"Available plotters: {', '.join(registry.list_plotters())}")

    # Find compatible plotters for the loaded data
    compatible_plotters = registry.find_compatible_plotters(benchmark_data)

    if not compatible_plotters:
        logger.warning("No compatible plotters found for the loaded data")
        return

    # Get styles for plotting
    styles = get_plot_styles()

    # Generate plots and collect results
    logger.info("Generating plots...")
    plot_results = {}

    for plotter in compatible_plotters:
        logger.info(f"Running {plotter.get_name()} plotter...")
        try:
            result = plotter.plot(benchmark_data, args.output_dir, styles=styles)
            plot_results[plotter.get_name()] = result
            logger.info(f"Successfully generated {plotter.get_name()} plot")
        except Exception as e:
            logger.error(f"Error running {plotter.get_name()} plotter: {e}")
            if args.verbose:
                logger.exception("Full traceback:")

    # Log summary
    if plot_results:
        logger.info("Generated plots:")
        for plotter_name, result in plot_results.items():
            if isinstance(result, dict) and "filename" in result:
                logger.info(f"  - {plotter_name}: {result['filename']}")
            else:
                logger.info(f"  - {plotter_name}: completed")
    else:
        logger.warning("No plots were generated successfully")


if __name__ == "__main__":
    main()
