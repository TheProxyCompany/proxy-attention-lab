"""Reporting utilities for benchmark analyzer."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from benchmarks.analyzer.config import (
    COL_BENCHMARK_NAME_BASE,
    COL_KERNEL_NAME,
    COL_MEAN_LATENCY,
    COL_SOURCE,
    COL_THROUGHPUT,
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _filter_for_plot_category(df: pd.DataFrame, category: str) -> pd.DataFrame:
    """Filter the DataFrame for a specific plot category."""
    if df.empty:
        return df

    # Define benchmark names for each category
    category_benchmark_mapping = {
        "latency_vs_seq_len": [
            "BM_PAL_LatencyVsSeqLen",
            "test_pal_latency_vs_seq_len",
            "BM_SDPA_LatencyVsSeqLen",
            "test_sdpa_latency_vs_seq_len",
        ],
        "latency_vs_head_dim": [
            "BM_PAL_LatencyVsHeadDim",
            "test_pal_latency_vs_head_dim",
            "BM_SDPA_LatencyVsHeadDim",
            "test_sdpa_latency_vs_head_dim",
        ],
        "latency_vs_effective_items": [
            "BM_PAL_LatencyVsNumItems",
            "test_pal_latency_vs_query_items",
            "BM_SDPA_LatencyVsNumItems",
            "test_sdpa_latency_vs_batch_size",
        ],
        "model_configs_latency": [
            "BM_PAL_ModelConfig",
            "test_pal_latency_model_configs",
            "BM_SDPA_ModelConfig",
            "test_sdpa_latency_model_configs",
        ],
    }

    benchmarks = category_benchmark_mapping.get(category, [])
    if not benchmarks:
        return pd.DataFrame()

    # Filter based on benchmark names if available
    if COL_BENCHMARK_NAME_BASE in df.columns:
        filtered_df = df[
            df[COL_BENCHMARK_NAME_BASE].apply(
                lambda x: any(x.startswith(b) for b in benchmarks) if isinstance(x, str) else False
            )
        ]

        if not filtered_df.empty:
            return filtered_df

    # If we get here, either we don't have benchmark name info or no matches were found
    # Try to filter based on the data columns relevant to each category
    if category == "latency_vs_seq_len":
        return df[df["seq_len"].notna()]
    elif category == "latency_vs_head_dim":
        return df[df["head_dim"].notna()]
    elif category == "latency_vs_effective_items":
        return df[df["effective_items"].notna()]
    elif category == "model_configs_latency":
        return df[df["model_config_name"].notna()]

    return pd.DataFrame()


def _get_metric_records(df: pd.DataFrame, category: str) -> dict[str, list[dict[str, Any]]] | list[dict[str, Any]]:
    """Convert filtered DataFrame to metrics records grouped by kernel (if available) or a flat list.

    Returns:
        Either a dictionary mapping kernel names to lists of records, or a flat list of records
        if kernel information is not available.
    """
    records = []

    # Get the filtered data for this category
    filtered_df = _filter_for_plot_category(df, category)
    if filtered_df.empty:
        logger.warning(f"No data for category {category} in metric records")
        return records

    # Define the columns to include in the metrics
    common_cols = [COL_SOURCE, COL_KERNEL_NAME, COL_MEAN_LATENCY, COL_THROUGHPUT]

    # Add category-specific columns
    if category == "latency_vs_seq_len":
        specific_cols = ["seq_len", "head_dim", "num_query_items", "batch_size", "num_q_heads"]
    elif category == "latency_vs_head_dim":
        specific_cols = ["head_dim", "seq_len", "num_query_items", "batch_size", "num_q_heads"]
    elif category == "latency_vs_effective_items":
        specific_cols = ["effective_items", "seq_len", "head_dim", "num_query_items", "batch_size", "num_q_heads"]
    elif category == "model_configs_latency":
        specific_cols = [
            "model_config_name",
            "seq_len",
            "head_dim",
            "num_query_items",
            "batch_size",
            "num_q_heads",
            "num_kv_heads",
            "tokens_per_page",
            "num_sequences_in_batch",
        ]
    else:
        specific_cols = []

    # Combine columns but only include those that exist in the DataFrame
    all_cols = common_cols + specific_cols
    existing_cols = [col for col in all_cols if col in filtered_df.columns]

    # Convert to records, but only include the columns that exist
    column_subset = filtered_df[existing_cols].copy()

    # Fill NaN values with Python None (better for JSON serialization)
    records = column_subset.replace({pd.NA: None}).to_dict(orient="records")

    # Group records by kernel if kernel_name is available
    if COL_KERNEL_NAME in filtered_df.columns:
        kernel_records = {}
        for record in records:
            kernel = record.get(COL_KERNEL_NAME, "unknown")
            if kernel not in kernel_records:
                kernel_records[kernel] = []
            kernel_records[kernel].append(record)

        # Convert to the final format if we have kernel information
        if kernel_records:
            return kernel_records

    # Otherwise return flat list of records
    return records


def generate_json_report(
    df: pd.DataFrame, output_dir: Path, plot_filenames: dict[str, str], kernel_filter: str | None = None
) -> None:
    """
    Generate a comprehensive JSON report with summary metrics.

    The report includes:
    - Overall metrics (count of unique sources, benchmarks, etc.)
    - Category-specific data points used in each plot
    - Filenames of generated plots

    Args:
        df: DataFrame with benchmark results.
        output_dir: Output directory for the report.
        plot_filenames: Dictionary mapping plot categories to filenames.
        kernel_filter: Optional kernel name to filter by (e.g., "paged_attention", "sdpa").
    """
    report = {"summary_metrics": {}, "plot_files": plot_filenames}

    # Store kernel filter in report if provided
    if kernel_filter:
        report["kernel_filter"] = kernel_filter
        logger.info(f"Applying kernel filter: {kernel_filter}")

    # Filter by kernel if specified
    filtered_df = df
    if kernel_filter and COL_KERNEL_NAME in df.columns:
        filtered_df = df[df[COL_KERNEL_NAME] == kernel_filter]
        if filtered_df.empty:
            logger.warning(f"No data found for kernel filter: {kernel_filter}")
            # Fall back to using all data, but note this in the report
            filtered_df = df
            report["warning"] = f"No data found for kernel filter '{kernel_filter}'. Using all kernels."

    if not filtered_df.empty:
        # Overall metrics
        report["summary_metrics"]["rows"] = len(filtered_df)

        # Count unique sources
        if COL_SOURCE in filtered_df.columns:
            sources = filtered_df[COL_SOURCE].unique().tolist()
            report["summary_metrics"]["sources"] = sources
            report["summary_metrics"]["source_count"] = len(sources)

        # Count unique benchmark types
        if COL_BENCHMARK_NAME_BASE in filtered_df.columns:
            benchmarks = filtered_df[COL_BENCHMARK_NAME_BASE].unique().tolist()
            report["summary_metrics"]["benchmarks"] = benchmarks
            report["summary_metrics"]["benchmark_count"] = len(benchmarks)

        # Count unique kernels
        if COL_KERNEL_NAME in filtered_df.columns:
            kernels = filtered_df[COL_KERNEL_NAME].unique().tolist()
            report["summary_metrics"]["kernels"] = kernels
            report["summary_metrics"]["kernel_count"] = len(kernels)

        # Add data for each plot category
        plot_categories = [
            "latency_vs_seq_len",
            "latency_vs_head_dim",
            "latency_vs_effective_items",
            "model_configs_latency",
        ]

        for category in plot_categories:
            # Get data points for this category (organized by kernel or flat)
            metrics = _get_metric_records(filtered_df, category)
            if metrics:
                # Add metrics to the report
                report["summary_metrics"][category] = metrics

                # Log metrics structure for debugging
                if isinstance(metrics, dict):
                    logger.debug(f"Category {category} has metrics for kernels: {list(metrics.keys())}")
                    logger.info(
                        f"Added {sum(len(v) for v in metrics.values())} records for {len(metrics)} kernels in category {category}"
                    )
                else:
                    logger.debug(f"Category {category} has {len(metrics)} metrics without kernel organization")
                    logger.info(f"Added {len(metrics)} records in category {category}")

    # Write report to file
    try:
        with open(output_dir / "results.json", "w") as f:
            json.dump(report, f, indent=2)
        logger.info(f"Report saved to {output_dir / 'results.json'}")
    except Exception as e:
        logger.error(f"Failed to write report: {e}")
