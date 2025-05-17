"""Data transformation utilities for benchmark analyzer."""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from typing import Any

import pandas as pd

from benchmarks.analyzer import config

# Set up logging
logger = logging.getLogger(__name__)

# Registry of parameter parsers for different benchmark types
# The registry maps (kernel, source, benchmark_name_base) to a parsing function
# Wildcards "*" can be used for any of the three elements
PARAM_PARSERS: dict[tuple[str, str, str], Callable[[str], dict[str, Any]]] = {}


def _parse_int(value: str) -> int:
    """Parse integer from string, returning 0 if parse fails."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


# Register parameter parsers for new kernels
def register_kernel_parsers(kernel_name: str, parsers: dict[tuple[str, str], Callable[[str], dict[str, Any]]]) -> None:
    """
    Register parameter parsers for a new kernel.

    Args:
        kernel_name: The name of the kernel (e.g., 'new_kernel_xyz')
        parsers: A dictionary mapping (source, benchmark_name_base) to parser functions
    """
    for (source, benchmark_name_base), parser_func in parsers.items():
        key = (kernel_name, source, benchmark_name_base)
        PARAM_PARSERS[key] = parser_func
        # Also register wildcards for more flexible matching
        PARAM_PARSERS[(kernel_name, source, "*")] = parser_func


# Helper functions for parameter parsing, to be registered in the PARAM_PARSERS registry
def _extract_cpp_pal_latency_vs_seq_len(params_str: str) -> dict[str, Any]:
    """Extract parameters for BM_PAL_LatencyVsSeqLen benchmark."""
    result = {}
    if not params_str:
        return result

    # Log the input parameters string for debugging
    logger.debug(f"Parsing C++ PAL seq_len parameters: '{params_str}'")

    parts = params_str.split("/")
    if len(parts) >= 2:
        # For C++ benchmarks with _mean suffix (Google Benchmark format)
        if "_mean" in parts[1]:
            seq_len_str = parts[1].split("_mean")[0]
            try:
                seq_len = int(seq_len_str)
                logger.debug(f"  Extracted seq_len={seq_len} from {parts[1]}")
            except ValueError:
                logger.warning(f"Failed to parse seq_len from '{parts[1]}'")
                seq_len = config.DEFAULT_SEQ_LEN
        else:
            seq_len = _parse_int(parts[1])

        # For first part (num_query_items)
        if "_mean" in parts[0]:
            num_items_str = parts[0].split("_mean")[0]
            try:
                num_query_items = int(num_items_str)
                logger.debug(f"  Extracted num_query_items={num_query_items} from {parts[0]}")
            except ValueError:
                logger.warning(f"Failed to parse num_query_items from '{parts[0]}'")
                num_query_items = config.DEFAULT_NUM_QUERY_ITEMS
        else:
            num_query_items = _parse_int(parts[0])

        logger.debug(f"  Extracted: num_query_items={num_query_items}, seq_len={seq_len}")
        result["num_query_items"] = num_query_items
        result["seq_len"] = seq_len
    elif len(parts) == 1:
        # For C++ benchmarks with _mean suffix (Google Benchmark format)
        if "_mean" in parts[0]:
            seq_len_str = parts[0].split("_mean")[0]
            try:
                seq_len = int(seq_len_str)
                logger.debug(f"  Extracted seq_len={seq_len} from {parts[0]}")
            except ValueError:
                logger.warning(f"Failed to parse seq_len from '{parts[0]}'")
                seq_len = config.DEFAULT_SEQ_LEN
        else:
            seq_len = _parse_int(parts[0])

        logger.debug(f"  Extracted: seq_len={seq_len}, using default num_query_items")
        result["seq_len"] = seq_len
        result["num_query_items"] = config.DEFAULT_NUM_QUERY_ITEMS

    # Fill remaining fixed parameters
    result.setdefault("head_dim", config.DEFAULT_HEAD_DIM)
    result.setdefault("num_q_heads", config.DEFAULT_NUM_Q_HEADS)
    result.setdefault("num_kv_heads", config.DEFAULT_NUM_KV_HEADS)
    result.setdefault("tokens_per_page", config.DEFAULT_TOKENS_PER_PAGE)
    result.setdefault("num_sequences_in_batch", config.DEFAULT_NUM_SEQUENCES_IN_BATCH)

    return result


def _extract_cpp_sdpa_latency_vs_seq_len(params_str: str) -> dict[str, Any]:
    """Extract parameters for BM_SDPA_LatencyVsSeqLen benchmark."""
    result = {}
    if not params_str:
        return result

    # Log the input parameters string for debugging
    logger.debug(f"Parsing C++ SDPA seq_len parameters: '{params_str}'")

    parts = params_str.split("/")
    if len(parts) >= 2:
        # For C++ benchmarks with _mean suffix (Google Benchmark format)
        if "_mean" in parts[1]:
            seq_len_str = parts[1].split("_mean")[0]
            try:
                seq_len = int(seq_len_str)
                logger.debug(f"  Extracted seq_len={seq_len} from {parts[1]}")
            except ValueError:
                logger.warning(f"Failed to parse seq_len from '{parts[1]}'")
                seq_len = config.DEFAULT_SEQ_LEN
        else:
            seq_len = _parse_int(parts[1])

        # For first part (batch_size)
        if "_mean" in parts[0]:
            batch_size_str = parts[0].split("_mean")[0]
            try:
                batch_size = int(batch_size_str)
                logger.debug(f"  Extracted batch_size={batch_size} from {parts[0]}")
            except ValueError:
                logger.warning(f"Failed to parse batch_size from '{parts[0]}'")
                batch_size = config.DEFAULT_BATCH_SIZE
        else:
            batch_size = _parse_int(parts[0])

        logger.debug(f"  Extracted: batch_size={batch_size}, seq_len={seq_len}")
        result["batch_size"] = batch_size
        result["seq_len"] = seq_len
    elif len(parts) == 1:
        # For C++ benchmarks with _mean suffix (Google Benchmark format)
        if "_mean" in parts[0]:
            seq_len_str = parts[0].split("_mean")[0]
            try:
                seq_len = int(seq_len_str)
                logger.debug(f"  Extracted seq_len={seq_len} from {parts[0]}")
            except ValueError:
                logger.warning(f"Failed to parse seq_len from '{parts[0]}'")
                seq_len = config.DEFAULT_SEQ_LEN
        else:
            seq_len = _parse_int(parts[0])

        logger.debug(f"  Extracted: seq_len={seq_len}, using default batch_size")
        result["seq_len"] = seq_len
        result["batch_size"] = config.DEFAULT_BATCH_SIZE

    result.setdefault("head_dim", config.DEFAULT_HEAD_DIM)
    result.setdefault("num_q_heads", config.DEFAULT_NUM_Q_HEADS)
    result.setdefault("num_kv_heads", config.DEFAULT_NUM_KV_HEADS)

    return result


def _extract_python_pal_latency_vs_seq_len(params_str: str) -> dict[str, Any]:
    """Extract parameters for test_pal_latency_vs_seq_len benchmark."""
    result = {}
    if not params_str:
        return result

    # Format: seq_len_val=64
    if "seq_len_val=" in params_str:
        result["seq_len"] = _parse_int(params_str.split("seq_len_val=")[1])
    else:
        result["seq_len"] = _parse_int(params_str)

    # Add default values for other parameters
    result["num_query_items"] = config.DEFAULT_NUM_QUERY_ITEMS
    result["head_dim"] = config.DEFAULT_HEAD_DIM
    result["tokens_per_page"] = config.DEFAULT_TOKENS_PER_PAGE
    result["num_sequences_in_batch"] = config.DEFAULT_NUM_SEQUENCES_IN_BATCH

    return result


def _extract_python_sdpa_latency_vs_seq_len(params_str: str) -> dict[str, Any]:
    """Extract parameters for test_sdpa_latency_vs_seq_len benchmark."""
    result = {}
    if not params_str:
        return result

    # Format: seq_len_val=64
    if "seq_len_val=" in params_str:
        result["seq_len"] = _parse_int(params_str.split("seq_len_val=")[1])
    else:
        result["seq_len"] = _parse_int(params_str)

    # Add default values for other parameters
    result["batch_size"] = config.DEFAULT_BATCH_SIZE
    result["num_q_heads"] = config.DEFAULT_NUM_Q_HEADS
    result["num_kv_heads"] = config.DEFAULT_NUM_KV_HEADS
    result["head_dim"] = config.DEFAULT_HEAD_DIM

    return result


def _calculate_throughput(row: dict[str, Any]) -> float | None:
    """Calculate throughput in items per second if not already present."""
    source = row.get(config.COL_SOURCE, "")
    full_name = row.get("full_name", "unknown")

    existing_throughput = row.get(config.COL_THROUGHPUT)
    if (
        config.COL_THROUGHPUT in row
        and existing_throughput is not None
        and not pd.isna(existing_throughput)
        and existing_throughput > 0
    ):
        # Already have throughput value
        logger.debug(f"Using existing throughput for {full_name}: {existing_throughput:.2f} items/sec")
        return float(existing_throughput)

    # Get the effective items if already calculated
    effective_items = row.get("effective_items")
    if effective_items is None or pd.isna(effective_items):
        # Calculate effective items based on source
        effective_items = _calculate_effective_items(row)
        logger.debug(f"Calculated effective_items for {full_name}: {effective_items}")

    # If we still don't have effective items, we can't calculate throughput
    if effective_items is None or pd.isna(effective_items) or effective_items <= 0:
        logger.warning(f"Cannot calculate throughput for {full_name}: no valid effective_items")
        return None

    # Get the mean latency
    mean_latency_ms = row.get(config.COL_MEAN_LATENCY, 0)
    if pd.isna(mean_latency_ms) or mean_latency_ms <= 0:
        logger.warning(f"Cannot calculate throughput for {full_name}: invalid latency {mean_latency_ms}")
        return None

    # Calculate throughput: items per second = items / (latency in seconds)
    mean_latency_s = mean_latency_ms / 1000.0
    throughput = effective_items / mean_latency_s

    # Special handling for Python benchmarks which often need throughput calculated
    if source.startswith("python_"):
        logger.info(f"Python benchmark throughput for {full_name}: {throughput:.2f} items/sec")
        logger.info(f"  effective_items={effective_items}, mean_latency_ms={mean_latency_ms:.2f} ms")
    else:
        logger.debug(f"Calculated throughput for {full_name}: {throughput:.2f} items/sec")
        logger.debug(f"  effective_items={effective_items}, mean_latency_ms={mean_latency_ms:.2f} ms")

    return throughput


def _calculate_effective_items(row: dict[str, Any]) -> int | None:
    """
    Calculate effective items for fair comparison between PAL and SDPA.

    This is a critical measure for comparing different kernels:
    - For PAL: effective_items = num_query_items
    - For SDPA: effective_items = batch_size * num_q_heads * seq_len

    Both represent the total number of query vectors processed by the attention kernel.
    """
    source = row.get(config.COL_SOURCE, "")
    kernel_name = row.get(config.COL_KERNEL_NAME, "")
    full_name = row.get("full_name", "unknown")

    # Use both source and kernel_name for more reliable detection
    if "pal" in source or kernel_name == "paged_attention":
        # For PAL, effective items is num_query_items
        num_query_items = row.get("num_query_items")
        if num_query_items is not None and not pd.isna(num_query_items) and num_query_items > 0:
            logger.debug(f"PAL effective_items for {full_name}: {num_query_items} (num_query_items)")
            return int(num_query_items)
        logger.warning(f"PAL benchmark without valid num_query_items: {full_name}")
        return 0

    elif "sdpa" in source or kernel_name == "sdpa":
        # For SDPA, effective items is batch_size * num_q_heads * seq_len
        batch_size = row.get("batch_size")
        num_q_heads = row.get("num_q_heads", config.DEFAULT_NUM_Q_HEADS)
        seq_len = row.get("seq_len", config.DEFAULT_SEQ_LEN)

        if (
            batch_size is not None
            and num_q_heads is not None
            and seq_len is not None
            and not pd.isna(batch_size)
            and not pd.isna(num_q_heads)
            and not pd.isna(seq_len)
            and batch_size > 0
            and num_q_heads > 0
            and seq_len > 0
        ):
            effective = batch_size * num_q_heads * seq_len
            logger.debug(
                f"SDPA effective_items for {full_name}: {effective} (batch_size={batch_size} * num_q_heads={num_q_heads} * seq_len={seq_len})"
            )
            return int(effective)

        logger.warning(f"SDPA benchmark without valid batch_size, num_q_heads, or seq_len: {full_name}")
        return 0

    logger.warning(f"Unknown source/kernel for effective_items calculation: {source}/{kernel_name} in {full_name}")
    return None


def _get_parser_for_benchmark(kernel_name: str, source: str, base_name: str) -> Callable[[str], dict[str, Any]]:
    """
    Get the appropriate parameter parser for the benchmark.

    Args:
        kernel_name: The kernel name (e.g., 'paged_attention', 'sdpa')
        source: The source (e.g., 'cpp_pal', 'python_sdpa')
        base_name: The benchmark name base (e.g., 'BM_PAL_LatencyVsSeqLen')

    Returns:
        A function that parses parameters from the params_str
    """
    # Try to find an exact match in the registry
    key = (kernel_name, source, base_name)
    if key in PARAM_PARSERS:
        return PARAM_PARSERS[key]

    # Try with wildcards
    key = (kernel_name, source, "*")
    if key in PARAM_PARSERS:
        return PARAM_PARSERS[key]

    key = (kernel_name, "*", base_name)
    if key in PARAM_PARSERS:
        return PARAM_PARSERS[key]

    key = ("*", source, base_name)
    if key in PARAM_PARSERS:
        return PARAM_PARSERS[key]

    key = ("*", "*", base_name)  # Match only by benchmark name
    if key in PARAM_PARSERS:
        return PARAM_PARSERS[key]

    return lambda params_str: {}


def extract_and_normalize_parameters(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enrich benchmark DataFrame with normalized parameters.

    This function extracts and normalizes parameters from the benchmark results,
    ensuring consistent structure across different benchmark types and sources.

    Args:
        df: DataFrame with raw benchmark results.

    Returns:
        DataFrame with normalized parameters added.
    """
    if df.empty:
        return df

    # Create a copy to avoid modifying the original
    df = df.copy()

    # Initialize parameter columns
    parameter_columns = [
        "seq_len",
        "head_dim",
        "num_query_items",
        "batch_size",
        "num_q_heads",
        "num_kv_heads",
        "tokens_per_page",
        "num_sequences_in_batch",
        "model_config_name",
        "effective_items",
    ]

    for col in parameter_columns:
        if col not in df.columns:
            df[col] = pd.NA

    # Count by source for debugging
    source_counts = df[config.COL_SOURCE].value_counts().to_dict()
    logger.info(f"Processing {len(df)} rows: {source_counts}")

    # First pass: Extract model config names and clean them
    for idx, row in df.iterrows():
        if (
            "model_configs" in row[config.COL_BENCHMARK_NAME_BASE]
            or "ModelConfig" in row[config.COL_BENCHMARK_NAME_BASE]
        ):
            model_config_name = row.get("model_config_name")
            if model_config_name and isinstance(model_config_name, str):
                # Clean up model name - remove _mean, _median, etc suffixes from C++ benchmarks
                clean_model_name = re.sub(r"_(mean|median|stddev|cv)$", "", model_config_name)
                if clean_model_name != model_config_name:
                    logger.info(f"Cleaned model config name: {model_config_name} -> {clean_model_name}")
                    df.at[idx, "model_config_name"] = clean_model_name

    # Process each row to extract and normalize parameters
    for idx, row in df.iterrows():
        base_name = row[config.COL_BENCHMARK_NAME_BASE]
        params_str = row[config.COL_PARAMS_STR]
        source = row[config.COL_SOURCE]
        kernel_name = row.get(config.COL_KERNEL_NAME, "unknown")
        full_name = row.get("full_name", "unknown")

        logger.debug(f"Processing benchmark: name={base_name}, source={source}, kernel={kernel_name}")

        # Get the appropriate parser function for this benchmark
        parser_func = _get_parser_for_benchmark(kernel_name, source, base_name)  # type: ignore[reportCallIssue]

        # Extract varying parameters using the parser function
        extracted_params = parser_func(params_str)  # type: ignore[reportCallIssue]
        logger.debug(f"Extracted varying parameters for {full_name}: {extracted_params}")

        if "pal" in source or kernel_name == "paged_attention":
            # Fill in standard PAL parameters
            if "num_query_items" not in extracted_params:
                extracted_params["num_query_items"] = config.DEFAULT_NUM_QUERY_ITEMS
            if "head_dim" not in extracted_params:
                extracted_params["head_dim"] = config.DEFAULT_HEAD_DIM
            if "seq_len" not in extracted_params:
                extracted_params["seq_len"] = config.DEFAULT_SEQ_LEN
            if "num_q_heads" not in extracted_params:
                extracted_params["num_q_heads"] = config.DEFAULT_NUM_Q_HEADS
            if "num_kv_heads" not in extracted_params:
                extracted_params["num_kv_heads"] = config.DEFAULT_NUM_KV_HEADS
            if "tokens_per_page" not in extracted_params:
                extracted_params["tokens_per_page"] = config.DEFAULT_TOKENS_PER_PAGE
            if "num_sequences_in_batch" not in extracted_params:
                extracted_params["num_sequences_in_batch"] = config.DEFAULT_NUM_SEQUENCES_IN_BATCH

        elif "sdpa" in source or kernel_name == "sdpa":
            # Fill in standard SDPA parameters
            if "batch_size" not in extracted_params:
                extracted_params["batch_size"] = config.DEFAULT_BATCH_SIZE
            if "head_dim" not in extracted_params:
                extracted_params["head_dim"] = config.DEFAULT_HEAD_DIM
            if "seq_len" not in extracted_params:
                extracted_params["seq_len"] = config.DEFAULT_SEQ_LEN
            if "num_q_heads" not in extracted_params:
                extracted_params["num_q_heads"] = config.DEFAULT_NUM_Q_HEADS
            if "num_kv_heads" not in extracted_params:
                extracted_params["num_kv_heads"] = config.DEFAULT_NUM_KV_HEADS

        logger.debug(f"Final parameters after defaults for {full_name}: {extracted_params}")

        # Update the DataFrame with extracted parameters
        for param_name, param_value in extracted_params.items():
            df.at[idx, param_name] = param_value

    # After setting initial parameters, make a second pass for calculations
    for idx, row in df.iterrows():
        full_name = row.get("full_name", "unknown")

        # Calculate effective items (for fair comparison between PAL and SDPA)
        # Important: Calculate effective_items before throughput
        effective_items = _calculate_effective_items(row.to_dict())
        if effective_items is not None:
            df.at[idx, "effective_items"] = effective_items

        # Re-fetch the row to include the effective_items we just set
        updated_row = df.loc[idx].to_dict()  # type: ignore[reportCallIssue]

        # Calculate throughput if not already present
        throughput = _calculate_throughput(updated_row)  # type: ignore[reportCallIssue]
        if throughput is not None:
            df.at[idx, config.COL_THROUGHPUT] = throughput
            logger.debug(f"Set throughput for {full_name} to {throughput:.2f} items/sec")

    # Ensure all relevant columns have proper numeric dtypes
    numeric_cols = [
        "seq_len",
        "head_dim",
        "num_query_items",
        "batch_size",
        "num_q_heads",
        "num_kv_heads",
        "tokens_per_page",
        "num_sequences_in_batch",
        "effective_items",
        config.COL_MEAN_LATENCY,
        config.COL_THROUGHPUT,
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Print summary stats after processing
    logger.info(f"After processing: {len(df)} rows")
    for col in numeric_cols:
        if col in df.columns:
            non_null_count = df[col].count()
            null_count = df[col].isna().sum()
            valid_ratio = non_null_count / len(df) if len(df) > 0 else 0
            logger.info(f"Column {col}: {non_null_count} non-null, {null_count} null (valid ratio: {valid_ratio:.2f})")

    return df


# PAL C++ parameter parsers
PARAM_PARSERS[("paged_attention", "cpp_pal", "BM_PAL_LatencyVsSeqLen")] = _extract_cpp_pal_latency_vs_seq_len
# PAL Python parameter parsers
PARAM_PARSERS[("paged_attention", "python_pal", "test_pal_latency_vs_seq_len")] = _extract_python_pal_latency_vs_seq_len

# SDPA Python parameter parsers
PARAM_PARSERS[("sdpa", "python_sdpa", "test_sdpa_latency_vs_seq_len")] = _extract_python_sdpa_latency_vs_seq_len
# SDPA C++ parameter parsers
PARAM_PARSERS[("sdpa", "cpp_sdpa", "BM_SDPA_LatencyVsSeqLen")] = _extract_cpp_sdpa_latency_vs_seq_len
