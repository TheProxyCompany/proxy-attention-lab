"""Data transformation utilities for benchmark analyzer."""

from __future__ import annotations

import json
import logging
from typing import Any

import pandas as pd
from analyzer import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _parse_int(value: str) -> int:
    """Parse integer from string, returning 0 if parse fails."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _parse_float(value: str) -> float:
    """Parse float from string, returning 0.0 if parse fails."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _extract_cpp_pal_params(base_name: str, params_str: str) -> dict[str, Any]:
    """Extract parameters from C++ PAL benchmark parameter string."""
    result = {}

    # For C++ benchmarks, parameters typically have a specific format like "64/128" (items/headDim)
    if not params_str:
        return result

    parts = params_str.split("/")

    # Handle different benchmark types
    if base_name == "BM_PAL_LatencyVsSeqLen":
        # Format: num_query_items/seq_len
        if len(parts) >= 2:
            result["num_query_items"] = _parse_int(parts[0])
            result["seq_len"] = _parse_int(parts[1])
        elif len(parts) == 1:
            # Only seq_len provided
            result["seq_len"] = _parse_int(parts[0])
            result["num_query_items"] = config.DEFAULT_NUM_QUERY_ITEMS

    elif base_name == "BM_PAL_LatencyVsHeadDim":
        # Format: num_query_items/head_dim
        if len(parts) >= 2:
            result["num_query_items"] = _parse_int(parts[0])
            result["head_dim"] = _parse_int(parts[1])
        elif len(parts) == 1:
            # Only head_dim provided
            result["head_dim"] = _parse_int(parts[0])
            result["num_query_items"] = config.DEFAULT_NUM_QUERY_ITEMS

    elif base_name == "BM_PAL_LatencyVsNumItems":
        # Format: num_query_items
        if len(parts) >= 1:
            result["num_query_items"] = _parse_int(parts[0])

    elif "ModelConfig" in base_name:
        # Model configurations might have specific parameters
        # We'll extract model_config_name elsewhere, and populate parameters in a separate step
        pass

    return result


def _extract_cpp_sdpa_params(base_name: str, params_str: str) -> dict[str, Any]:
    """Extract parameters from C++ SDPA benchmark parameter string."""
    result = {}

    if not params_str:
        return result

    parts = params_str.split("/")

    # Handle different benchmark types
    if base_name == "BM_SDPA_LatencyVsSeqLen":
        # Format: batch_size/seq_len
        if len(parts) >= 2:
            result["batch_size"] = _parse_int(parts[0])
            result["seq_len"] = _parse_int(parts[1])
        elif len(parts) == 1:
            # Only seq_len provided
            result["seq_len"] = _parse_int(parts[0])
            result["batch_size"] = config.DEFAULT_BATCH_SIZE

    elif base_name == "BM_SDPA_LatencyVsHeadDim":
        # Format: batch_size/head_dim
        if len(parts) >= 2:
            result["batch_size"] = _parse_int(parts[0])
            result["head_dim"] = _parse_int(parts[1])
        elif len(parts) == 1:
            # Only head_dim provided
            result["head_dim"] = _parse_int(parts[0])
            result["batch_size"] = config.DEFAULT_BATCH_SIZE

    elif base_name == "BM_SDPA_LatencyVsNumItems":
        # Format: batch_size
        if len(parts) >= 1:
            result["batch_size"] = _parse_int(parts[0])

    elif "ModelConfig" in base_name:
        # Model configurations might have specific parameters
        # We'll extract model_config_name elsewhere, and populate parameters in a separate step
        pass

    return result


def _extract_python_pal_params(base_name: str, params_str: str) -> dict[str, Any]:
    """Extract parameters from Python PAL benchmark parameter string."""
    result = {}

    # For Python benchmarks, parameters are typically simpler, like "64" for the parametrized value
    if not params_str:
        return result

    # Handle different benchmark types
    if base_name == "test_pal_latency_vs_seq_len":
        # Format: seq_len_val=64
        if "seq_len_val=" in params_str:
            result["seq_len"] = _parse_int(params_str.split("seq_len_val=")[1])
        else:
            result["seq_len"] = _parse_int(params_str)

        # Add default values for other parameters
        result["num_query_items"] = config.DEFAULT_NUM_QUERY_ITEMS
        result["head_dim"] = config.DEFAULT_HEAD_DIM

    elif base_name == "test_pal_latency_vs_head_dim":
        # Format: head_dim_val=128
        if "head_dim_val=" in params_str:
            result["head_dim"] = _parse_int(params_str.split("head_dim_val=")[1])
        else:
            result["head_dim"] = _parse_int(params_str)

        # Add default values for other parameters
        result["num_query_items"] = config.DEFAULT_NUM_QUERY_ITEMS
        result["seq_len"] = config.DEFAULT_SEQ_LEN

    elif base_name == "test_pal_latency_vs_query_items":
        # Format: num_query_items_val=64
        if "num_query_items_val=" in params_str:
            result["num_query_items"] = _parse_int(params_str.split("num_query_items_val=")[1])
        else:
            result["num_query_items"] = _parse_int(params_str)

        # Add default values for other parameters
        result["head_dim"] = config.DEFAULT_HEAD_DIM
        result["seq_len"] = config.DEFAULT_SEQ_LEN

    elif base_name == "test_pal_latency_model_configs":
        # Model config parameters will be extracted separately from model_params_raw
        pass

    return result


def _extract_python_sdpa_params(base_name: str, params_str: str) -> dict[str, Any]:
    """Extract parameters from Python SDPA benchmark parameter string."""
    result = {}

    if not params_str:
        return result

    # Handle different benchmark types
    if base_name == "test_sdpa_latency_vs_seq_len":
        # Format: seq_len_val=64
        if "seq_len_val=" in params_str:
            result["seq_len"] = _parse_int(params_str.split("seq_len_val=")[1])
        else:
            result["seq_len"] = _parse_int(params_str)

        # Add default values for other parameters
        result["batch_size"] = config.DEFAULT_BATCH_SIZE
        result["num_q_heads"] = config.DEFAULT_NUM_Q_HEADS
        result["head_dim"] = config.DEFAULT_HEAD_DIM

    elif base_name == "test_sdpa_latency_vs_head_dim":
        # Format: head_dim_val=128
        if "head_dim_val=" in params_str:
            result["head_dim"] = _parse_int(params_str.split("head_dim_val=")[1])
        else:
            result["head_dim"] = _parse_int(params_str)

        # Add default values for other parameters
        result["batch_size"] = config.DEFAULT_BATCH_SIZE
        result["num_q_heads"] = config.DEFAULT_NUM_Q_HEADS
        result["seq_len"] = config.DEFAULT_SEQ_LEN

    elif base_name == "test_sdpa_latency_vs_batch_size":
        # Format: batch_size_val=64
        if "batch_size_val=" in params_str:
            result["batch_size"] = _parse_int(params_str.split("batch_size_val=")[1])
        else:
            result["batch_size"] = _parse_int(params_str)

        # Add default values for other parameters
        result["num_q_heads"] = config.DEFAULT_NUM_Q_HEADS
        result["head_dim"] = config.DEFAULT_HEAD_DIM
        result["seq_len"] = config.DEFAULT_SEQ_LEN

    elif base_name == "test_sdpa_latency_model_configs":
        # Model config parameters will be extracted separately from model_params_raw
        pass

    return result


def _extract_model_config_params(row: dict[str, Any]) -> dict[str, Any]:
    """Extract parameters from model config benchmarks."""
    result = {}

    model_config_name = row.get("model_config_name")
    if not model_config_name:
        return result

    # Different parameter extraction based on source
    source = row.get(config.COL_SOURCE, "")
    base_name = row.get(config.COL_BENCHMARK_NAME_BASE, "")

    if "python_pal" in source or "test_pal_" in base_name:
        # For Python PAL, check if we have raw model params
        if "model_params_raw" in row:
            try:
                # Try to parse the raw parameter string as JSON
                params_raw = row.get("model_params_raw")
                params_dict = {}
                if isinstance(params_raw, str):
                    # Attempt to parse as JSON if it looks like a dict
                    try:
                        params_dict = json.loads(params_raw)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse model_params_raw as JSON: {e}")

                        # Extract relevant parameters
                        result["num_query_items"] = params_dict.get("num_query_items", config.DEFAULT_NUM_QUERY_ITEMS)
                        result["num_q_heads"] = params_dict.get("num_q_heads", config.DEFAULT_NUM_Q_HEADS)
                        result["num_kv_heads"] = params_dict.get("num_kv_heads", config.DEFAULT_NUM_KV_HEADS)
                        result["head_dim"] = params_dict.get("head_dim", config.DEFAULT_HEAD_DIM)
                        result["seq_len"] = params_dict.get("seq_len", config.DEFAULT_SEQ_LEN)
                        result["tokens_per_page"] = params_dict.get("tokens_per_page", config.DEFAULT_TOKENS_PER_PAGE)
                        result["num_sequences_in_batch"] = params_dict.get(
                            "num_sequences_in_batch", config.DEFAULT_NUM_SEQUENCES_IN_BATCH
                        )
            except (json.JSONDecodeError, TypeError) as e:
                logger.warning(f"Failed to parse model_params_raw: {e}")

    elif "python_sdpa" in source or "test_sdpa_" in base_name:
        # For Python SDPA, check if we have raw model params
        if "model_params_raw" in row:
            try:
                params_raw = row["model_params_raw"]
                if isinstance(params_raw, str) and params_raw.startswith("{") and params_raw.endswith("}"):
                    params_dict = json.loads(params_raw)
                    result["batch_size"] = params_dict.get("batch_size", config.DEFAULT_BATCH_SIZE)
                    result["num_q_heads"] = params_dict.get("num_q_heads", config.DEFAULT_NUM_Q_HEADS)
                    result["num_kv_heads"] = params_dict.get("num_kv_heads", config.DEFAULT_NUM_KV_HEADS)
                    result["head_dim"] = params_dict.get("head_dim", config.DEFAULT_HEAD_DIM)
                    result["seq_len"] = params_dict.get("seq_len", config.DEFAULT_SEQ_LEN)
            except (json.JSONDecodeError, TypeError) as e:
                logger.warning(f"Failed to parse model_params_raw: {e}")

    elif "cpp_pal" in source or "BM_PAL_" in base_name:
        # For C++ PAL, hardcode parameters based on model config name
        if model_config_name == "Llama3_70B_Sim":
            result["num_query_items"] = 64 * 64  # 64 tokens in batch * 64 query heads
            result["num_q_heads"] = 64
            result["num_kv_heads"] = 8
            result["head_dim"] = 128
            result["seq_len"] = 1024
            result["tokens_per_page"] = 64
            result["num_sequences_in_batch"] = 1
        elif model_config_name == "Qwen_8B_Sim":
            result["num_query_items"] = 64 * 32  # 64 tokens in batch * 32 query heads
            result["num_q_heads"] = 32
            result["num_kv_heads"] = 32
            result["head_dim"] = 128
            result["seq_len"] = 1024
            result["tokens_per_page"] = 64
            result["num_sequences_in_batch"] = 1
        elif model_config_name == "Qwen2.5_72B_Sim":
            result["num_query_items"] = 64 * 128  # 64 tokens in batch * 128 query heads
            result["num_q_heads"] = 128
            result["num_kv_heads"] = 8
            result["head_dim"] = 128
            result["seq_len"] = 1024
            result["tokens_per_page"] = 64
            result["num_sequences_in_batch"] = 1

    elif "cpp_sdpa" in source or "BM_SDPA_" in base_name:
        # For C++ SDPA, hardcode parameters based on model config name
        if model_config_name == "Llama3_70B_Sim":
            result["batch_size"] = 4
            result["num_q_heads"] = 64
            result["num_kv_heads"] = 8
            result["head_dim"] = 128
            result["seq_len"] = 1024
        elif model_config_name == "Qwen_8B_Sim":
            result["batch_size"] = 4
            result["num_q_heads"] = 32
            result["num_kv_heads"] = 32
            result["head_dim"] = 128
            result["seq_len"] = 1024
        elif model_config_name == "Qwen2.5_72B_Sim":
            result["batch_size"] = 4
            result["num_q_heads"] = 128
            result["num_kv_heads"] = 8
            result["head_dim"] = 128
            result["seq_len"] = 1024

    return result


def _calculate_throughput(row: dict[str, Any]) -> float | None:
    """Calculate throughput in items per second if not already present."""
    if config.COL_THROUGHPUT in row and row[config.COL_THROUGHPUT] is not None and row[config.COL_THROUGHPUT] > 0:
        # Already have throughput value
        return row[config.COL_THROUGHPUT]

    # Calculate based on source and available parameters
    source = row.get(config.COL_SOURCE, "")
    mean_latency_ms = row.get(config.COL_MEAN_LATENCY, 0)

    if mean_latency_ms <= 0:
        return None

    # Convert ms to seconds for throughput calculation
    mean_latency_s = mean_latency_ms / 1000.0

    if "pal" in source:
        # For PAL, use num_query_items
        num_query_items = row.get("num_query_items", 0)
        if num_query_items is not None and num_query_items > 0:
            return num_query_items / mean_latency_s
    elif "sdpa" in source:
        # For SDPA, use batch_size * num_q_heads
        batch_size = row.get("batch_size", 0)
        num_q_heads = row.get("num_q_heads", config.DEFAULT_NUM_Q_HEADS)
        if batch_size is not None and num_q_heads is not None and batch_size > 0 and num_q_heads > 0:
            return (batch_size * num_q_heads) / mean_latency_s

    return None


def _calculate_effective_items(row: dict[str, Any]) -> int | None:
    """Calculate effective items for fair comparison between PAL and SDPA."""
    source = row.get(config.COL_SOURCE, "")

    if "pal" in source:
        # For PAL, effective items is num_query_items
        num_query_items = row.get("num_query_items", 0)
        return num_query_items if num_query_items is not None else 0
    elif "sdpa" in source:
        # For SDPA, effective items is batch_size * num_q_heads
        batch_size = row.get("batch_size", 0)
        num_q_heads = row.get("num_q_heads", config.DEFAULT_NUM_Q_HEADS)
        if batch_size is not None and num_q_heads is not None and batch_size > 0 and num_q_heads > 0:
            return batch_size * num_q_heads
        return 0

    return None


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

    # Process each row to extract and normalize parameters
    for idx, row in df.iterrows():
        base_name = row[config.COL_BENCHMARK_NAME_BASE]
        params_str = row[config.COL_PARAMS_STR]
        source = row[config.COL_SOURCE]

        # Extract parameters based on source and benchmark type
        extracted_params = {}

        if "cpp_pal" in source or (base_name and base_name.startswith("BM_PAL_")):
            extracted_params = _extract_cpp_pal_params(base_name, params_str)
        elif "cpp_sdpa" in source or (base_name and base_name.startswith("BM_SDPA_")):
            extracted_params = _extract_cpp_sdpa_params(base_name, params_str)
        elif "python_pal" in source or (base_name and base_name.startswith("test_pal_")):
            extracted_params = _extract_python_pal_params(base_name, params_str)
        elif "python_sdpa" in source or (base_name and base_name.startswith("test_sdpa_")):
            extracted_params = _extract_python_sdpa_params(base_name, params_str)

        # Additional processing for model config benchmarks
        if "model_configs" in base_name or "ModelConfig" in base_name:
            model_config_params = _extract_model_config_params(row.to_dict())
            extracted_params.update(model_config_params)

        # Update the DataFrame with extracted parameters
        for param_name, param_value in extracted_params.items():
            df.at[idx, param_name] = param_value

        # Calculate throughput if not already present
        throughput = _calculate_throughput(row.to_dict())
        if throughput is not None:
            df.at[idx, config.COL_THROUGHPUT] = throughput

        # Calculate effective items (for fair comparison between PAL and SDPA)
        effective_items = _calculate_effective_items(row.to_dict())
        if effective_items is not None:
            df.at[idx, "effective_items"] = effective_items

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

    return df
