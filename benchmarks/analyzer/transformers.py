"""Data transformation utilities for benchmark analyzer."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from typing import Any

import pandas as pd

from benchmarks.analyzer import config

logging.basicConfig(level=logging.INFO)
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


def _parse_float(value: str) -> float:
    """Parse float from string, returning 0.0 if parse fails."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


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

    parts = params_str.split("/")
    if len(parts) >= 2:
        result["num_query_items"] = _parse_int(parts[0])
        result["seq_len"] = _parse_int(parts[1])
    elif len(parts) == 1:
        result["seq_len"] = _parse_int(parts[0])
        result["num_query_items"] = config.DEFAULT_NUM_QUERY_ITEMS

    # Fill remaining fixed parameters
    result.setdefault("head_dim", config.DEFAULT_HEAD_DIM)
    result.setdefault("num_q_heads", config.DEFAULT_NUM_Q_HEADS)
    result.setdefault("num_kv_heads", config.DEFAULT_NUM_KV_HEADS)
    result.setdefault("tokens_per_page", config.DEFAULT_TOKENS_PER_PAGE)
    result.setdefault("num_sequences_in_batch", config.DEFAULT_NUM_SEQUENCES_IN_BATCH)

    return result


def _extract_cpp_pal_latency_vs_head_dim(params_str: str) -> dict[str, Any]:
    """Extract parameters for BM_PAL_LatencyVsHeadDim benchmark."""
    result = {}
    if not params_str:
        return result

    parts = params_str.split("/")
    if len(parts) >= 2:
        result["num_query_items"] = _parse_int(parts[0])
        result["head_dim"] = _parse_int(parts[1])
    elif len(parts) == 1:
        result["head_dim"] = _parse_int(parts[0])
        result["num_query_items"] = config.DEFAULT_NUM_QUERY_ITEMS

    result.setdefault("seq_len", config.DEFAULT_SEQ_LEN)
    result.setdefault("num_q_heads", config.DEFAULT_NUM_Q_HEADS)
    result.setdefault("num_kv_heads", config.DEFAULT_NUM_KV_HEADS)
    result.setdefault("tokens_per_page", config.DEFAULT_TOKENS_PER_PAGE)
    result.setdefault("num_sequences_in_batch", config.DEFAULT_NUM_SEQUENCES_IN_BATCH)

    return result


def _extract_cpp_pal_latency_vs_num_items(params_str: str) -> dict[str, Any]:
    """Extract parameters for BM_PAL_LatencyVsNumItems benchmark."""
    result = {}
    if not params_str:
        return result

    parts = params_str.split("/")
    if len(parts) >= 1:
        result["num_query_items"] = _parse_int(parts[0])

    result.setdefault("seq_len", config.DEFAULT_SEQ_LEN)
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

    parts = params_str.split("/")
    if len(parts) >= 2:
        result["batch_size"] = _parse_int(parts[0])
        result["seq_len"] = _parse_int(parts[1])
    elif len(parts) == 1:
        result["seq_len"] = _parse_int(parts[0])
        result["batch_size"] = config.DEFAULT_BATCH_SIZE

    result.setdefault("head_dim", config.DEFAULT_HEAD_DIM)
    result.setdefault("num_q_heads", config.DEFAULT_NUM_Q_HEADS)
    result.setdefault("num_kv_heads", config.DEFAULT_NUM_KV_HEADS)

    return result


def _extract_cpp_sdpa_latency_vs_head_dim(params_str: str) -> dict[str, Any]:
    """Extract parameters for BM_SDPA_LatencyVsHeadDim benchmark."""
    result = {}
    if not params_str:
        return result

    parts = params_str.split("/")
    if len(parts) >= 2:
        result["batch_size"] = _parse_int(parts[0])
        result["head_dim"] = _parse_int(parts[1])
    elif len(parts) == 1:
        result["head_dim"] = _parse_int(parts[0])
        result["batch_size"] = config.DEFAULT_BATCH_SIZE

    result.setdefault("seq_len", config.DEFAULT_SEQ_LEN)
    result.setdefault("num_q_heads", config.DEFAULT_NUM_Q_HEADS)
    result.setdefault("num_kv_heads", config.DEFAULT_NUM_KV_HEADS)

    return result


def _extract_cpp_sdpa_latency_vs_num_items(params_str: str) -> dict[str, Any]:
    """Extract parameters for BM_SDPA_LatencyVsNumItems benchmark."""
    result = {}
    if not params_str:
        return result

    parts = params_str.split("/")
    if len(parts) >= 1:
        result["batch_size"] = _parse_int(parts[0])

    result.setdefault("seq_len", config.DEFAULT_SEQ_LEN)
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


def _extract_python_pal_latency_vs_head_dim(params_str: str) -> dict[str, Any]:
    """Extract parameters for test_pal_latency_vs_head_dim benchmark."""
    result = {}
    if not params_str:
        return result

    # Format: head_dim_val=128
    if "head_dim_val=" in params_str:
        result["head_dim"] = _parse_int(params_str.split("head_dim_val=")[1])
    else:
        result["head_dim"] = _parse_int(params_str)

    # Add default values for other parameters
    result["num_query_items"] = config.DEFAULT_NUM_QUERY_ITEMS
    result["seq_len"] = config.DEFAULT_SEQ_LEN
    result["tokens_per_page"] = config.DEFAULT_TOKENS_PER_PAGE
    result["num_sequences_in_batch"] = config.DEFAULT_NUM_SEQUENCES_IN_BATCH

    return result


def _extract_python_pal_latency_vs_query_items(params_str: str) -> dict[str, Any]:
    """Extract parameters for test_pal_latency_vs_query_items benchmark."""
    result = {}
    if not params_str:
        return result

    # Format: num_query_items_val=64
    if "num_query_items_val=" in params_str:
        result["num_query_items"] = _parse_int(params_str.split("num_query_items_val=")[1])
    else:
        result["num_query_items"] = _parse_int(params_str)

    # Add default values for other parameters
    result["head_dim"] = config.DEFAULT_HEAD_DIM
    result["seq_len"] = config.DEFAULT_SEQ_LEN
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


def _extract_python_sdpa_latency_vs_head_dim(params_str: str) -> dict[str, Any]:
    """Extract parameters for test_sdpa_latency_vs_head_dim benchmark."""
    result = {}
    if not params_str:
        return result

    # Format: head_dim_val=128
    if "head_dim_val=" in params_str:
        result["head_dim"] = _parse_int(params_str.split("head_dim_val=")[1])
    else:
        result["head_dim"] = _parse_int(params_str)

    # Add default values for other parameters
    result["batch_size"] = config.DEFAULT_BATCH_SIZE
    result["num_q_heads"] = config.DEFAULT_NUM_Q_HEADS
    result["num_kv_heads"] = config.DEFAULT_NUM_KV_HEADS
    result["seq_len"] = config.DEFAULT_SEQ_LEN

    return result


def _extract_python_sdpa_latency_vs_batch_size(params_str: str) -> dict[str, Any]:
    """Extract parameters for test_sdpa_latency_vs_batch_size benchmark."""
    result = {}
    if not params_str:
        return result

    # Format: batch_size_val=64
    if "batch_size_val=" in params_str:
        result["batch_size"] = _parse_int(params_str.split("batch_size_val=")[1])
    else:
        result["batch_size"] = _parse_int(params_str)

    # Add default values for other parameters
    result["num_q_heads"] = config.DEFAULT_NUM_Q_HEADS
    result["num_kv_heads"] = config.DEFAULT_NUM_KV_HEADS
    result["head_dim"] = config.DEFAULT_HEAD_DIM
    result["seq_len"] = config.DEFAULT_SEQ_LEN

    return result


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

    # Check if we have this model config in our config parameters
    if model_config_name in config.MODEL_CONFIG_PARAMETERS:
        model_params = config.MODEL_CONFIG_PARAMETERS[model_config_name]

        # Get shared parameters that apply to all configurations
        result["num_q_heads"] = model_params["num_q_heads"]
        result["num_kv_heads"] = model_params["num_kv_heads"]
        result["head_dim"] = model_params["head_dim"]
        result["seq_len"] = model_params["seq_len"]

        # Different parameter extraction based on source
        source = row.get(config.COL_SOURCE, "")

        # Add source-specific parameters
        if "pal" in source.lower():
            result["tokens_per_page"] = model_params.get("tokens_per_page", config.DEFAULT_TOKENS_PER_PAGE)
            result["num_sequences_in_batch"] = model_params.get(
                "num_sequences_in_batch", config.DEFAULT_NUM_SEQUENCES_IN_BATCH
            )
            result["num_query_items"] = model_params.get("pal_num_query_items", config.DEFAULT_NUM_QUERY_ITEMS)
        elif "sdpa" in source.lower():
            result["batch_size"] = model_params.get("sdpa_batch_size", config.DEFAULT_BATCH_SIZE)

        return result

    # Python benchmark might have raw parameters
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
                        return result

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
            except Exception as e:
                logger.warning(f"Failed to parse model_params_raw: {e}")

    elif ("python_sdpa" in source or "test_sdpa_" in base_name) and "model_params_raw" in row:
        try:
            params_raw = row["model_params_raw"]
            if isinstance(params_raw, str) and params_raw.startswith("{") and params_raw.endswith("}"):
                params_dict = json.loads(params_raw)
                result["batch_size"] = params_dict.get("batch_size", config.DEFAULT_BATCH_SIZE)
                result["num_q_heads"] = params_dict.get("num_q_heads", config.DEFAULT_NUM_Q_HEADS)
                result["num_kv_heads"] = params_dict.get("num_kv_heads", config.DEFAULT_NUM_KV_HEADS)
                result["head_dim"] = params_dict.get("head_dim", config.DEFAULT_HEAD_DIM)
                result["seq_len"] = params_dict.get("seq_len", config.DEFAULT_SEQ_LEN)
        except Exception as e:
            logger.warning(f"Failed to parse model_params_raw: {e}")

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

    # Fallback to old-style extraction functions
    if "cpp_pal" in source or base_name.startswith("BM_PAL_"):
        if "LatencyVsSeqLen" in base_name:
            return _extract_cpp_pal_latency_vs_seq_len
        elif "LatencyVsHeadDim" in base_name:
            return _extract_cpp_pal_latency_vs_head_dim
        elif "LatencyVsNumItems" in base_name:
            return _extract_cpp_pal_latency_vs_num_items
    elif "cpp_sdpa" in source or base_name.startswith("BM_SDPA_"):
        if "LatencyVsSeqLen" in base_name:
            return _extract_cpp_sdpa_latency_vs_seq_len
        elif "LatencyVsHeadDim" in base_name:
            return _extract_cpp_sdpa_latency_vs_head_dim
        elif "LatencyVsNumItems" in base_name:
            return _extract_cpp_sdpa_latency_vs_num_items
    elif "python_pal" in source or base_name.startswith("test_pal_"):
        if "latency_vs_seq_len" in base_name:
            return _extract_python_pal_latency_vs_seq_len
        elif "latency_vs_head_dim" in base_name:
            return _extract_python_pal_latency_vs_head_dim
        elif "latency_vs_query_items" in base_name:
            return _extract_python_pal_latency_vs_query_items
    elif "python_sdpa" in source or base_name.startswith("test_sdpa_"):
        if "latency_vs_seq_len" in base_name:
            return _extract_python_sdpa_latency_vs_seq_len
        elif "latency_vs_head_dim" in base_name:
            return _extract_python_sdpa_latency_vs_head_dim
        elif "latency_vs_batch_size" in base_name:
            return _extract_python_sdpa_latency_vs_batch_size

    # Default to empty function
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

    # Process each row to extract and normalize parameters
    for idx, row in df.iterrows():
        base_name = row[config.COL_BENCHMARK_NAME_BASE]
        params_str = row[config.COL_PARAMS_STR]
        source = row[config.COL_SOURCE]
        kernel_name = row.get(config.COL_KERNEL_NAME, "unknown")

        # Get the appropriate parser function for this benchmark
        parser_func = _get_parser_for_benchmark(kernel_name, source, base_name)

        # Extract parameters using the parser function
        extracted_params = parser_func(params_str)

        # Additional processing for model config benchmarks
        if "model_configs" in base_name or "ModelConfig" in base_name:
            model_config_params = _extract_model_config_params(row.to_dict())
            extracted_params.update(model_config_params)

        # Update the DataFrame with extracted parameters
        for param_name, param_value in extracted_params.items():
            df.at[idx, param_name] = param_value

        # Re-fetch the updated row for calculations
        updated_row = df.loc[idx].to_dict()  # type: ignore[call-overload]

        # Calculate throughput if not already present
        throughput = _calculate_throughput(updated_row)
        if throughput is not None:
            df.at[idx, config.COL_THROUGHPUT] = throughput

        # Calculate effective items (for fair comparison between PAL and SDPA)
        effective_items = _calculate_effective_items(updated_row)
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


# Register all parameter parsers in the PARAM_PARSERS registry
# Format: (kernel_name, source, benchmark_name_base) -> parser_function

# PAL C++ parameter parsers
PARAM_PARSERS[("paged_attention", "cpp_pal", "BM_PAL_LatencyVsSeqLen")] = _extract_cpp_pal_latency_vs_seq_len
PARAM_PARSERS[("paged_attention", "cpp_pal", "BM_PAL_LatencyVsHeadDim")] = _extract_cpp_pal_latency_vs_head_dim
PARAM_PARSERS[("paged_attention", "cpp_pal", "BM_PAL_LatencyVsNumItems")] = _extract_cpp_pal_latency_vs_num_items

# SDPA C++ parameter parsers
PARAM_PARSERS[("sdpa", "cpp_sdpa", "BM_SDPA_LatencyVsSeqLen")] = _extract_cpp_sdpa_latency_vs_seq_len
PARAM_PARSERS[("sdpa", "cpp_sdpa", "BM_SDPA_LatencyVsHeadDim")] = _extract_cpp_sdpa_latency_vs_head_dim
PARAM_PARSERS[("sdpa", "cpp_sdpa", "BM_SDPA_LatencyVsNumItems")] = _extract_cpp_sdpa_latency_vs_num_items

# PAL Python parameter parsers
PARAM_PARSERS[("paged_attention", "python_pal", "test_pal_latency_vs_seq_len")] = _extract_python_pal_latency_vs_seq_len
PARAM_PARSERS[("paged_attention", "python_pal", "test_pal_latency_vs_head_dim")] = (
    _extract_python_pal_latency_vs_head_dim
)
PARAM_PARSERS[("paged_attention", "python_pal", "test_pal_latency_vs_query_items")] = (
    _extract_python_pal_latency_vs_query_items
)

# SDPA Python parameter parsers
PARAM_PARSERS[("sdpa", "python_sdpa", "test_sdpa_latency_vs_seq_len")] = _extract_python_sdpa_latency_vs_seq_len
PARAM_PARSERS[("sdpa", "python_sdpa", "test_sdpa_latency_vs_head_dim")] = _extract_python_sdpa_latency_vs_head_dim
PARAM_PARSERS[("sdpa", "python_sdpa", "test_sdpa_latency_vs_batch_size")] = _extract_python_sdpa_latency_vs_batch_size

# Also register base name lookup (regardless of kernel or source) for common benchmark types
PARAM_PARSERS[("*", "*", "BM_PAL_LatencyVsSeqLen")] = _extract_cpp_pal_latency_vs_seq_len
PARAM_PARSERS[("*", "*", "BM_PAL_LatencyVsHeadDim")] = _extract_cpp_pal_latency_vs_head_dim
PARAM_PARSERS[("*", "*", "BM_PAL_LatencyVsNumItems")] = _extract_cpp_pal_latency_vs_num_items
PARAM_PARSERS[("*", "*", "BM_SDPA_LatencyVsSeqLen")] = _extract_cpp_sdpa_latency_vs_seq_len
PARAM_PARSERS[("*", "*", "BM_SDPA_LatencyVsHeadDim")] = _extract_cpp_sdpa_latency_vs_head_dim
PARAM_PARSERS[("*", "*", "BM_SDPA_LatencyVsNumItems")] = _extract_cpp_sdpa_latency_vs_num_items
PARAM_PARSERS[("*", "*", "test_pal_latency_vs_seq_len")] = _extract_python_pal_latency_vs_seq_len
PARAM_PARSERS[("*", "*", "test_pal_latency_vs_head_dim")] = _extract_python_pal_latency_vs_head_dim
PARAM_PARSERS[("*", "*", "test_pal_latency_vs_query_items")] = _extract_python_pal_latency_vs_query_items
PARAM_PARSERS[("*", "*", "test_sdpa_latency_vs_seq_len")] = _extract_python_sdpa_latency_vs_seq_len
PARAM_PARSERS[("*", "*", "test_sdpa_latency_vs_head_dim")] = _extract_python_sdpa_latency_vs_head_dim
PARAM_PARSERS[("*", "*", "test_sdpa_latency_vs_batch_size")] = _extract_python_sdpa_latency_vs_batch_size
