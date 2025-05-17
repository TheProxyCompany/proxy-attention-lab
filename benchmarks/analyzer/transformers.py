"""Data transformation utilities for benchmark analyzer."""

from __future__ import annotations

import json
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


def _extract_cpp_pal_latency_vs_head_dim(params_str: str) -> dict[str, Any]:
    """Extract parameters for BM_PAL_LatencyVsHeadDim benchmark."""
    result = {}
    if not params_str:
        return result

    # Log the input parameters string for debugging
    logger.debug(f"Parsing C++ PAL head_dim parameters: '{params_str}'")

    parts = params_str.split("/")
    if len(parts) >= 2:
        # For C++ benchmarks with _mean suffix (Google Benchmark format)
        if "_mean" in parts[1]:
            head_dim_str = parts[1].split("_mean")[0]
            try:
                head_dim = int(head_dim_str)
                logger.debug(f"  Extracted head_dim={head_dim} from {parts[1]}")
            except ValueError:
                logger.warning(f"Failed to parse head_dim from '{parts[1]}'")
                head_dim = config.DEFAULT_HEAD_DIM
        else:
            head_dim = _parse_int(parts[1])

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

        logger.debug(f"  Extracted: num_query_items={num_query_items}, head_dim={head_dim}")
        result["num_query_items"] = num_query_items
        result["head_dim"] = head_dim
    elif len(parts) == 1:
        # For C++ benchmarks with _mean suffix (Google Benchmark format)
        if "_mean" in parts[0]:
            head_dim_str = parts[0].split("_mean")[0]
            try:
                head_dim = int(head_dim_str)
                logger.debug(f"  Extracted head_dim={head_dim} from {parts[0]}")
            except ValueError:
                logger.warning(f"Failed to parse head_dim from '{parts[0]}'")
                head_dim = config.DEFAULT_HEAD_DIM
        else:
            head_dim = _parse_int(parts[0])

        logger.debug(f"  Extracted: head_dim={head_dim}, using default num_query_items")
        result["head_dim"] = head_dim
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

    # Log the input parameters string for debugging
    logger.debug(f"Parsing C++ PAL num_items parameters: '{params_str}'")

    parts = params_str.split("/")
    if len(parts) >= 1:
        # For C++ benchmarks with _mean suffix (Google Benchmark format)
        # Extract the number from patterns like "32_mean"
        if "_mean" in parts[0]:
            num_items_str = parts[0].split("_mean")[0]
            try:
                result["num_query_items"] = int(num_items_str)
                logger.debug(f"Extracted num_query_items={result['num_query_items']} from {parts[0]}")
            except ValueError:
                logger.warning(f"Failed to parse num_query_items from '{parts[0]}'")
                result["num_query_items"] = config.DEFAULT_NUM_QUERY_ITEMS
        else:
            result["num_query_items"] = _parse_int(parts[0])
            logger.debug(f"Extracted num_query_items={result['num_query_items']}")

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


def _extract_cpp_sdpa_latency_vs_head_dim(params_str: str) -> dict[str, Any]:
    """Extract parameters for BM_SDPA_LatencyVsHeadDim benchmark."""
    result = {}
    if not params_str:
        return result

    # Log the input parameters string for debugging
    logger.debug(f"Parsing C++ SDPA head_dim parameters: '{params_str}'")

    parts = params_str.split("/")
    if len(parts) >= 2:
        # For C++ benchmarks with _mean suffix (Google Benchmark format)
        if "_mean" in parts[1]:
            head_dim_str = parts[1].split("_mean")[0]
            try:
                head_dim = int(head_dim_str)
                logger.debug(f"  Extracted head_dim={head_dim} from {parts[1]}")
            except ValueError:
                logger.warning(f"Failed to parse head_dim from '{parts[1]}'")
                head_dim = config.DEFAULT_HEAD_DIM
        else:
            head_dim = _parse_int(parts[1])

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

        logger.debug(f"  Extracted: batch_size={batch_size}, head_dim={head_dim}")
        result["batch_size"] = batch_size
        result["head_dim"] = head_dim
    elif len(parts) == 1:
        # For C++ benchmarks with _mean suffix (Google Benchmark format)
        if "_mean" in parts[0]:
            head_dim_str = parts[0].split("_mean")[0]
            try:
                head_dim = int(head_dim_str)
                logger.debug(f"  Extracted head_dim={head_dim} from {parts[0]}")
            except ValueError:
                logger.warning(f"Failed to parse head_dim from '{parts[0]}'")
                head_dim = config.DEFAULT_HEAD_DIM
        else:
            head_dim = _parse_int(parts[0])

        logger.debug(f"  Extracted: head_dim={head_dim}, using default batch_size")
        result["head_dim"] = head_dim
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

    # Log the input parameters string for debugging
    logger.debug(f"Parsing C++ SDPA num_items parameters: '{params_str}'")

    parts = params_str.split("/")
    if len(parts) >= 1:
        # For C++ benchmarks with _mean suffix (Google Benchmark format)
        # Extract the number from patterns like "32_mean"
        if "_mean" in parts[0]:
            batch_size_str = parts[0].split("_mean")[0]
            try:
                result["batch_size"] = int(batch_size_str)
                logger.debug(f"Extracted batch_size={result['batch_size']} from {parts[0]}")
            except ValueError:
                logger.warning(f"Failed to parse batch_size from '{parts[0]}'")
                result["batch_size"] = config.DEFAULT_BATCH_SIZE
        else:
            result["batch_size"] = _parse_int(parts[0])
            logger.debug(f"Extracted batch_size={result['batch_size']}")

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

    # Clean up model name - remove _mean, _median, etc suffixes that might be in C++ benchmarks
    clean_model_name = re.sub(r"_(mean|median|stddev|cv)$", "", model_config_name)

    # Handle special case for Qwen2.5 vs Qwen2_5 naming differences
    if "Qwen2_5" in clean_model_name:
        clean_model_name = clean_model_name.replace("Qwen2_5", "Qwen2.5")

    model_config_name = clean_model_name
    logger.debug(f"Processing model config: {model_config_name} (cleaned from {row.get('model_config_name')})")

    # Check if we have this model config in our config parameters
    if model_config_name in config.MODEL_CONFIG_PARAMETERS:
        model_params = config.MODEL_CONFIG_PARAMETERS[model_config_name]

        # Get shared parameters that apply to all configurations
        result["num_q_heads"] = model_params["num_q_heads"]
        result["num_kv_heads"] = model_params["num_kv_heads"]
        result["head_dim"] = model_params["head_dim"]
        result["seq_len"] = model_params["seq_len"]

        # Update the model_config_name to use the cleaned version
        result["model_config_name"] = model_config_name

        # Different parameter extraction based on source
        source = row.get(config.COL_SOURCE, "")
        kernel_name = row.get(config.COL_KERNEL_NAME, "")

        # Add source-specific parameters
        if "pal" in source.lower() or kernel_name == "paged_attention":
            result["tokens_per_page"] = model_params.get("tokens_per_page", config.DEFAULT_TOKENS_PER_PAGE)
            result["num_sequences_in_batch"] = model_params.get(
                "num_sequences_in_batch", config.DEFAULT_NUM_SEQUENCES_IN_BATCH
            )
            result["num_query_items"] = model_params.get("pal_num_query_items", config.DEFAULT_NUM_QUERY_ITEMS)
        elif "sdpa" in source.lower() or kernel_name == "sdpa":
            result["batch_size"] = model_params.get("sdpa_batch_size", config.DEFAULT_BATCH_SIZE)
            # Make sure num_q_heads is set for SDPA to calculate effective_items correctly
            result["num_q_heads"] = model_params["num_q_heads"]

        logger.debug(f"Extracted model params for {model_config_name}: {result}")
        return result
    else:
        logger.warning(f"Model config '{model_config_name}' not found in config.MODEL_CONFIG_PARAMETERS")

        # Try to handle Python benchmark parameters from raw params
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
    source = row.get(config.COL_SOURCE, "")
    full_name = row.get("full_name", "unknown")

    if config.COL_THROUGHPUT in row and row[config.COL_THROUGHPUT] is not None and row[config.COL_THROUGHPUT] > 0:
        # Already have throughput value
        logger.debug(f"Using existing throughput for {full_name}: {row[config.COL_THROUGHPUT]:.2f} items/sec")
        return row[config.COL_THROUGHPUT]

    # Get the effective items if already calculated
    effective_items = row.get("effective_items")
    if effective_items is None:
        # Calculate effective items based on source
        effective_items = _calculate_effective_items(row)
        logger.debug(f"Calculated effective_items for {full_name}: {effective_items}")

    # If we still don't have effective items, we can't calculate throughput
    if effective_items is None or effective_items <= 0:
        logger.warning(f"Cannot calculate throughput for {full_name}: no valid effective_items")
        return None

    # Get the mean latency
    mean_latency_ms = row.get(config.COL_MEAN_LATENCY, 0)
    if mean_latency_ms <= 0:
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
        num_query_items = row.get("num_query_items", 0)
        if num_query_items is not None and num_query_items > 0:
            logger.debug(f"PAL effective_items for {full_name}: {num_query_items} (num_query_items)")
            return num_query_items
        logger.warning(f"PAL benchmark without valid num_query_items: {full_name}")
        return 0

    elif "sdpa" in source or kernel_name == "sdpa":
        # For SDPA, effective items is batch_size * num_q_heads * seq_len
        batch_size = row.get("batch_size", 0)
        num_q_heads = row.get("num_q_heads", config.DEFAULT_NUM_Q_HEADS)
        seq_len = row.get("seq_len", config.DEFAULT_SEQ_LEN)

        if (
            batch_size is not None
            and num_q_heads is not None
            and seq_len is not None
            and batch_size > 0
            and num_q_heads > 0
            and seq_len > 0
        ):
            effective = batch_size * num_q_heads * seq_len
            logger.debug(
                f"SDPA effective_items for {full_name}: {effective} (batch_size={batch_size} * num_q_heads={num_q_heads} * seq_len={seq_len})"
            )
            return effective

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
        parser_func = _get_parser_for_benchmark(kernel_name, source, base_name)

        # Extract varying parameters using the parser function
        extracted_params = parser_func(params_str)
        logger.debug(f"Extracted varying parameters for {full_name}: {extracted_params}")

        # Additional processing for model config benchmarks
        if "model_configs" in base_name or "ModelConfig" in base_name:
            model_config_params = _extract_model_config_params(row.to_dict())
            logger.debug(f"Extracted model config parameters for {full_name}: {model_config_params}")
            extracted_params.update(model_config_params)
        else:
            # For non-model-config benchmarks, ensure all standard parameters have defaults
            # If not already populated by the parser_func
            # This ensures a complete set of parameters for each benchmark type

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
        throughput = _calculate_throughput(updated_row)
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

    # Log a few examples for debugging
    for source in df[config.COL_SOURCE].unique():
        sample = df[df[config.COL_SOURCE] == source].head(1)
        if not sample.empty:
            row = sample.iloc[0].to_dict()
            throughput = row.get(config.COL_THROUGHPUT)
            throughput_str = f"{throughput:.2f}" if throughput is not None else "None"
            logger.info(
                f"Sample {source} row: seq_len={row.get('seq_len')}, head_dim={row.get('head_dim')}, "
                + f"effective_items={row.get('effective_items')}, throughput={throughput_str}"
            )

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
