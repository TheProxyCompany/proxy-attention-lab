# Copyright 2024 The Proxy Company. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Performance benchmarks for paged attention using pytest-benchmark.

This module provides comprehensive performance benchmarking for the paged_attention
operation in the Proxy Attention Lab (PAL) library using the pytest-benchmark framework.

The benchmarks measure the latency of the paged_attention kernel under various
parameter configurations:

1. Sequence length (test_pal_latency_vs_seq_len):
   Evaluates how latency scales with different sequence lengths.

2. Head dimension (test_pal_latency_vs_head_dim):
   Evaluates how latency scales with different head dimensions.

3. Query items (test_pal_latency_vs_query_items):
   Evaluates how latency scales with different numbers of query items.

4. Model configurations (test_pal_latency_model_configs):
   Tests performance with parameter sets that simulate real-world model configurations.

Usage:
    Run all benchmarks:
    $ pytest tests/paged_attention/test_pal_performance.py -v

    Run a specific benchmark:
    $ pytest tests/paged_attention/test_pal_performance.py::test_pal_latency_vs_seq_len -v

    Save benchmark results to JSON:
    $ pytest tests/paged_attention/test_pal_performance.py --benchmark-json=results.json

Notes:
    - These benchmarks use pytest-benchmark to obtain robust measurements with proper
      warm-up and statistical analysis.
    - All input tensors are pre-computed outside the benchmark timer to ensure
      only the core operation is measured.
    - mx.eval() is called within the benchmarked function to ensure GPU operations complete.
"""

import logging
from typing import Any

import mlx.core as mx
import mlx.nn
import pytest

from proxy_attention_lab import paged_attention

logger = logging.getLogger(__name__)

# Fixed batch size for latency vs sequence length comparisons
# Using a small number to better understand per-sequence scaling behavior
COMPARISON_BATCH_SIZE = 4

# Define baseline configuration for benchmarks
BASELINE_CONFIG = {
    "num_query_items": 0,  # This will be set dynamically in the tests
    "num_q_heads": 1,
    "num_kv_heads": 1,
    "head_dim": 128,
    "seq_len": 128,  # History length
    "tokens_per_page": 64,
    "num_sequences_in_batch": COMPARISON_BATCH_SIZE,
    "dtype": mx.float16,
}


def setup_sdpa_benchmark_inputs(params: dict[str, Any]) -> tuple[mx.array, mx.array, mx.array, float, mx.array | None]:
    """
    Create all necessary input tensors for the MLX scaled_dot_product_attention benchmark.

    Args:
        params: Dictionary containing benchmark parameters:
            - batch_size: Number of sequences in batch
            - num_q_heads: Number of query heads
            - num_kv_heads: Number of K/V heads
            - head_dim: Dimension of each head
            - seq_len: Sequence length
            - dtype: Data type for tensors (e.g., mx.float16)

    Returns:
        Tuple containing all the input tensors in the order needed for scaled_dot_product_attention:
            - queries: [batch_size, num_q_heads, seq_len, head_dim]
            - keys: [batch_size, num_kv_heads, seq_len, head_dim]
            - values: [batch_size, num_kv_heads, seq_len, head_dim]
            - scale: scaling factor (1.0 / sqrt(head_dim))
            - causal_mask: causal attention mask [seq_len, seq_len] or None

    Raises:
        ValueError: If parameters are incompatible
    """
    # Compute scale factor
    scale = 1.0 / mx.sqrt(float(params["head_dim"]))

    # Create input tensors
    queries = mx.random.normal(
        (params["batch_size"], params["num_q_heads"], params["seq_len"], params["head_dim"]), dtype=params["dtype"]
    )

    keys = mx.random.normal(
        (params["batch_size"], params["num_kv_heads"], params["seq_len"], params["head_dim"]), dtype=params["dtype"]
    )

    values = mx.random.normal(
        (params["batch_size"], params["num_kv_heads"], params["seq_len"], params["head_dim"]), dtype=params["dtype"]
    )

    # Create causal mask
    causal_mask = mlx.nn.MultiHeadAttention.create_additive_causal_mask(params["seq_len"]).astype(params["dtype"])

    # Evaluate all tensors before returning to ensure they're computed
    mx.eval(queries)
    mx.eval(keys)
    mx.eval(values)
    mx.eval(causal_mask)

    return queries, keys, values, scale, causal_mask


def setup_pal_benchmark_inputs(params: dict[str, Any]) -> tuple[mx.array, ...]:
    """
    Create all necessary input tensors for the paged_attention kernel benchmark.

    This function supports both standard benchmarks and batched prefill scenarios.
    For batched prefill benchmarks (used in test_pal_latency_vs_seq_len):
    - num_query_items = num_sequences_in_batch * seq_len * num_q_heads
    - Query token offsets represent the position of each token in its sequence (1-indexed)
    - Each sequence is assigned a contiguous block of tokens

    Args:
        params: Dictionary containing benchmark parameters:
            - num_query_items: Total number of query-head items to process
            - num_q_heads: Number of query heads
            - num_kv_heads: Number of K/V heads
            - head_dim: Dimension of each head
            - seq_len: Effective history length for all items
            - tokens_per_page: Number of tokens per page (typically 64)
            - num_sequences_in_batch: Number of sequences in the batch
            - dtype: Data type for tensors (e.g., mx.float16)

    Returns:
        Tuple containing all the input tensors in the order needed for paged_attention:
            - queries: [num_tokens, num_q_heads, head_dim]
            - k_cache_pool: [num_total_physical_pages, tokens_per_page, num_kv_heads, head_dim]
            - v_cache_pool: [num_total_physical_pages, tokens_per_page, num_kv_heads, head_dim]
            - page_table: [num_sequences_in_batch, num_logical_pages_per_seq]
            - sequence_lengths: [num_sequences_in_batch]
            - query_to_seq_map: [num_tokens]
            - query_token_offset: [num_tokens]

    Raises:
        ValueError: If parameters are incompatible (e.g., num_query_items not divisible by num_q_heads)
    """
    # Validate input parameters
    if params["num_q_heads"] > 0 and params["num_query_items"] % params["num_q_heads"] != 0:
        raise ValueError(
            f"num_query_items ({params['num_query_items']}) must be divisible by num_q_heads ({params['num_q_heads']})"
        )

    # Derived parameters
    num_tokens = params["num_query_items"] // params["num_q_heads"]
    num_logical_pages_per_seq = (params["seq_len"] + params["tokens_per_page"] - 1) // params["tokens_per_page"]
    num_total_physical_pages = params["num_sequences_in_batch"] * num_logical_pages_per_seq

    # Create input tensors
    # Queries: [num_tokens, num_q_heads, head_dim]
    queries = mx.random.normal((num_tokens, params["num_q_heads"], params["head_dim"]), dtype=params["dtype"])

    # K/V cache pools: [num_total_physical_pages, tokens_per_page, num_kv_heads, head_dim]
    k_cache_pool = mx.random.normal(
        (num_total_physical_pages, params["tokens_per_page"], params["num_kv_heads"], params["head_dim"]),
        dtype=params["dtype"],
    )

    v_cache_pool = mx.random.normal(
        (num_total_physical_pages, params["tokens_per_page"], params["num_kv_heads"], params["head_dim"]),
        dtype=params["dtype"],
    )

    # Page table: [num_sequences_in_batch, num_logical_pages_per_seq]
    # Each sequence maps to a unique contiguous block of physical pages
    page_table_list = []
    for b_idx in range(params["num_sequences_in_batch"]):
        sequence_physical_page_indices = [
            b_idx * num_logical_pages_per_seq + l_idx for l_idx in range(num_logical_pages_per_seq)
        ]
        page_table_list.append(sequence_physical_page_indices)
    page_table = mx.array(page_table_list, dtype=mx.uint32)

    # Sequence lengths: [num_sequences_in_batch]
    sequence_lengths = mx.array([params["seq_len"]] * params["num_sequences_in_batch"], dtype=mx.int32)

    # Query to sequence map: [num_tokens]
    if params["num_sequences_in_batch"] == 1:
        query_to_seq_map = mx.zeros(num_tokens, dtype=mx.int32)
    else:
        # For the batched prefill scenario, where each sequence has the same length
        if num_tokens % params["num_sequences_in_batch"] != 0:
            raise ValueError(
                f"For multi-sequence batches, num_tokens ({num_tokens}) must be divisible by "
                f"num_sequences_in_batch ({params['num_sequences_in_batch']})"
            )

        tokens_per_seq_in_map = num_tokens // params["num_sequences_in_batch"]

        # For prefill scenario, each sequence gets its own block of tokens
        # If seq_len is specified and equals tokens_per_seq_in_map, we're in a prefill scenario
        if tokens_per_seq_in_map == params["seq_len"]:
            # Example: if num_sequences_in_batch=4, seq_len=2048,
            # query_to_seq_map should be [0...0 (2048 times), 1...1 (2048 times), ..., 3...3 (2048 times)]
            map_list = []
            for i in range(params["num_sequences_in_batch"]):
                map_list.extend([i] * params["seq_len"])
            query_to_seq_map = mx.array(map_list, dtype=mx.int32)
        else:
            # Default behavior for non-prefill scenarios
            query_to_seq_map = mx.repeat(
                mx.arange(params["num_sequences_in_batch"], dtype=mx.int32), repeats=tokens_per_seq_in_map
            )

    # Query token offset: [num_tokens]
    # For prefill scenario with batch_size > 1, use position offsets for each sequence
    if params["num_sequences_in_batch"] > 1 and num_tokens // params["num_sequences_in_batch"] == params["seq_len"]:
        # Example: if num_sequences_in_batch=4, seq_len=2048
        # query_token_offset should be [1,2,...,2048, 1,2,...,2048, ..., 1,2,...,2048]
        single_sequence_offsets = mx.arange(1, params["seq_len"] + 1, dtype=mx.int32)
        query_token_offset = mx.concatenate([single_sequence_offsets] * params["num_sequences_in_batch"])
    else:
        # Default behavior for non-prefill scenarios - all tokens attend to full history
        query_token_offset = mx.array([params["seq_len"]] * num_tokens, dtype=mx.int32)

    # Evaluate all tensors before returning to ensure they're computed
    mx.eval(queries)
    mx.eval(k_cache_pool)
    mx.eval(v_cache_pool)
    mx.eval(page_table)
    mx.eval(sequence_lengths)
    mx.eval(query_to_seq_map)
    mx.eval(query_token_offset)

    return (
        queries,
        k_cache_pool,
        v_cache_pool,
        page_table,
        sequence_lengths,
        query_to_seq_map,
        query_token_offset,
    )


def test_cold_start(benchmark):
    """
    Warmup the paged_attention kernel.
    """
    logger.info("Attempting to warm up the paged_attention kernel...")
    try:
        params = BASELINE_CONFIG.copy()
        input_tensors = setup_pal_benchmark_inputs(params)

        _ = benchmark(lambda: paged_attention(*input_tensors))

        logger.info("Paged_attention kernel warm-up sequence concluded successfully.")
    except Exception as e:
        logger.error(
            f"An issue occurred during paged_attention kernel warm-up: {e}",
            exc_info=True,
        )
        pytest.fail(f"Paged_attention kernel warm-up failed: {e}")


@pytest.mark.parametrize("seq_len_val", [64, 128, 256, 512, 1024, 2048])
def test_pal_latency_vs_seq_len(benchmark, seq_len_val):
    """
    Benchmark paged_attention operation performance across different sequence lengths.

    This test uses a fixed batch size (COMPARISON_BATCH_SIZE) to measure how latency
    scales with sequence length for a consistent number of parallel sequences.
    Both PAL and SDPA benchmarks will use the same fixed batch size for direct comparison.

    Args:
        benchmark: pytest-benchmark fixture for performance measurement
        seq_len_val: sequence length value to test
    """
    # Create test parameters from baseline with specified sequence length
    params = BASELINE_CONFIG.copy()
    params["seq_len"] = seq_len_val

    # Use fixed batch size with varying sequence length
    # Each sequence has seq_len_val tokens, so the total tokens is batch_size * seq_len_val
    params["num_sequences_in_batch"] = COMPARISON_BATCH_SIZE
    params["num_query_items"] = COMPARISON_BATCH_SIZE * seq_len_val * params["num_q_heads"]

    # Setup input tensors (evaluated during setup)
    input_tensors = setup_pal_benchmark_inputs(params)

    # Store the complete parameters for the analyzer
    if hasattr(benchmark, "extra_info"):
        benchmark.extra_info["run_params"] = params  # Store exact params used for this run
    else:
        # Fallback if extra_info is not available in some pytest-benchmark versions
        logger.warning("benchmark.extra_info not available. Parameter logging for analyzer might be limited.")

    # Define benchmark function that evaluates the result
    def operation_to_benchmark():
        out = paged_attention(*input_tensors)
        mx.eval(out)
        return out

    # Run benchmark
    result = benchmark(operation_to_benchmark)

    # Assert the output has expected shape and valid values
    queries = input_tensors[0]
    num_tokens = queries.shape[0]
    num_q_heads = queries.shape[1]
    expected_items = num_tokens * num_q_heads
    expected_shape = (expected_items, params["head_dim"])

    assert result.shape == expected_shape
    assert mx.isfinite(result).all()


@pytest.mark.parametrize("seq_len_val", [64, 128, 256, 512, 1024, 2048])
def test_sdpa_latency_vs_seq_len(benchmark, seq_len_val):
    """
    Benchmark MLX scaled_dot_product_attention operation performance across different sequence lengths.

    This test uses a fixed batch size (COMPARISON_BATCH_SIZE) to measure how latency
    scales with sequence length for a consistent number of parallel sequences.
    Both PAL and SDPA benchmarks use the same fixed batch size for direct comparison.

    Args:
        benchmark: pytest-benchmark fixture for performance measurement
        seq_len_val: sequence length value to test
    """
    # Create parameters with the fixed batch size and specified sequence length
    params = {
        "batch_size": COMPARISON_BATCH_SIZE,
        "num_q_heads": BASELINE_CONFIG["num_q_heads"],
        "num_kv_heads": BASELINE_CONFIG["num_kv_heads"],
        "head_dim": BASELINE_CONFIG["head_dim"],
        "seq_len": seq_len_val,
        "dtype": BASELINE_CONFIG["dtype"],
    }

    # Setup input tensors (evaluated during setup)
    q, k, v, scale, mask = setup_sdpa_benchmark_inputs(params)

    # Define benchmark function that evaluates the result
    def operation_to_benchmark():
        output = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)
        mx.eval(output)
        return output

    # Run benchmark
    result = benchmark(operation_to_benchmark)

    # Assert the output has expected shape and valid values
    expected_shape = (params["batch_size"], params["num_q_heads"], params["seq_len"], params["head_dim"])
    assert result.shape == expected_shape
    assert mx.isfinite(result).all()
