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

import logging

import mlx.core as mx
import mlx.nn
import pytest

from proxy_attention_lab import calculate_page_size, paged_attention

logger = logging.getLogger(__name__)

# Gemma 3 Model Config, 2048 tokens
BASELINE_CONFIG = {
    "batch_size": 1,
    "seq_len": 2048,  # tokens
    "num_q_heads": 32,
    "num_kv_heads": 16,
    "head_dim": 128,
    "tokens_per_page": 16,
    "dtype": mx.float16,
}


@pytest.mark.parametrize("seq_len_val", [64, 128, 256, 512, 1024, 2048, 4096])
def test_pal_latency_vs_seq_len(benchmark, seq_len_val):
    """
    Benchmark paged_attention operation performance across different sequence lengths.

    This test uses a fixed batch size (COMPARISON_BATCH_SIZE) to measure how latency
    scales with sequence length for a consistent number of parallel sequences.
    Both PAL and MLX benchmarks will use the same fixed batch size for direct comparison.

    Args:
        benchmark: pytest-benchmark fixture for performance measurement
        seq_len_val: sequence length value to test
    """
    # Create test parameters from baseline with specified sequence length
    params = BASELINE_CONFIG.copy()
    params["seq_len"] = seq_len_val
    num_q_heads = params["num_q_heads"]
    num_kv_heads = params["num_kv_heads"]
    head_dim = params["head_dim"]
    dtype = params["dtype"]
    batch_size = params["batch_size"]
    seq_len = params["seq_len"]
    tokens_per_page = calculate_page_size(head_dim, num_q_heads, num_kv_heads)

    # Setup input tensors
    num_tokens = batch_size * seq_len
    num_logical_pages_per_seq = (seq_len + tokens_per_page - 1) // tokens_per_page
    num_total_physical_pages = batch_size * num_logical_pages_per_seq

    queries = mx.random.normal((num_tokens, num_q_heads, head_dim), dtype=dtype)
    k_cache_pool = mx.random.normal((num_total_physical_pages, tokens_per_page, num_kv_heads, head_dim), dtype=dtype)
    v_cache_pool = mx.random.normal((num_total_physical_pages, tokens_per_page, num_kv_heads, head_dim), dtype=dtype)
    # Create page table mapping
    page_table_list = []
    for b_idx in range(batch_size):
        sequence_physical_page_indices = [
            b_idx * num_logical_pages_per_seq + l_idx for l_idx in range(num_logical_pages_per_seq)
        ]
        page_table_list.append(sequence_physical_page_indices)
    page_table = mx.array(page_table_list, dtype=mx.uint32)

    # Set sequence length for each batch item
    sequence_lengths = mx.array([seq_len] * batch_size, dtype=mx.int32)

    # query_to_seq_map: maps each token in pal_queries to its sequence index
    # pal_queries has tokens ordered as [seq0_tokens, seq1_tokens, ...]
    query_to_seq_map = mx.repeat(mx.arange(batch_size, dtype=mx.int32), repeats=seq_len)

    # query_token_offset: for causal attention, 1-indexed position within the sequence
    # Offsets are [1, 2, ..., SL, 1, 2, ..., SL, ...]
    query_token_offset = mx.tile(mx.arange(1, seq_len + 1, dtype=mx.int32), batch_size)

    queries = mx.contiguous(queries)
    k_cache_pool = mx.contiguous(k_cache_pool)
    v_cache_pool = mx.contiguous(v_cache_pool)
    page_table = mx.contiguous(page_table)
    sequence_lengths = mx.contiguous(sequence_lengths)
    query_to_seq_map = mx.contiguous(query_to_seq_map)
    query_token_offset = mx.contiguous(query_token_offset)

    mx.eval(queries)
    mx.eval(k_cache_pool)
    mx.eval(v_cache_pool)
    mx.eval(page_table)
    mx.eval(sequence_lengths)
    mx.eval(query_to_seq_map)
    mx.eval(query_token_offset)

    # Define benchmark function that evaluates the result
    def operation_to_benchmark():
        out = paged_attention(
            queries,
            k_cache_pool,
            v_cache_pool,
            page_table,
            sequence_lengths,
            query_to_seq_map,
            query_token_offset,
            use_fused_kernel=False,
        )
        mx.eval(out)
        return out

    # Run benchmark
    result = benchmark(operation_to_benchmark)

    # Assert the output has expected shape and valid values
    num_tokens = queries.shape[0]
    num_q_heads = queries.shape[1]
    expected_shape = (num_tokens * num_q_heads, head_dim)

    assert result.shape == expected_shape
    assert mx.isfinite(result).all()


@pytest.mark.parametrize("seq_len_val", [64, 128, 256, 512, 1024, 2048, 4096])
def test_mlx_latency_vs_seq_len(benchmark, seq_len_val):
    """
    Benchmark MLX scaled_dot_product_attention operation performance across different sequence lengths.

    This test uses a fixed batch size (COMPARISON_BATCH_SIZE) to measure how latency
    scales with sequence length for a consistent number of parallel sequences.
    Both PAL and MLX benchmarks use the same fixed batch size for direct comparison.

    Args:
        benchmark: pytest-benchmark fixture for performance measurement
        seq_len_val: sequence length value to test
    """
    # Create parameters with the fixed batch size and specified sequence length
    params = BASELINE_CONFIG.copy()
    params["seq_len"] = seq_len_val

    # Add benchmark metadata if supported
    if hasattr(benchmark, "extra_info"):
        benchmark.extra_info["run_params"] = params.copy()

    # Setup input tensors (evaluated during setup)
    scale = 1.0 / mx.sqrt(float(params["head_dim"]))
    batch_size = params["batch_size"]
    num_q_heads = params["num_q_heads"]
    num_kv_heads = params["num_kv_heads"]
    head_dim = params["head_dim"]
    seq_len = params["seq_len"]
    dtype = params["dtype"]

    # Create input tensors - ensuring shapes scale with seq_len_val
    queries = mx.random.normal((batch_size, num_q_heads, seq_len, head_dim), dtype=dtype)

    keys = mx.random.normal((batch_size, num_kv_heads, seq_len, head_dim), dtype=dtype)

    values = mx.random.normal((batch_size, num_kv_heads, seq_len, head_dim), dtype=dtype)

    # Create causal mask that scales with sequence length
    causal_mask = mlx.nn.MultiHeadAttention.create_additive_causal_mask(seq_len).astype(dtype)

    # Log tensor shapes for verification
    logger.info(f"PREFILL SDPA - seq_len: {seq_len}")
    logger.info(f"  queries shape: {queries.shape}")
    logger.info(f"  keys shape: {keys.shape}")
    logger.info(f"  values shape: {values.shape}")
    logger.info(f"  causal_mask shape: {causal_mask.shape}")

    # Define benchmark function that evaluates the result
    def operation_to_benchmark():
        output = mx.fast.scaled_dot_product_attention(queries, keys, values, scale=scale, mask=causal_mask)
        mx.eval(output)
        return output

    # Run benchmark
    result = benchmark(operation_to_benchmark)

    # Assert the output has expected shape and valid values
    expected_shape = (batch_size, num_q_heads, seq_len, head_dim)
    assert result.shape == expected_shape
    assert mx.isfinite(result).all()


def setup_pal_decode_inputs(params):
    """
    Setup PAL decode benchmark inputs.

    This function prepares inputs for the decode phase where the model generates
    a single new token while attending to a history of previously computed tokens.

    Args:
        params: Dictionary of parameters including history_len, batch_size, etc.

    Returns:
        Tuple of tensors needed for PAL paged_attention decode operation.
    """
    tokens_per_page = params["tokens_per_page"]
    num_q_heads = params["num_q_heads"]
    num_kv_heads = params["num_kv_heads"]
    head_dim = params["head_dim"]
    dtype = params["dtype"]
    batch_size = params["batch_size"]
    history_len = params["history_len"]

    # For decode, queries has batch_size tokens (one new token per sequence)
    num_tokens = batch_size

    # Calculate page table sizes based on history
    num_logical_pages_per_seq = (history_len + tokens_per_page - 1) // tokens_per_page
    num_total_physical_pages = batch_size * num_logical_pages_per_seq

    # Create query tensor for the single new token per sequence
    queries = mx.random.normal((num_tokens, num_q_heads, head_dim), dtype=dtype)

    # Create KV cache pools sized for the entire history
    k_cache_pool = mx.random.normal((num_total_physical_pages, tokens_per_page, num_kv_heads, head_dim), dtype=dtype)
    v_cache_pool = mx.random.normal((num_total_physical_pages, tokens_per_page, num_kv_heads, head_dim), dtype=dtype)

    # Create page table mapping
    page_table_list = []
    for b_idx in range(batch_size):
        sequence_physical_page_indices = [
            b_idx * num_logical_pages_per_seq + l_idx for l_idx in range(num_logical_pages_per_seq)
        ]
        page_table_list.append(sequence_physical_page_indices)
    page_table = mx.array(page_table_list, dtype=mx.uint32)

    # Set sequence length for each batch item to the history length
    sequence_lengths = mx.array([history_len] * batch_size, dtype=mx.int32)

    # query_to_seq_map: maps each token in queries to its sequence index
    # For decode, this is just the sequence indices as we have one token per sequence
    query_to_seq_map = mx.arange(batch_size, dtype=mx.int32)

    # query_token_offset: position of the new token after the history
    # For decode, this is history_len + 1 for each token
    query_token_offset = mx.array([history_len + 1] * batch_size, dtype=mx.int32)

    queries = mx.contiguous(queries)
    k_cache_pool = mx.contiguous(k_cache_pool)
    v_cache_pool = mx.contiguous(v_cache_pool)
    page_table = mx.contiguous(page_table)
    sequence_lengths = mx.contiguous(sequence_lengths)
    query_to_seq_map = mx.contiguous(query_to_seq_map)
    query_token_offset = mx.contiguous(query_token_offset)

    mx.eval(queries)
    mx.eval(k_cache_pool)
    mx.eval(v_cache_pool)
    mx.eval(page_table)
    mx.eval(sequence_lengths)
    mx.eval(query_to_seq_map)
    mx.eval(query_token_offset)

    return queries, k_cache_pool, v_cache_pool, page_table, sequence_lengths, query_to_seq_map, query_token_offset


def setup_sdpa_decode_inputs(params):
    """
    Setup SDPA decode benchmark inputs.

    This function prepares inputs for the decode phase where the model generates
    a single new token while attending to a history of previously computed tokens.

    Args:
        params: Dictionary of parameters including history_len, batch_size, etc.

    Returns:
        Tuple of tensors needed for SDPA operation in decode mode.
    """
    scale = 1.0 / mx.sqrt(float(params["head_dim"]))
    batch_size = params["batch_size"]
    num_q_heads = params["num_q_heads"]
    num_kv_heads = params["num_kv_heads"]
    head_dim = params["head_dim"]
    history_len = params["history_len"]
    dtype = params["dtype"]

    # Create queries tensor for a single token per sequence
    # Shape: [batch_size, num_q_heads, 1, head_dim]
    queries = mx.random.normal((batch_size, num_q_heads, 1, head_dim), dtype=dtype)

    # Create keys and values tensors for the history
    # Shape: [batch_size, num_kv_heads, history_len, head_dim]
    keys = mx.random.normal((batch_size, num_kv_heads, history_len, head_dim), dtype=dtype)
    values = mx.random.normal((batch_size, num_kv_heads, history_len, head_dim), dtype=dtype)

    # Create a mask of zeros to allow full attention from query to all history
    # Shape: [1, history_len] - additive mask where zeros allow full attention
    causal_mask = mx.zeros((1, history_len), dtype=dtype)

    # Log the tensor shapes to verify scaling
    logger.info(f"DECODE SDPA - history_len: {history_len}")
    logger.info(f"  queries shape: {queries.shape}")
    logger.info(f"  keys shape: {keys.shape}")
    logger.info(f"  values shape: {values.shape}")
    logger.info(f"  mask shape: {None if causal_mask is None else causal_mask.shape}")

    return queries, keys, values, scale, causal_mask


@pytest.mark.parametrize("history_len_val", [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072])
def test_pal_decode_latency_vs_history_len(benchmark, history_len_val):
    """
    Benchmark paged_attention decode operation performance across different history lengths.

    This test measures the latency of generating a single new token while attending to
    a varying amount of history in the KV cache. It uses a fixed batch size to measure
    how decode latency scales with history length.

    Args:
        benchmark: pytest-benchmark fixture for performance measurement
        history_len_val: history length value to test (size of existing KV cache)
    """
    # Create test parameters for decode phase
    params = BASELINE_CONFIG.copy()
    params["history_len"] = history_len_val
    params["tokens_per_page"] = calculate_page_size(params["head_dim"], params["num_q_heads"], params["num_kv_heads"])

    # Calculate the number of query items for benchmarking info
    params["num_query_items"] = params["batch_size"] * params["num_q_heads"]

    # Add benchmark metadata if supported
    if hasattr(benchmark, "extra_info"):
        benchmark.extra_info["run_params"] = params.copy()

    # Setup decode inputs
    queries, k_hist, v_hist, pt, slens_hist, q_map, q_off = setup_pal_decode_inputs(params)

    # Define benchmark function that evaluates the result
    def operation_to_benchmark():
        out = paged_attention(queries, k_hist, v_hist, pt, slens_hist, q_map, q_off, use_fused_kernel=True)
        mx.eval(out)
        return out

    # Run benchmark
    result = benchmark(operation_to_benchmark)

    # Assert the output has expected shape and valid values
    num_tokens = queries.shape[0]
    num_q_heads = queries.shape[1]
    head_dim = params["head_dim"]
    expected_shape = (num_tokens * num_q_heads, head_dim)

    assert result.shape == expected_shape
    assert mx.isfinite(result).all()


@pytest.mark.parametrize("history_len_val", [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072])
def test_mlx_decode_latency_vs_history_len(benchmark, history_len_val):
    """
    Benchmark MLX scaled_dot_product_attention decode operation performance across different history lengths.

    This test measures the latency of generating a single new token while attending to
    a varying amount of history in the KV cache. It uses a fixed batch size to measure
    how decode latency scales with history length.

    Args:
        benchmark: pytest-benchmark fixture for performance measurement
        history_len_val: history length value to test (size of existing KV cache)
    """
    # Create test parameters for decode phase
    params = BASELINE_CONFIG.copy()
    params["history_len"] = history_len_val

    # Add benchmark metadata if supported
    if hasattr(benchmark, "extra_info"):
        benchmark.extra_info["run_params"] = params.copy()

    # Setup decode inputs
    queries, keys, values, scale, causal_mask = setup_sdpa_decode_inputs(params)

    # Define benchmark function that evaluates the result
    def operation_to_benchmark():
        output = mx.fast.scaled_dot_product_attention(queries, keys, values, scale=scale, mask=causal_mask)
        mx.eval(output)
        return output

    # Run benchmark
    result = benchmark(operation_to_benchmark)

    # Assert the output has expected shape and valid values
    expected_shape = (params["batch_size"], params["num_q_heads"], 1, params["head_dim"])
    assert result.shape == expected_shape
    assert mx.isfinite(result).all()
