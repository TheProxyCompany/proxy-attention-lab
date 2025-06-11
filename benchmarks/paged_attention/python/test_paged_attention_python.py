# Copyright 2025 The Proxy Company. All Rights Reserved.
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
import pytest

from proxy_attention_lab import get_optimal_page_size, paged_attention

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
    k_cache_pool = mx.random.normal((num_total_physical_pages, num_kv_heads, tokens_per_page, head_dim), dtype=dtype)
    v_cache_pool = mx.random.normal((num_total_physical_pages, num_kv_heads, tokens_per_page, head_dim), dtype=dtype)

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

    queries = mx.contiguous(queries)
    k_cache_pool = mx.contiguous(k_cache_pool)
    v_cache_pool = mx.contiguous(v_cache_pool)
    page_table = mx.contiguous(page_table)
    sequence_lengths = mx.contiguous(sequence_lengths)

    mx.eval(queries)
    mx.eval(k_cache_pool)
    mx.eval(v_cache_pool)
    mx.eval(page_table)
    mx.eval(sequence_lengths)

    return queries, k_cache_pool, v_cache_pool, page_table, sequence_lengths


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
    return queries, keys, values, scale, causal_mask


# benchmarked up to 1048576 tokens
@pytest.mark.parametrize(
    "history_len_val", [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576]
)
@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
def test_pal_decode_latency_vs_history_len(benchmark, history_len_val, dtype):
    """
    Benchmark paged_attention decode operation performance across different history lengths.

    This test measures the latency of generating a single new token while attending to
    a varying amount of history in the KV cache. It uses a fixed batch size to measure
    how decode latency scales with history length.

    Args:
        benchmark: pytest-benchmark fixture for performance measurement
        history_len_val: history length value to test (size of existing KV cache)
    """
    mx.clear_cache()
    # Create test parameters for decode phase
    params = BASELINE_CONFIG.copy()
    params["history_len"] = history_len_val
    params["dtype"] = dtype
    params["tokens_per_page"] = get_optimal_page_size()

    # Setup decode inputs
    queries, k_hist, v_hist, pt, slens_hist = setup_pal_decode_inputs(params)

    # Define benchmark function that evaluates the result
    def operation_to_benchmark():
        out = paged_attention(queries, k_hist, v_hist, pt, slens_hist)
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


# benchmarked up to 1048576 tokens
@pytest.mark.parametrize(
    "history_len_val", [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576]
)
@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
def test_mlx_decode_latency_vs_history_len(benchmark, history_len_val, dtype):
    """
    Benchmark MLX scaled_dot_product_attention decode operation performance across different history lengths.

    This test measures the latency of generating a single new token while attending to
    a varying amount of history in the KV cache. It uses a fixed batch size to measure
    how decode latency scales with history length.

    Args:
        benchmark: pytest-benchmark fixture for performance measurement
        history_len_val: history length value to test (size of existing KV cache)
    """
    mx.clear_cache()
    # Create test parameters for decode phase
    params = BASELINE_CONFIG.copy()
    params["history_len"] = history_len_val
    params["dtype"] = dtype
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
