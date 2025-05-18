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

from proxy_attention_lab import paged_attention

logger = logging.getLogger(__name__)

# Define baseline configuration for benchmarks
BASELINE_CONFIG = {
    "batch_size": 1,
    "seq_len": 2048,  # tokens
    "num_q_heads": 32,
    "num_kv_heads": 16,
    "head_dim": 128,
    "tokens_per_page": 64,
    "dtype": mx.float16,
}


@pytest.mark.parametrize("seq_len_val", [64, 128, 256, 512, 1024, 2048, 4096, 8192])
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
    tokens_per_page = params["tokens_per_page"]
    num_q_heads = params["num_q_heads"]
    num_kv_heads = params["num_kv_heads"]
    head_dim = params["head_dim"]
    dtype = params["dtype"]
    batch_size = params["batch_size"]
    seq_len = params["seq_len"]

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
            queries, k_cache_pool, v_cache_pool, page_table, sequence_lengths, query_to_seq_map, query_token_offset
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


@pytest.mark.parametrize("seq_len_val", [64, 128, 256, 512, 1024, 2048, 4096, 8192])
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

    # Setup input tensors (evaluated during setup)
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

    mx.eval(queries)
    mx.eval(keys)
    mx.eval(values)
    mx.eval(causal_mask)

    # Define benchmark function that evaluates the result
    def operation_to_benchmark():
        output = mx.fast.scaled_dot_product_attention(
            queries,
            keys,
            values,
            scale=scale,
            mask=causal_mask,
        )
        mx.eval(output)
        return output

    # Run benchmark
    result = benchmark(operation_to_benchmark)

    # Assert the output has expected shape and valid values
    expected_shape = (params["batch_size"], params["num_q_heads"], params["seq_len"], params["head_dim"])
    assert result.shape == expected_shape
    assert mx.isfinite(result).all()
