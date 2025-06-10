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
"""Determinism tests for the paged attention operation.

This module verifies that repeated calls to the paged_attention kernel with
identical inputs produce bit-for-bit identical outputs.
"""

import logging

import mlx.core as mx
import pytest

from proxy_attention_lab import paged_attention

logger = logging.getLogger(__name__)


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
@pytest.mark.parametrize("history_length", [16, 128, 1024, 4096])
@pytest.mark.parametrize("batch_size", [1])  # problem with multiple sequences
@pytest.mark.parametrize("head_dim", [32, 64, 128, 256])
def test_paged_attention_determinism(history_length, batch_size, head_dim, dtype) -> None:
    """Test that paged_attention output is deterministic for identical inputs.

    This test configures a moderately complex scenario, calls paged_attention twice
    with the same inputs, and asserts that the resulting output arrays are
    identical.
    """
    # --- Configuration ---
    num_q_heads = 32
    num_kv_heads = 16
    tokens_per_page = 16
    num_total_pages = history_length // tokens_per_page
    max_logical_pages_per_seq = (tokens_per_page * 2) // tokens_per_page  # e.g., 2 blocks

    # Seed for reproducibility of input data generation
    mx.random.seed(11)

    # --- Setup Test Inputs (Identical for both calls) ---
    # 1. Queries: 3D [Sequences, NumQHeads, HeadDim]
    queries_shape = (batch_size, num_q_heads, head_dim)
    py_queries = mx.random.normal(queries_shape, dtype=dtype)

    # 2. K/V Cache Pools
    kv_cache_shape = (num_total_pages, num_kv_heads, tokens_per_page, head_dim)
    py_k_cache_pool = mx.random.normal(kv_cache_shape, dtype=dtype)
    py_v_cache_pool = mx.random.normal(kv_cache_shape, dtype=dtype)

    # 3. Page Table: [NumSequencesInBatch, MaxLogicalBlocksPerSeq]
    py_page_table = mx.random.randint(0, num_total_pages, [batch_size, max_logical_pages_per_seq], dtype=mx.uint32)

    # 4. Sequence Lengths: [NumSequencesInBatch]
    max_seq_len_possible = max_logical_pages_per_seq * tokens_per_page
    py_sequence_lengths = mx.random.randint(1, max_seq_len_possible + 1, [batch_size], dtype=mx.int32)

    # --- Call paged_attention the first time ---
    logger.info(f"Test: {test_paged_attention_determinism.__name__} (dtype={dtype})")
    logger.info("  First call to paged_attention...")
    mx.eval(py_queries)
    mx.eval(py_k_cache_pool)
    mx.eval(py_v_cache_pool)
    mx.eval(py_page_table)
    mx.eval(py_sequence_lengths)

    output1 = paged_attention(
        py_queries,
        py_k_cache_pool,
        py_v_cache_pool,
        py_page_table,
        py_sequence_lengths,
    )
    mx.eval(output1)

    # --- Call paged_attention the second time with identical inputs ---
    logger.info("  Second call to paged_attention with identical inputs...")
    output2 = paged_attention(
        py_queries,
        py_k_cache_pool,
        py_v_cache_pool,
        py_page_table,
        py_sequence_lengths,
    )
    mx.eval(output2)

    # --- Assertions ---
    assert output1.shape == output2.shape, f"Output shapes differ: {output1.shape} vs {output2.shape}"
    assert output1.dtype == output2.dtype, f"Output dtypes differ: {output1.dtype} vs {output2.dtype}"

    if not mx.array_equal(output1, output2).item():
        logger.error("Non-deterministic output detected!")
        logger.error(f"Output 1 sample: {output1[0, : min(output1.shape[1], 4)] if output1.size > 0 else 'empty'}")
        logger.error(f"Output 2 sample: {output2[0, : min(output2.shape[1], 4)] if output2.size > 0 else 'empty'}")
        mean_diff = mx.mean(mx.abs(output1 - output2)).item()
        logger.error(f"Mean difference: {mean_diff:.3f}")
        max_diff = mx.max(mx.abs(output1 - output2)).item()
        logger.error(f"Max difference: {max_diff:.3f}")

    assert mx.array_equal(output1, output2).item(), (
        "Paged attention fused output is not deterministic. Outputs differ between two identical calls."
    )

    logger.info("  Result: Outputs are identical - determinism verified.")
