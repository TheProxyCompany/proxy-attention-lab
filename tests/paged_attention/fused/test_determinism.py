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


@pytest.mark.parametrize("history_length", [16, 128, 1024, 4096])
@pytest.mark.parametrize("num_queries_tokens", [4, 16])
@pytest.mark.parametrize("tokens_per_page", [16, 64])
def test_paged_attention_determinism(history_length, num_queries_tokens, tokens_per_page) -> None:
    """Test that paged_attention output is deterministic for identical inputs.

    This test configures a moderately complex scenario, calls paged_attention twice
    with the same inputs, and asserts that the resulting output arrays are
    identical.
    """
    # --- Configuration ---
    num_q_heads = 32
    num_kv_heads = 16
    head_dim = 128
    num_total_pages = history_length // tokens_per_page
    max_logical_blocks_per_seq = (tokens_per_page * 2) // tokens_per_page  # e.g., 2 blocks

    # Seed for reproducibility of input data generation
    mx.random.seed(11)

    # --- Setup Test Inputs (Identical for both calls) ---
    # 1. Queries: 3D [NumQueryTokens, NumQHeads, HeadDim]
    # num_queries_tokens here means the first dimension of the Q array
    # Total items dispatched will be num_queries_tokens * num_q_heads
    queries_shape = (num_queries_tokens, num_q_heads, head_dim)
    # Using fixed random seed to ensure consistent inputs for determinism test
    py_queries = mx.random.normal(queries_shape, dtype=mx.float16)

    # 2. K/V Cache Pools
    kv_cache_shape = (num_total_pages, tokens_per_page, num_kv_heads, head_dim)
    py_k_cache_pool = mx.random.normal(kv_cache_shape, dtype=mx.float16)
    py_v_cache_pool = mx.random.normal(kv_cache_shape, dtype=mx.float16)

    # 3. Page Table: [NumSequencesInBatch, MaxLogicalBlocksPerSeq]
    # Ensure page IDs are valid (0 to num_total_pages - 1)
    py_page_table = mx.random.randint(0, num_total_pages, [1, max_logical_blocks_per_seq], dtype=mx.uint32)

    # 4. Sequence Lengths: [NumSequencesInBatch]
    # Ensure lengths are within reasonable bounds (e.g., up to max_logical_blocks_per_seq * tokens_per_page)
    max_seq_len_possible = max_logical_blocks_per_seq * tokens_per_page
    py_sequence_lengths = mx.random.randint(1, max_seq_len_possible + 1, [1], dtype=mx.int32)

    # 5. Query to Sequence Map: [NumQueryTokens] (mapping each of the first dim of Q to a sequence)
    # Values from 0 to num_sequences_in_batch - 1
    py_query_to_seq_map = mx.random.randint(
        0,
        1,
        [num_queries_tokens],  # Matches the first dimension of 3D queries
        dtype=mx.int32,
    )

    # 6. Query Token Offset: [NumQueryTokens]
    _query_token_offset_list = [0] * num_queries_tokens
    for i in range(num_queries_tokens):
        # Need to access scalar values from mx.array for Python's random.randint
        seq_idx = py_query_to_seq_map[i].item()
        max_offset = py_sequence_lengths[seq_idx].item()
        if max_offset > 0:
            _query_token_offset_list[i] = mx.random.randint(0, max_offset, []).item()
        else:  # Should not happen based on sequence_lengths generation, but defensive
            _query_token_offset_list[i] = 0
    py_query_token_offset = mx.array(_query_token_offset_list, dtype=mx.int32)

    # --- Call paged_attention the first time ---
    logger.info(f"Test: {test_paged_attention_determinism.__name__}")
    logger.info("  First call to paged_attention...")
    mx.eval(py_queries)
    mx.eval(py_k_cache_pool)
    mx.eval(py_v_cache_pool)
    mx.eval(py_page_table)
    mx.eval(py_sequence_lengths)
    mx.eval(py_query_to_seq_map)
    mx.eval(py_query_token_offset)

    output1 = paged_attention(
        py_queries,
        py_k_cache_pool,
        py_v_cache_pool,
        py_page_table,
        py_sequence_lengths,
        py_query_to_seq_map,
        py_query_token_offset,
        use_fused_kernel=True,
    )
    mx.eval(output1)  # Ensure computation is done

    # --- Call paged_attention the second time with identical inputs ---
    logger.info("  Second call to paged_attention with identical inputs...")
    output2 = paged_attention(
        py_queries,
        py_k_cache_pool,
        py_v_cache_pool,
        py_page_table,
        py_sequence_lengths,
        py_query_to_seq_map,
        py_query_token_offset,
        use_fused_kernel=True,
    )
    mx.eval(output2)  # Ensure computation is done

    # --- Assertions ---
    assert output1.shape == output2.shape, f"Output shapes differ: {output1.shape} vs {output2.shape}"
    assert output1.dtype == output2.dtype, f"Output dtypes differ: {output1.dtype} vs {output2.dtype}"

    # For debugging, print if they are not equal
    if not mx.array_equal(output1, output2).item():
        logger.error("Non-deterministic output detected!")
        logger.error(f"Output 1 sample: {output1[0, : min(output1.shape[1], 4)] if output1.size > 0 else 'empty'}")
        logger.error(f"Output 2 sample: {output2[0, : min(output2.shape[1], 4)] if output2.size > 0 else 'empty'}")
        mean_diff = mx.mean(mx.abs(output1 - output2)).item()
        logger.error(f"Mean difference: {mean_diff:.3f}")
        max_diff = mx.max(mx.abs(output1 - output2)).item()
        logger.error(f"Max difference: {max_diff:.3f}")

    assert mx.array_equal(output1, output2).item(), (
        "Paged attention prefill output is not deterministic. Outputs differ between two identical calls."
    )

    logger.info("  Result: Outputs are identical - determinism verified.")
