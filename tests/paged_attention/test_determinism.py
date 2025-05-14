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
"""Determinism tests for the paged attention operation.

This module verifies that repeated calls to the paged_attention kernel with
identical inputs produce bit-for-bit identical outputs.
"""

import logging

import mlx.core as mx
import numpy as np

from proxy_attention_lab import paged_attention

logger = logging.getLogger(__name__)


def test_paged_attention_determinism() -> None:
    """Test that paged_attention output is deterministic for identical inputs.

    This test configures a moderately complex scenario, calls paged_attention twice
    with the same inputs, and asserts that the resulting output arrays are
    identical.
    """
    # --- Configuration ---
    # Use a configuration that exercises various aspects of the kernel
    num_queries_tokens = 4  # Corresponds to num_items_to_process if 2D, or num_tokens if 3D
    num_q_heads = 2
    head_dim = 64  # A common head dimension

    num_total_pages = 4
    tokens_per_page = 16  # Smaller to force more paging if seq_len is long
    num_kv_heads = 2  # MHA scenario

    num_sequences_in_batch = 2
    max_logical_blocks_per_seq = (tokens_per_page * 2) // tokens_per_page  # e.g., 2 blocks

    # Seed for reproducibility of input data generation
    mx.random.seed(42)

    # --- Setup Test Inputs (Identical for both calls) ---
    # 1. Queries: 3D [NumQueryTokens, NumQHeads, HeadDim]
    # num_queries_tokens here means the first dimension of the Q array
    # Total items dispatched will be num_queries_tokens * num_q_heads
    queries_shape = (num_queries_tokens, num_q_heads, head_dim)
    # Explicitly create numpy array first, then convert, to ensure identical initial data
    np_queries = np.random.normal(size=queries_shape).astype(np.float16)
    py_queries = mx.array(np_queries)

    # 2. K/V Cache Pools
    kv_cache_shape = (num_total_pages, tokens_per_page, num_kv_heads, head_dim)
    np_k_cache_pool = np.random.normal(size=kv_cache_shape).astype(np.float16)
    np_v_cache_pool = np.random.normal(size=kv_cache_shape).astype(np.float16)
    py_k_cache_pool = mx.array(np_k_cache_pool)
    py_v_cache_pool = mx.array(np_v_cache_pool)

    # 3. Page Table: [NumSequencesInBatch, MaxLogicalBlocksPerSeq]
    # Ensure page IDs are valid (0 to num_total_pages - 1)
    np_page_table = np.random.randint(
        0, num_total_pages, size=(num_sequences_in_batch, max_logical_blocks_per_seq)
    ).astype(np.uint32)
    py_page_table = mx.array(np_page_table)

    # 4. Sequence Lengths: [NumSequencesInBatch]
    # Ensure lengths are within reasonable bounds (e.g., up to max_logical_blocks_per_seq * tokens_per_page)
    max_seq_len_possible = max_logical_blocks_per_seq * tokens_per_page
    np_sequence_lengths = np.random.randint(1, max_seq_len_possible + 1, size=(num_sequences_in_batch,)).astype(
        np.int32
    )
    py_sequence_lengths = mx.array(np_sequence_lengths)

    # 5. Query to Sequence Map: [NumQueryTokens] (mapping each of the first dim of Q to a sequence)
    # Values from 0 to num_sequences_in_batch - 1
    np_query_to_seq_map = np.random.randint(
        0,
        num_sequences_in_batch,
        size=(num_queries_tokens,),  # Matches the first dimension of 3D queries
    ).astype(np.int32)
    py_query_to_seq_map = mx.array(np_query_to_seq_map)

    # 6. Query Token Offset: [NumQueryTokens]
    # Offset for each query token within its sequence
    # Ensure offsets are less than the corresponding sequence_lengths
    np_query_token_offset = np.zeros((num_queries_tokens,), dtype=np.int32)
    for i in range(num_queries_tokens):
        seq_idx = np_query_to_seq_map[i]
        np_query_token_offset[i] = np.random.randint(0, np_sequence_lengths[seq_idx])
    py_query_token_offset = mx.array(np_query_token_offset)

    # --- Call paged_attention the first time ---
    logger.info("Determinism Test: First call to paged_attention.")
    output1 = paged_attention(
        py_queries,
        py_k_cache_pool,
        py_v_cache_pool,
        py_page_table,
        py_sequence_lengths,
        py_query_to_seq_map,
        py_query_token_offset,
    )
    mx.eval(output1)  # Ensure computation is done

    # --- Call paged_attention the second time with identical inputs ---
    # Re-create from numpy arrays to ensure no aliasing or in-place modification issues
    # (though MLX arrays are usually immutable, this is an extra safeguard for test setup)
    py_queries_2 = mx.array(np_queries)
    py_k_cache_pool_2 = mx.array(np_k_cache_pool)
    py_v_cache_pool_2 = mx.array(np_v_cache_pool)
    py_page_table_2 = mx.array(np_page_table)
    py_sequence_lengths_2 = mx.array(np_sequence_lengths)
    py_query_to_seq_map_2 = mx.array(np_query_to_seq_map)
    py_query_token_offset_2 = mx.array(np_query_token_offset)

    logger.info("Determinism Test: Second call to paged_attention with identical inputs.")
    output2 = paged_attention(
        py_queries_2,
        py_k_cache_pool_2,
        py_v_cache_pool_2,
        py_page_table_2,
        py_sequence_lengths_2,
        py_query_to_seq_map_2,
        py_query_token_offset_2,
    )
    mx.eval(output2)  # Ensure computation is done

    # --- Assertions ---
    assert output1.shape == output2.shape, f"Output shapes differ: {output1.shape} vs {output2.shape}"
    assert output1.dtype == output2.dtype, f"Output dtypes differ: {output1.dtype} vs {output2.dtype}"

    # For bit-for-bit exactness, mx.array_equal is appropriate.
    # If there were any concerns about ultra-minor float variations due to non-associativity
    # on different runs (highly unlikely for SIMD reductions on same GPU),
    # mx.allclose would be a fallback, but array_equal is the goal here.
    assert mx.array_equal(output1, output2), (
        "Paged attention output is not deterministic. Outputs differ between two identical calls."
    )

    logger.info("test_paged_attention_determinism PASSED: Outputs are identical.")
