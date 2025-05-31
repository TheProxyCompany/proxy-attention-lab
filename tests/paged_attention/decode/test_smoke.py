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
"""Basic smoke test for paged attention functionality."""

import logging

import mlx.core as mx

from proxy_attention_lab import paged_attention

logger = logging.getLogger(__name__)


def test_paged_attention_smoke() -> None:
    """Verify that the paged_attention function runs without errors on simple inputs.

    This test creates minimal inputs to check that the function executes successfully
    and produces an output with the expected shape and type. The test serves as a basic
    sanity check for the attention mechanism, verifying:

    1. The function can be called with correctly shaped random inputs
    2. The function produces an output without crashing
    3. The output has the expected dimensions based on input parameters
    4. The output contains valid numerical values (no NaN or Inf)
    """
    logger.info(f"Test: {test_paged_attention_smoke.__name__}")

    # Query tensor parameters
    num_queries = 4
    num_q_heads = 2
    head_dim = 8

    # KV cache parameters
    num_total_pages = 2
    tokens_per_page = 64
    num_kv_heads = 2

    # Batch parameters
    num_sequences_in_batch = 1  # For this smoke test, keep it simple with one sequence
    max_logical_blocks_per_seq_val = 1  # Sequence uses at most 1 logical block

    logger.info("  Test Configuration:")
    logger.info(f"    Queries: {num_queries} queries, {num_q_heads} heads, {head_dim} dimensions")
    logger.info(f"    KV cache: {num_total_pages} pages, {tokens_per_page} tokens per page, {num_kv_heads} KV heads")
    logger.info(f"    Batch: {num_sequences_in_batch} sequences, {max_logical_blocks_per_seq_val} blocks per sequence")

    # Create test inputs with random values
    mock_queries = mx.random.normal((num_queries, num_q_heads, head_dim)).astype(mx.float16)
    mock_k_cache_pool = mx.random.normal((num_total_pages, tokens_per_page, num_kv_heads, head_dim)).astype(mx.float16)
    mock_v_cache_pool = mx.random.normal((num_total_pages, tokens_per_page, num_kv_heads, head_dim)).astype(mx.float16)

    # Create page table: maps logical blocks to physical pages
    # Shape: [num_sequences_in_batch, max_logical_blocks_per_seq_val]
    # For this smoke test, just point to physical page 0
    page_table_content = [[0] * max_logical_blocks_per_seq_val for _ in range(num_sequences_in_batch)]
    mock_page_table = mx.array(page_table_content, dtype=mx.uint32)

    # Metadata arrays
    mock_sequence_lengths = mx.array([tokens_per_page // 2] * num_sequences_in_batch, dtype=mx.int32)
    mock_query_to_seq_map = mx.zeros(num_queries, dtype=mx.int32)
    mock_query_token_offset = mx.arange(num_queries, dtype=mx.int32)

    logger.info("  Input Shapes:")
    logger.info(f"    Queries: {mock_queries.shape}")
    logger.info(f"    K cache: {mock_k_cache_pool.shape}")
    logger.info(f"    V cache: {mock_v_cache_pool.shape}")
    logger.info(f"    Page table: {mock_page_table.shape}")
    logger.info(f"    Sequence lengths: {mock_sequence_lengths.shape}")
    logger.info(f"    Query token offsets: {mock_query_token_offset}")

    mx.eval(mock_queries)
    mx.eval(mock_k_cache_pool)
    mx.eval(mock_v_cache_pool)
    mx.eval(mock_page_table)
    mx.eval(mock_sequence_lengths)
    mx.eval(mock_query_to_seq_map)
    mx.eval(mock_query_token_offset)

    try:
        # Run the paged attention operation
        out = paged_attention(
            mock_queries,
            mock_k_cache_pool,
            mock_v_cache_pool,
            mock_page_table,
            mock_sequence_lengths,
            mock_query_to_seq_map,
            mock_query_token_offset,
            is_prefill=False,
        )
        mx.eval(out)

        # For 3D queries [NumTokens, NumQHeads, HeadDim],
        # output is [NumTokens * NumQHeads, HeadDim] (2D layout)
        total_items = num_queries * num_q_heads
        expected_output_shape = (total_items, head_dim)

        logger.info("  Output Verification:")
        logger.info(f"    Expected shape: {expected_output_shape}")
        logger.info(f"    Actual shape: {out.shape}")
        logger.info(f"    Dtype: {out.dtype}")
        logger.info(f"    Contains finite values: {mx.isfinite(out).all()}")

        # Verify output properties
        assert out.shape == expected_output_shape, (
            f"Output shape {out.shape} does not match expected shape {expected_output_shape}"
        )
        assert out.dtype == mock_queries.dtype, (
            f"Output dtype {out.dtype} does not match query dtype {mock_queries.dtype}"
        )
        assert mx.isfinite(out).all(), "Output attention vectors contain NaN or Inf values"

    except Exception as e:
        logger.error(f"Paged attention smoke test failed: {e}", exc_info=True)
        raise
