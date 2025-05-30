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

from proxy_attention_lab import calculate_page_size, paged_attention

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

    NOTE: Refactored to align with prefill input contract where:
    - queries must contain ALL query tokens for sequences being prefilled
    - TotalNumQueryTokensInBatch MUST equal sum(sequence_lengths)
    - query_to_seq_map and query_token_offset must match the contiguous query layout
    """
    logger.info(f"Test: {test_paged_attention_smoke.__name__}")

    # Query tensor parameters
    num_q_heads = 2
    num_kv_heads = 2
    head_dim = 8

    # Calculate tokens_per_page using the kernel's expected page size
    tokens_per_page = calculate_page_size(head_dim, num_q_heads, num_kv_heads)

    # Batch parameters
    num_sequences_in_batch = 1  # For this smoke test, keep it simple with one sequence

    # For prefill, we need to provide all query tokens for the sequence
    # Let's prefill 4 tokens for our single sequence
    sequence_lengths_list = [4]
    total_num_query_tokens = sum(sequence_lengths_list)  # = 4

    # KV cache parameters
    num_total_pages = 2
    max_logical_blocks_per_seq_val = 1  # Sequence uses at most 1 logical block

    logger.info("  Test Configuration:")
    logger.info(f"    Queries: {total_num_query_tokens} total query tokens, {num_q_heads} heads, {head_dim} dimensions")
    logger.info(f"    KV cache: {num_total_pages} pages, {tokens_per_page} tokens per page, {num_kv_heads} KV heads")
    logger.info(f"    Batch: {num_sequences_in_batch} sequences, sequence_lengths={sequence_lengths_list}")

    # Create test inputs with random values
    # For prefill, queries shape is [TotalNumQueryTokensInBatch, NumQHeads, HeadDim]
    mock_queries = mx.random.normal((total_num_query_tokens, num_q_heads, head_dim)).astype(mx.float16)
    mock_k_cache_pool = mx.random.normal((num_total_pages, tokens_per_page, num_kv_heads, head_dim)).astype(mx.float16)
    mock_v_cache_pool = mx.random.normal((num_total_pages, tokens_per_page, num_kv_heads, head_dim)).astype(mx.float16)

    # Create page table: maps logical blocks to physical pages
    # Shape: [num_sequences_in_batch, max_logical_blocks_per_seq_val]
    # For this smoke test, just point to physical page 0
    page_table_content = [[0] * max_logical_blocks_per_seq_val for _ in range(num_sequences_in_batch)]
    mock_page_table = mx.array(page_table_content, dtype=mx.uint32)

    # Metadata arrays aligned with prefill contract
    mock_sequence_lengths = mx.array(sequence_lengths_list, dtype=mx.int32)

    # query_to_seq_map must map each query token to its sequence
    # For one sequence with 4 tokens: [0, 0, 0, 0]
    mock_query_to_seq_map = mx.zeros(total_num_query_tokens, dtype=mx.int32)

    # query_token_offset: position of each query within its sequence
    # For prefill from start: [0, 1, 2, 3]
    mock_query_token_offset = mx.arange(total_num_query_tokens, dtype=mx.int32)

    logger.info("  Input Shapes:")
    logger.info(f"    Queries: {mock_queries.shape}")
    logger.info(f"    K cache: {mock_k_cache_pool.shape}")
    logger.info(f"    V cache: {mock_v_cache_pool.shape}")
    logger.info(f"    Page table: {mock_page_table.shape}")
    logger.info(f"    Sequence lengths: {mock_sequence_lengths.shape}")
    logger.info(f"    Query token offsets: {mock_query_token_offset}")

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
            is_prefill=True,
        )
        mx.eval(out)

        # For 3D queries [NumTokens, NumQHeads, HeadDim],
        # output is [NumTokens * NumQHeads, HeadDim] (2D layout)
        total_items = total_num_query_tokens * num_q_heads
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
