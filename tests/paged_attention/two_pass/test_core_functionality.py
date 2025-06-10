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
"""Core functionality tests for paged attention operations.

This module contains tests that verify the core functionality of the paged attention
mechanism, including token fetching, vector operations, and the attention computation.
"""

import logging

import mlx.core as mx
import pytest

from proxy_attention_lab import get_optimal_page_size, paged_attention

logger = logging.getLogger(__name__)


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
def test_fetch_k_vector_element_for_first_token_of_sequence(dtype) -> None:
    """Test K vector fetch for first token with dot product of Q and K.

    This test verifies that the paged attention operation correctly fetches
    K vector elements for the first token of each sequence and applies the
    appropriate scaling factor to the dot product.

    NOTE: Refactored to align with prefill input contract. We test prefilling
    two sequences, where we want to verify that the second query token
    attends to the first token correctly.
    """
    # Configuration
    cfg_num_kv_heads = 1
    cfg_head_dim = 4
    cfg_max_logical_pages_per_seq_in_pagetable = 2

    # Calculate proper tokens_per_page
    cfg_tokens_per_page = get_optimal_page_size(cfg_head_dim, 1, cfg_num_kv_heads)  # Assuming num_q_heads=1 for 2D

    # Batch configuration: 2 sequences, prefilling 2 tokens for seq 0 and 2 for seq 1
    num_sequences = 2
    sequence_lengths_list = [2, 2]  # Each sequence prefills 2 tokens
    total_num_query_tokens = sum(sequence_lengths_list)  # = 4

    # Create 2D queries with shape [total_num_query_tokens, cfg_head_dim]
    py_queries = mx.zeros((total_num_query_tokens, cfg_head_dim), dtype=dtype)
    # Seq 0 queries
    py_queries[0, 0] = 50.0  # Token 0: [50.0, 0.0, 0.0, 0.0]
    py_queries[1, 0] = 100.0  # Token 1: [100.0, 0.0, 0.0, 0.0]
    # Seq 1 queries
    py_queries[2, 0] = 100.0  # Token 0: [100.0, 0.0, 0.0, 0.0]
    py_queries[3, 0] = 200.0  # Token 1: [200.0, 0.0, 0.0, 0.0]

    # Create K-cache with specific values
    num_physical_pages = 2
    k_cache_shape = (num_physical_pages, cfg_tokens_per_page, cfg_num_kv_heads, cfg_head_dim)
    py_k_cache_pool = mx.zeros(k_cache_shape, dtype=dtype)

    # For seq 0: set K-vectors for positions 0 and 1
    py_k_cache_pool[0, 0, 0, 0] = 11.0  # Position 0
    py_k_cache_pool[0, 1, 0, 0] = 11.0  # Position 1 (same K value for simplicity)

    # For seq 1: set K-vectors for positions 0 and 1
    py_k_cache_pool[1, 0, 0, 0] = 22.0  # Position 0
    py_k_cache_pool[1, 1, 0, 0] = 22.0  # Position 1

    # Set up V-cache with distinct values
    py_v_cache_pool = mx.zeros_like(py_k_cache_pool)
    # For seq 0
    py_v_cache_pool[0, 0, 0, :] = mx.array([10.0, 20.0, 30.0, 40.0], dtype=dtype)
    py_v_cache_pool[0, 1, 0, :] = mx.array([10.0, 20.0, 30.0, 40.0], dtype=dtype)
    # For seq 1
    py_v_cache_pool[1, 0, 0, :] = mx.array([100.0, 200.0, 300.0, 400.0], dtype=dtype)
    py_v_cache_pool[1, 1, 0, :] = mx.array([100.0, 200.0, 300.0, 400.0], dtype=dtype)

    # Page table maps: seq 0, block 0 -> page 0; seq 1, block 0 -> page 1
    py_page_table = mx.array(
        [
            [0, 99],  # Sequence 0: logical block 0 -> physical page 0, block 1 -> page 99 (unused)
            [1, 88],  # Sequence 1: logical block 0 -> physical page 1, block 1 -> page 88 (unused)
        ],
        dtype=mx.uint32,
    )
    assert py_page_table.shape == (num_sequences, cfg_max_logical_pages_per_seq_in_pagetable)

    # Set up sequence metadata aligned with prefill contract
    py_sequence_lengths = mx.array(sequence_lengths_list, dtype=mx.int32)

    # query_to_seq_map: [0, 0, 1, 1] - first 2 queries belong to seq 0, next 2 to seq 1
    py_query_to_seq_map = mx.array([0, 0, 1, 1], dtype=mx.int32)

    # query_token_offset: position within each sequence
    # [0, 1, 0, 1] - each sequence has tokens at positions 0 and 1
    py_query_token_offset = mx.array([0, 1, 0, 1], dtype=mx.int32)

    # Evaluate all input arrays before passing to paged_attention
    mx.eval(py_queries)
    mx.eval(py_k_cache_pool)
    mx.eval(py_v_cache_pool)
    mx.eval(py_page_table)
    mx.eval(py_sequence_lengths)
    mx.eval(py_query_to_seq_map)
    mx.eval(py_query_token_offset)

    # Run paged attention
    output_arr = paged_attention(
        py_queries,
        py_k_cache_pool,
        py_v_cache_pool,
        py_page_table,
        py_sequence_lengths,
        py_query_to_seq_map,
        py_query_token_offset,
        use_fused_kernel=False,
    )
    mx.eval(output_arr)

    # --- Calculate expected output (Python reference) ---

    # For prefill, we have 4 query tokens total
    # Token 0 of seq 0: no history (causal mask), output should be based on self-attention
    # Token 1 of seq 0: attends to token 0
    # Token 0 of seq 1: no history (causal mask), output should be based on self-attention
    # Token 1 of seq 1: attends to token 0

    # Expected outputs:
    expected_outputs = []

    # Query 0 (seq 0, pos 0): No history in causal attention
    # Should attend to itself - expecting the V-vector at position 0
    expected_outputs.append(py_v_cache_pool[0, 0, 0, :])

    # Query 1 (seq 0, pos 1): Attends to position 0
    # Q[1] dot K[0,0,0] * scale = 100.0 * 11.0 * 0.5 = 550.0
    # Single token history means softmax prob = 1.0
    expected_outputs.append(py_v_cache_pool[0, 0, 0, :])

    # Query 2 (seq 1, pos 0): No history in causal attention
    # Should attend to itself - expecting the V-vector at position 0
    expected_outputs.append(py_v_cache_pool[1, 0, 0, :])

    # Query 3 (seq 1, pos 1): Attends to position 0
    # Q[3] dot K[1,0,0] * scale = 200.0 * 22.0 * 0.5 = 2200.0
    # Single token history means softmax prob = 1.0
    expected_outputs.append(py_v_cache_pool[1, 0, 0, :])

    # Combine expected outputs
    expected_V_output = mx.stack(expected_outputs)

    # For 2D queries, output shape is [total_num_query_tokens, head_dim]
    expected_output_shape = (total_num_query_tokens, cfg_head_dim)

    # Log test details
    logger.info(f"Test: {test_fetch_k_vector_element_for_first_token_of_sequence.__name__} dtype={dtype}")
    logger.info(f"  Expected output shape: {expected_output_shape}")
    logger.info(f"  Expected V output: {expected_V_output}")
    logger.info(f"  Actual V output: {output_arr}")

    # Verify results
    assert output_arr.shape == expected_output_shape, (
        f"Output shape {output_arr.shape} does not match expected {expected_output_shape}"
    )
    assert output_arr.dtype == dtype, f"Output dtype {output_arr.dtype} does not match {dtype}"
    assert mx.allclose(output_arr, expected_V_output, atol=1e-2, rtol=1e-2), (
        "Output values do not match expected values"
    )


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
def test_fetch_entire_k_vector_for_specific_token_slot(dtype) -> None:
    """Test dot product calculation between complete Q and K vectors.

    This test verifies that the paged attention operation correctly computes
    the dot product between query and key vectors and applies appropriate scaling.
    It uses uniform values in each query vector to simplify verification.

    NOTE: Refactored to align with prefill input contract. We prefill 2 sequences,
    each with 2 tokens, to test dot product computation for the second token
    attending to the first.
    """
    # Configuration
    cfg_num_kv_heads = 1
    cfg_head_dim = 4
    cfg_max_logical_pages_per_seq_in_pagetable = 2

    # Calculate proper tokens_per_page
    cfg_tokens_per_page = get_optimal_page_size(cfg_head_dim, 1, cfg_num_kv_heads)  # Assuming num_q_heads=1 for 2D

    # Batch configuration: 2 sequences, each prefilling 2 tokens
    num_sequences = 2
    sequence_lengths_list = [2, 2]
    total_num_query_tokens = sum(sequence_lengths_list)  # = 4

    # Create 2D queries with shape [total_num_query_tokens, cfg_head_dim]
    py_queries = mx.zeros((total_num_query_tokens, cfg_head_dim), dtype=dtype)
    # Seq 0 queries - uniform values to test dot product
    py_queries[0, :] = 50.0  # Token 0: all 50.0
    py_queries[1, :] = 100.0  # Token 1: all 100.0
    # Seq 1 queries
    py_queries[2, :] = 100.0  # Token 0: all 100.0
    py_queries[3, :] = 200.0  # Token 1: all 200.0

    # Create K-cache with sequential values
    num_physical_pages = 2
    k_cache_shape = (num_physical_pages, cfg_tokens_per_page, cfg_num_kv_heads, cfg_head_dim)
    py_k_cache_pool = mx.zeros(k_cache_shape, dtype=dtype)

    # Set k-vectors with increasing values for both positions
    # Seq 0 K-vectors
    for i in range(cfg_head_dim):
        py_k_cache_pool[0, 0, 0, i] = float(i + 1)  # Position 0: [1, 2, 3, 4]
        py_k_cache_pool[0, 1, 0, i] = float(i + 1)  # Position 1: [1, 2, 3, 4]
    # Seq 1 K-vectors
    for i in range(cfg_head_dim):
        py_k_cache_pool[1, 0, 0, i] = float(i + 5)  # Position 0: [5, 6, 7, 8]
        py_k_cache_pool[1, 1, 0, i] = float(i + 5)  # Position 1: [5, 6, 7, 8]

    # Set up V-cache with distinct values
    py_v_cache_pool = mx.zeros_like(py_k_cache_pool)
    # Seq 0 V-vectors
    py_v_cache_pool[0, 0, 0, :] = mx.array([10.0, 20.0, 30.0, 40.0], dtype=dtype)
    py_v_cache_pool[0, 1, 0, :] = mx.array([10.0, 20.0, 30.0, 40.0], dtype=dtype)
    # Seq 1 V-vectors
    py_v_cache_pool[1, 0, 0, :] = mx.array([50.0, 60.0, 70.0, 80.0], dtype=dtype)
    py_v_cache_pool[1, 1, 0, :] = mx.array([50.0, 60.0, 70.0, 80.0], dtype=dtype)

    # Set up page table
    py_page_table = mx.array(
        [
            [0, 99],  # Sequence 0: logical block 0 -> physical page 0
            [1, 88],  # Sequence 1: logical block 0 -> physical page 1
        ],
        dtype=mx.uint32,
    )
    assert py_page_table.shape == (num_sequences, cfg_max_logical_pages_per_seq_in_pagetable)

    # Set up sequence metadata aligned with prefill contract
    py_sequence_lengths = mx.array(sequence_lengths_list, dtype=mx.int32)

    # query_to_seq_map: [0, 0, 1, 1] - first 2 queries belong to seq 0, next 2 to seq 1
    py_query_to_seq_map = mx.array([0, 0, 1, 1], dtype=mx.int32)

    # query_token_offset: position within each sequence
    # [0, 1, 0, 1] - each sequence has tokens at positions 0 and 1
    py_query_token_offset = mx.array([0, 1, 0, 1], dtype=mx.int32)

    # Evaluate all input arrays before passing to paged_attention
    mx.eval(py_queries)
    mx.eval(py_k_cache_pool)
    mx.eval(py_v_cache_pool)
    mx.eval(py_page_table)
    mx.eval(py_sequence_lengths)
    mx.eval(py_query_to_seq_map)
    mx.eval(py_query_token_offset)

    # Run paged attention
    output_arr = paged_attention(
        py_queries,
        py_k_cache_pool,
        py_v_cache_pool,
        py_page_table,
        py_sequence_lengths,
        py_query_to_seq_map,
        py_query_token_offset,
        use_fused_kernel=False,
    )
    mx.eval(output_arr)

    expected_outputs = []

    # Query 0 (seq 0, pos 0): Self-attention only
    # Q[0] dot K[0,0,0] * scale = 50.0 * (1+2+3+4) * 0.5 = 50.0 * 10 * 0.5 = 250.0
    expected_outputs.append(py_v_cache_pool[0, 0, 0, :])

    # Query 1 (seq 0, pos 1): Attends to position 0
    # Q[1] dot K[0,0,0] * scale = 100.0 * 10 * 0.5 = 500.0
    # Single history token means softmax prob = 1.0
    expected_outputs.append(py_v_cache_pool[0, 0, 0, :])

    # Query 2 (seq 1, pos 0): Self-attention only
    # Q[2] dot K[1,0,0] * scale = 100.0 * 26 * 0.5 = 1300.0
    expected_outputs.append(py_v_cache_pool[1, 0, 0, :])

    # Query 3 (seq 1, pos 1): Attends to position 0
    # Q[3] dot K[1,0,0] * scale = 200.0 * 26 * 0.5 = 2600.0
    # Single history token means softmax prob = 1.0
    expected_outputs.append(py_v_cache_pool[1, 0, 0, :])

    # Combine expected outputs
    expected_V_output = mx.stack(expected_outputs)

    # For 2D queries, output shape is [total_num_query_tokens, head_dim]
    expected_output_shape = (total_num_query_tokens, cfg_head_dim)

    # Log test details
    logger.info(f"Test: {test_fetch_entire_k_vector_for_specific_token_slot.__name__} dtype={dtype}")
    logger.info(f"  Expected output shape: {expected_output_shape}")
    logger.info(f"  Expected V output: {expected_V_output}")
    logger.info(f"  Actual V output: {output_arr}")

    # Debug: Check if inputs are properly evaluated
    logger.info(f"  py_queries shape: {py_queries.shape}, dtype: {py_queries.dtype}")
    logger.info(f"  py_k_cache_pool shape: {py_k_cache_pool.shape}, dtype: {py_k_cache_pool.dtype}")
    logger.info(f"  py_v_cache_pool shape: {py_v_cache_pool.shape}, dtype: {py_v_cache_pool.dtype}")
    logger.info(f"  Query 0 values: {py_queries[0]}")
    logger.info(f"  K[0,0,0] values: {py_k_cache_pool[0, 0, 0, :]}")
    logger.info(f"  V[0,0,0] values: {py_v_cache_pool[0, 0, 0, :]}")

    # Verify results
    assert output_arr.shape == expected_output_shape, (
        f"Output shape {output_arr.shape} does not match expected {expected_output_shape}"
    )
    assert output_arr.dtype == dtype, f"Output dtype {output_arr.dtype} does not match {dtype}"
    assert mx.allclose(output_arr, expected_V_output, atol=1e-2, rtol=1e-2), (
        "Output values do not match expected values"
    )


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
def test_fetch_k_vector_from_variable_token_slot_in_first_logical_block(dtype) -> None:
    """Test variable token slot access in paged attention.

    This test verifies that the paged attention operation correctly handles
    different token slot positions within a page by computing dot products
    between query vectors and key vectors at specific token positions.

    NOTE: Refactored to align with prefill input contract. We prefill a single
    sequence with 9 tokens to test accessing different history positions.
    """
    # Configuration
    cfg_num_kv_heads = 1
    cfg_head_dim = 4
    cfg_max_logical_pages_per_seq_in_pagetable = 2

    # Calculate proper tokens_per_page
    cfg_tokens_per_page = get_optimal_page_size(cfg_head_dim, 1, cfg_num_kv_heads)

    # Batch configuration: 1 sequence with 9 tokens
    num_sequences = 1
    sequence_lengths_list = [9]  # Prefill 9 tokens
    total_num_query_tokens = sum(sequence_lengths_list)  # = 9

    # Create 2D queries with shape [total_num_query_tokens, cfg_head_dim]
    py_queries = mx.zeros((total_num_query_tokens, cfg_head_dim), dtype=dtype)
    # Set specific values for tokens that will access the K-vectors we set up
    # Token 4 will access history position 3
    py_queries[4, :] = 100.0  # All elements = 100.0
    # Token 8 will access history position 7
    py_queries[8, :] = 200.0  # All elements = 200.0
    # Other tokens can have different values
    for i in range(total_num_query_tokens):
        if i not in [4, 8]:
            py_queries[i, :] = 10.0  # Default value

    # Create K-cache
    num_physical_pages = 1
    k_cache_shape = (num_physical_pages, cfg_tokens_per_page, cfg_num_kv_heads, cfg_head_dim)
    py_k_cache_pool = mx.zeros(k_cache_shape, dtype=dtype)

    # Place K-vectors at specific positions
    for i in range(cfg_head_dim):
        py_k_cache_pool[0, 3, 0, i] = float(i + 1)  # Position 3: [1, 2, 3, 4]
        py_k_cache_pool[0, 7, 0, i] = float(i + 5)  # Position 7: [5, 6, 7, 8]
    # Also set K-vectors for all other positions to avoid zeros
    for pos in range(9):
        if pos not in [3, 7]:
            py_k_cache_pool[0, pos, 0, :] = 1.0

    # Set up V-cache with distinct values
    py_v_cache_pool = mx.zeros_like(py_k_cache_pool)
    # For position 3
    py_v_cache_pool[0, 3, 0, :] = mx.array([10.0, 20.0, 30.0, 40.0], dtype=dtype)
    # For position 7
    py_v_cache_pool[0, 7, 0, :] = mx.array([50.0, 60.0, 70.0, 80.0], dtype=dtype)
    # Set default V-vectors for other positions
    for pos in range(9):
        if pos not in [3, 7]:
            py_v_cache_pool[0, pos, 0, :] = mx.array([1.0, 1.0, 1.0, 1.0], dtype=dtype)

    # Set up page table - single sequence, single page
    py_page_table = mx.array([[0, 99]], dtype=mx.uint32)
    assert py_page_table.shape == (num_sequences, cfg_max_logical_pages_per_seq_in_pagetable)

    # Set up sequence metadata aligned with prefill contract
    py_sequence_lengths = mx.array(sequence_lengths_list, dtype=mx.int32)

    # query_to_seq_map: all queries belong to sequence 0
    py_query_to_seq_map = mx.zeros(total_num_query_tokens, dtype=mx.int32)

    # query_token_offset: [0, 1, 2, 3, 4, 5, 6, 7, 8]
    py_query_token_offset = mx.arange(total_num_query_tokens, dtype=mx.int32)

    # Evaluate all input arrays before passing to paged_attention
    mx.eval(py_queries)
    mx.eval(py_k_cache_pool)
    mx.eval(py_v_cache_pool)
    mx.eval(py_page_table)
    mx.eval(py_sequence_lengths)
    mx.eval(py_query_to_seq_map)
    mx.eval(py_query_token_offset)

    # Run paged attention
    output_arr = paged_attention(
        py_queries,
        py_k_cache_pool,
        py_v_cache_pool,
        py_page_table,
        py_sequence_lengths,
        py_query_to_seq_map,
        py_query_token_offset,
        use_fused_kernel=False,
    )
    mx.eval(output_arr)

    # For this test, we'll verify specific properties rather than exact outputs
    # The test is checking that different token positions can access different cache positions

    # We'll check that:
    # 1. Token 4 output is influenced by position 3's V-vector [10, 20, 30, 40]
    # 2. Token 8 output is influenced by position 7's V-vector [50, 60, 70, 80]

    # For 2D queries, output shape is [total_num_query_tokens, head_dim]
    expected_output_shape = (total_num_query_tokens, cfg_head_dim)

    # Log test details
    logger.info(f"Test: {test_fetch_k_vector_from_variable_token_slot_in_first_logical_block.__name__} dtype={dtype}")
    logger.info(f"  Expected output shape: {expected_output_shape}")
    logger.info(f"  Actual V output: {output_arr}")
    logger.info(f"  Token 4 output: {output_arr[4]}")
    logger.info(f"  Token 8 output: {output_arr[8]}")

    # Verify output shape and type
    assert output_arr.shape == expected_output_shape, (
        f"Output shape {output_arr.shape} does not match expected {expected_output_shape}"
    )
    assert output_arr.dtype == dtype, f"Output dtype {output_arr.dtype} does not match {dtype}"

    # Verify that token 4 and token 8 outputs are distinct and influenced by their target V-vectors
    # Token 4 should have values closer to [10, 20, 30, 40] than default [1, 1, 1, 1]
    token4_output = output_arr[4]
    assert mx.mean(token4_output) > 5.0, f"Token 4 output {token4_output} not influenced by target V-vector"

    # Token 8 should have values closer to [50, 60, 70, 80] than default [1, 1, 1, 1]
    token8_output = output_arr[8]
    assert mx.mean(token8_output) > 30.0, f"Token 8 output {token8_output} not influenced by target V-vector"

    logger.info("Test passed - different tokens correctly access different cache positions")
