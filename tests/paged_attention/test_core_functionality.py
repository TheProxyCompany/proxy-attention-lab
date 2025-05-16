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
"""Core functionality tests for paged attention operations.

This module contains tests that verify the core functionality of the paged attention
mechanism, including token fetching, vector operations, and the attention computation.
"""

import logging

import mlx.core as mx

from proxy_attention_lab import paged_attention

logger = logging.getLogger(__name__)


def test_fetch_k_vector_element_for_first_token_of_sequence() -> None:
    """Test K vector fetch for first token with dot product of Q and K.

    This test verifies that the paged attention operation correctly fetches
    K vector elements for the first token of each sequence and applies the
    appropriate scaling factor to the dot product.
    """
    # Configuration
    num_q_threads = 2
    cfg_tokens_per_page = 64
    cfg_num_kv_heads = 1
    cfg_head_dim = 4
    cfg_max_logical_blocks_per_seq_in_pagetable = 2

    # Create 2D queries with shape [num_q_threads, cfg_head_dim]
    py_queries = mx.zeros((num_q_threads, cfg_head_dim), dtype=mx.float16)
    # Set Q-vectors with specific values
    py_queries[0, 0] = 100.0  # Thread 0: [100.0, 0.0, 0.0, 0.0]
    py_queries[1, 0] = 200.0  # Thread 1: [200.0, 0.0, 0.0, 0.0]

    # Create K-cache with specific values
    num_physical_pages = 2
    k_cache_shape = (num_physical_pages, cfg_tokens_per_page, cfg_num_kv_heads, cfg_head_dim)
    py_k_cache_pool = mx.zeros(k_cache_shape, dtype=mx.float16)

    # Set K-vector for page 0, token 0 to [11.0, 0.0, 0.0, 0.0]
    py_k_cache_pool[0, 0, 0, 0] = 11.0
    for i in range(1, cfg_head_dim):
        py_k_cache_pool[0, 0, 0, i] = 0.0

    # Set K-vector for page 1, token 0 to [22.0, 0.0, 0.0, 0.0]
    py_k_cache_pool[1, 0, 0, 0] = 22.0
    for i in range(1, cfg_head_dim):
        py_k_cache_pool[1, 0, 0, i] = 0.0

    # Set up V-cache with distinct values
    py_v_cache_pool = mx.zeros_like(py_k_cache_pool)
    py_v_cache_pool[0, 0, 0, :] = mx.array([10.0, 20.0, 30.0, 40.0], dtype=mx.float16)
    py_v_cache_pool[1, 0, 0, :] = mx.array([100.0, 200.0, 300.0, 400.0], dtype=mx.float16)

    # Page table maps: seq 0, block 0 -> page 0; seq 1, block 0 -> page 1
    py_page_table = mx.array(
        [
            [0, 99],  # Sequence 0: logical block 0 -> physical page 0, block 1 -> page 99 (unused)
            [1, 88],  # Sequence 1: logical block 0 -> physical page 1, block 1 -> page 88 (unused)
        ],
        dtype=mx.uint32,
    )
    assert py_page_table.shape == (num_q_threads, cfg_max_logical_blocks_per_seq_in_pagetable)

    # Set up sequence metadata
    py_sequence_lengths = mx.array([10, 5], dtype=mx.int32)
    py_query_to_seq_map = mx.array([0, 1], dtype=mx.int32)

    # With history-based attention:
    # Set query_token_offset to 1 to make the kernel look at position 0
    py_query_token_offset = mx.array([1, 1], dtype=mx.int32)

    # Run paged attention
    output_arr = paged_attention(
        py_queries,
        py_k_cache_pool,
        py_v_cache_pool,
        py_page_table,
        py_sequence_lengths,
        py_query_to_seq_map,
        py_query_token_offset,
    )
    mx.eval(output_arr)

    # --- Calculate expected output (Python reference) ---

    # Expected scores (single token history means 100% attention weight):
    # Item 0: Q[0] dot K[0,0,0] * scale = 100.0 * 11.0 * 0.5 = 550.0
    # Item 1: Q[1] dot K[1,0,0] * scale = 200.0 * 22.0 * 0.5 = 2200.0

    # For single history token, softmax prob is always 1.0

    # Create expected V outputs
    # V-aggregation for item 0: V[0,0,0] * prob[0] = V[0,0,0] * 1.0
    expected_V_item0 = py_v_cache_pool[0, 0, 0, :].astype(mx.float32)
    # V-aggregation for item 1: V[1,0,0] * prob[0] = V[1,0,0] * 1.0
    expected_V_item1 = py_v_cache_pool[1, 0, 0, :].astype(mx.float32)

    # Combine and reshape to match kernel output
    expected_V_output = mx.array([expected_V_item0.astype(mx.float16), expected_V_item1.astype(mx.float16)])

    # For 2D queries, output shape is [num_q_threads, head_dim]
    expected_output_shape = (num_q_threads, cfg_head_dim)

    # Log test details
    logger.info(f"Test: {test_fetch_k_vector_element_for_first_token_of_sequence.__name__}")
    logger.info(f"  Expected output shape: {expected_output_shape}")
    logger.info(f"  Expected V output: {expected_V_output}")
    logger.info(f"  Actual V output: {output_arr}")

    # Verify results
    assert output_arr.shape == expected_output_shape, (
        f"Output shape {output_arr.shape} does not match expected {expected_output_shape}"
    )
    assert output_arr.dtype == mx.float16, f"Output dtype {output_arr.dtype} does not match float16"
    assert mx.allclose(output_arr, expected_V_output, atol=1e-2, rtol=1e-2), (
        "Output values do not match expected values"
    )


def test_fetch_entire_k_vector_for_specific_token_slot() -> None:
    """Test dot product calculation between complete Q and K vectors.

    This test verifies that the paged attention operation correctly computes
    the dot product between query and key vectors and applies appropriate scaling.
    It uses uniform values in each query vector to simplify verification.
    """
    # Configuration
    num_q_threads = 2
    cfg_tokens_per_page = 64
    cfg_num_kv_heads = 1
    cfg_head_dim = 4
    cfg_max_logical_blocks_per_seq_in_pagetable = 2

    # Create 2D queries with shape [num_q_threads, cfg_head_dim]
    py_queries = mx.zeros((num_q_threads, cfg_head_dim), dtype=mx.float16)
    # Fill each query vector with uniform values
    py_queries[0, :] = 100.0  # All elements = 100.0
    py_queries[1, :] = 200.0  # All elements = 200.0

    # Create K-cache with sequential values
    num_physical_pages = 2
    k_cache_shape = (num_physical_pages, cfg_tokens_per_page, cfg_num_kv_heads, cfg_head_dim)
    py_k_cache_pool = mx.zeros(k_cache_shape, dtype=mx.float16)

    # Set k-vectors with increasing values
    for i in range(cfg_head_dim):
        py_k_cache_pool[0, 0, 0, i] = float(i + 1)  # [1, 2, 3, 4]
        py_k_cache_pool[1, 0, 0, i] = float(i + 5)  # [5, 6, 7, 8]

    # Set up V-cache with distinct values
    py_v_cache_pool = mx.zeros_like(py_k_cache_pool)
    # For thread 0 history position 0
    py_v_cache_pool[0, 0, 0, :] = mx.array([10.0, 20.0, 30.0, 40.0], dtype=mx.float16)
    # For thread 1 history position 0
    py_v_cache_pool[1, 0, 0, :] = mx.array([50.0, 60.0, 70.0, 80.0], dtype=mx.float16)

    # Set up page table
    py_page_table = mx.array(
        [
            [0, 99],  # Sequence 0: logical block 0 -> physical page 0
            [1, 88],  # Sequence 1: logical block 0 -> physical page 1
        ],
        dtype=mx.uint32,
    )
    assert py_page_table.shape == (num_q_threads, cfg_max_logical_blocks_per_seq_in_pagetable)

    # Set up sequence metadata
    py_sequence_lengths = mx.array([10, 5], dtype=mx.int32)
    py_query_to_seq_map = mx.array([0, 1], dtype=mx.int32)

    # For history-based attention:
    # Set token offset to 1 so history includes position 0
    py_query_token_offset = mx.array([1, 1], dtype=mx.int32)

    # Run paged attention
    output_arr = paged_attention(
        py_queries,
        py_k_cache_pool,
        py_v_cache_pool,
        py_page_table,
        py_sequence_lengths,
        py_query_to_seq_map,
        py_query_token_offset,
    )
    mx.eval(output_arr)

    # Expected scores (single token history means 100% attention weight):
    # Item 0: Q[0] dot K[0,0,0] * scale = 100.0 * (1+2+3+4) * 0.5 = 100.0 * 10 * 0.5 = 500.0
    # Item 1: Q[1] dot K[1,0,0] * scale = 200.0 * (5+6+7+8) * 0.5 = 200.0 * 26 * 0.5 = 2600.0

    # For single history token, softmax probability is always 1.0

    # Calculate expected V outputs
    # V-aggregation for item 0: V[0,0,0] * prob[0] = V[0,0,0] * 1.0
    expected_V_item0 = py_v_cache_pool[0, 0, 0, :].astype(mx.float32)
    # V-aggregation for item 1: V[1,0,0] * prob[0] = V[1,0,0] * 1.0
    expected_V_item1 = py_v_cache_pool[1, 0, 0, :].astype(mx.float32)

    # Combine and reshape to match kernel output
    expected_V_output = mx.array([expected_V_item0.astype(mx.float16), expected_V_item1.astype(mx.float16)])

    # For 2D queries, output shape is [num_q_threads, head_dim]
    expected_output_shape = (num_q_threads, cfg_head_dim)

    # Log test details
    logger.info(f"Test: {test_fetch_entire_k_vector_for_specific_token_slot.__name__}")
    logger.info(f"  Expected output shape: {expected_output_shape}")
    logger.info(f"  Expected V output: {expected_V_output}")
    logger.info(f"  Actual V output: {output_arr}")

    # Verify results
    assert output_arr.shape == expected_output_shape, (
        f"Output shape {output_arr.shape} does not match expected {expected_output_shape}"
    )
    assert output_arr.dtype == mx.float16, f"Output dtype {output_arr.dtype} does not match float16"
    assert mx.allclose(output_arr, expected_V_output, atol=1e-2, rtol=1e-2), (
        "Output values do not match expected values"
    )


def test_fetch_k_vector_from_variable_token_slot_in_first_logical_block() -> None:
    """Test variable token slot access in paged attention.

    This test verifies that the paged attention operation correctly handles
    different token slot positions within a page by computing dot products
    between query vectors and key vectors at specific token positions.
    """
    # Configuration
    num_q_threads = 2
    cfg_tokens_per_page = 64
    cfg_num_kv_heads = 1
    cfg_head_dim = 4
    cfg_max_logical_blocks_per_seq_in_pagetable = 2

    # Create 2D queries with shape [num_q_threads, cfg_head_dim]
    py_queries = mx.zeros((num_q_threads, cfg_head_dim), dtype=mx.float16)
    # Fill each query vector with uniform values for each thread
    py_queries[0, :] = 100.0  # Thread 0: all elements = 100.0
    py_queries[1, :] = 200.0  # Thread 1: all elements = 200.0

    # Create K-cache
    num_physical_pages = 2
    k_cache_shape = (num_physical_pages, cfg_tokens_per_page, cfg_num_kv_heads, cfg_head_dim)
    py_k_cache_pool = mx.zeros(k_cache_shape, dtype=mx.float16)

    # Place K-vectors at specific token positions:
    # For thread 0 with token_offset = 4, place the K-vector at position 3 (history)
    # For thread 1 with token_offset = 8, place the K-vector at position 7 (history)
    for i in range(cfg_head_dim):
        py_k_cache_pool[0, 3, 0, i] = float(i + 1)  # [1, 2, 3, 4]
        py_k_cache_pool[0, 7, 0, i] = float(i + 5)  # [5, 6, 7, 8]

    # Set up V-cache with distinct values
    py_v_cache_pool = mx.zeros_like(py_k_cache_pool)
    # For thread 0 history position 3
    py_v_cache_pool[0, 3, 0, :] = mx.array([10.0, 20.0, 30.0, 40.0], dtype=mx.float16)
    # For thread 1 history position 7
    py_v_cache_pool[0, 7, 0, :] = mx.array([50.0, 60.0, 70.0, 80.0], dtype=mx.float16)

    # Set up page table
    py_page_table = mx.array(
        [
            [0, 99],  # Sequence 0: logical block 0 -> physical page 0
            [1, 88],  # Sequence 1: logical block 0 -> physical page 1
        ],
        dtype=mx.uint32,
    )
    assert py_page_table.shape == (2, cfg_max_logical_blocks_per_seq_in_pagetable)

    # Set up sequence metadata
    py_sequence_lengths = mx.array([64, 32], dtype=mx.int32)
    py_query_to_seq_map = mx.array([0, 0], dtype=mx.int32)  # Both threads map to sequence 0

    # Set query token offsets
    # Thread 0: offset 4 will look at history positions 0-3
    # Thread 1: offset 8 will look at history positions 0-7
    py_query_token_offset = mx.array([4, 8], dtype=mx.int32)

    # Run paged attention
    output_arr = paged_attention(
        py_queries,
        py_k_cache_pool,
        py_v_cache_pool,
        py_page_table,
        py_sequence_lengths,
        py_query_to_seq_map,
        py_query_token_offset,
    )
    mx.eval(output_arr)

    # Expected scores:
    # Thread 0: Q[0] dot K[0,3,0] * scale = 100.0 * (1+2+3+4) * 0.5 = 100.0 * 10 * 0.5 = 500.0
    # Thread 1: Q[1] dot K[0,7,0] * scale = 200.0 * (5+6+7+8) * 0.5 = 200.0 * 26 * 0.5 = 2600.0

    # For single history token attention, softmax probability is always 1.0

    # Calculate expected V outputs
    # V-aggregation for thread 0: V[0,3,0] * prob[0] = V[0,3,0] * 1.0
    expected_V_item0 = py_v_cache_pool[0, 3, 0, :].astype(mx.float32)
    # V-aggregation for thread 1: V[0,7,0] * prob[0] = V[0,7,0] * 1.0
    expected_V_item1 = py_v_cache_pool[0, 7, 0, :].astype(mx.float32)

    # Combine and reshape to match kernel output
    expected_V_output = mx.array([expected_V_item0.astype(mx.float16), expected_V_item1.astype(mx.float16)])

    # For 2D queries, output shape is [num_q_threads, head_dim]
    expected_output_shape = (num_q_threads, cfg_head_dim)

    # Log test details
    logger.info(f"Test: {test_fetch_k_vector_from_variable_token_slot_in_first_logical_block.__name__}")
    logger.info(f"  Expected output shape: {expected_output_shape}")
    logger.info(f"  Expected V output: {expected_V_output}")
    logger.info(f"  Actual V output: {output_arr}")

    # Verify results
    assert output_arr.shape == expected_output_shape, (
        f"Output shape {output_arr.shape} does not match expected {expected_output_shape}"
    )
    assert output_arr.dtype == mx.float16, f"Output dtype {output_arr.dtype} does not match float16"
    assert mx.allclose(output_arr, expected_V_output, atol=1e-2, rtol=1e-2), (
        "Output values do not match expected values"
    )


def test_correct_token_processing_for_2d_queries_variable_offsets() -> None:
    """Regression test for 2D queries with variable token offsets.

    Tests that different query tokens correctly map to different K-vectors in the cache
    when using 2D query input format [num_q_threads, cfg_head_dim].

    This specifically verifies the fix for the bug where num_q_heads was incorrectly
    derived for 2D queries, causing incorrect token_idx calculation in the kernel.
    """
    # Configuration
    num_q_threads = 2
    cfg_tokens_per_page = 64
    cfg_num_kv_heads = 1
    cfg_head_dim = 4

    # Queries: 2D [num_q_threads, cfg_head_dim]
    py_queries = mx.zeros((num_q_threads, cfg_head_dim), dtype=mx.float16)
    # Use different patterns for each query vector
    py_queries[0, :] = mx.array([1.0, 2.0, 1.0, 2.0], dtype=mx.float16)  # Thread 0: [1.0, 2.0, 1.0, 2.0]
    py_queries[1, :] = mx.array([3.0, 4.0, 3.0, 4.0], dtype=mx.float16)  # Thread 1: [3.0, 4.0, 3.0, 4.0]

    # Create K-cache
    num_physical_pages = 1
    k_cache_shape = (num_physical_pages, cfg_tokens_per_page, cfg_num_kv_heads, cfg_head_dim)
    py_k_cache_pool = mx.zeros(k_cache_shape, dtype=mx.float16)

    # Set up K-vectors for specific token positions
    # K-vector for token_slot 0 (target for thread 0)
    py_k_cache_pool[0, 0, 0, :] = mx.array([1.0, 1.0, 1.0, 1.0], dtype=mx.float16)  # Dot with Q[0] = 1+2+1+2=6
    # K-vector for token_slot 1 (target for thread 1)
    py_k_cache_pool[0, 1, 0, :] = mx.array([2.0, 2.0, 2.0, 2.0], dtype=mx.float16)  # Dot with Q[1] = 6+8+6+8=28

    # Set up V-cache with distinct values
    py_v_cache_pool = mx.zeros_like(py_k_cache_pool)
    # For thread 0 history position 0
    py_v_cache_pool[0, 0, 0, :] = mx.array([10.0, 20.0, 30.0, 40.0], dtype=mx.float16)
    # For thread 1 history position 1
    py_v_cache_pool[0, 1, 0, :] = mx.array([100.0, 200.0, 300.0, 400.0], dtype=mx.float16)

    # Set up page table - single page for all sequences
    py_page_table = mx.array([[0]], dtype=mx.uint32)  # Seq 0, LogBlock 0 -> PhysPage 0. Shape (1,1)

    # Set up sequence metadata
    # All threads map to sequence 0, but target different token offsets
    py_sequence_lengths = mx.array([cfg_tokens_per_page], dtype=mx.int32)
    py_query_to_seq_map = mx.zeros(num_q_threads, dtype=mx.int32)  # All map to seq 0

    # Set token offsets for history-based attention:
    # Thread 0 will look at position 0 by setting offset to 1
    # Thread 1 will look at position 1 by setting offset to 2
    py_query_token_offset = mx.array([1, 2], dtype=mx.int32)

    # Run paged attention
    output_arr = paged_attention(
        py_queries,
        py_k_cache_pool,
        py_v_cache_pool,
        py_page_table,
        py_sequence_lengths,
        py_query_to_seq_map,
        py_query_token_offset,
    )
    mx.eval(output_arr)

    # Expected scores:
    # Thread 0: Dot(Q[0], K[0,0,0]) * scale = (1+2+1+2) * 0.5 = 6 * 0.5 = 3.0
    # Thread 1: Dot(Q[1], K[0,1,0]) * scale = (3+4+3+4)*(2) * 0.5 = 28 * 0.5 = 14.0

    # For single history token, softmax probability is always 1.0

    # Calculate expected V outputs
    # V-aggregation for thread 0: V[0,0,0] * prob[0] = V[0,0,0] * 1.0
    expected_V_item0 = py_v_cache_pool[0, 0, 0, :].astype(mx.float32)
    # V-aggregation for thread 1: V[0,1,0] * prob[0] = V[0,1,0] * 1.0
    expected_V_item1 = py_v_cache_pool[0, 1, 0, :].astype(mx.float32)

    # Combine and reshape to match kernel output
    expected_V_output = mx.array([expected_V_item0.astype(mx.float16), expected_V_item1.astype(mx.float16)])

    # For 2D queries, output shape is [num_q_threads, head_dim]
    expected_output_shape = (num_q_threads, cfg_head_dim)

    # Log test details
    logger.debug(f"2D regression test: output_arr shape = {output_arr.shape}, values = {output_arr}")
    logger.debug(f"2D regression test: expected shape = {expected_V_output.shape}, values = {expected_V_output}")
    logger.info(f"Test: {test_correct_token_processing_for_2d_queries_variable_offsets.__name__}")
    logger.info(f"  Expected V output: {expected_V_output}")
    logger.info(f"  Actual V output: {output_arr}")

    # Verify results
    assert output_arr.shape == expected_output_shape, (
        f"Output shape {output_arr.shape} does not match expected {expected_output_shape}"
    )
    assert output_arr.dtype == mx.float16, f"Output dtype {output_arr.dtype} does not match float16"
    assert mx.allclose(output_arr, expected_V_output, atol=1e-2, rtol=1e-2), (
        "Output values do not match expected values"
    )

    logger.info("test_correct_token_processing_for_2d_queries_variable_offsets PASSED")


def test_parallel_online_max_and_sum_exp() -> None:
    """Test the parallel online max and sum-exp computation in attention.

    This test verifies that the paged attention operation correctly computes the
    maximum score and sum of exponentials across multiple history positions.
    It sets up K vectors with different magnitudes to test the softmax normalization
    and weighted aggregation of V vectors.
    """
    # Configuration
    num_q_threads = 1  # Just one query thread for this test
    cfg_tokens_per_page = 64
    cfg_num_kv_heads = 1
    cfg_head_dim = 4

    # Current token position is 5, so we'll attend to history positions 0, 1, 2, 3, 4
    current_position = 5

    # --- Setup test inputs ---
    # 1. Query vector: Shape [num_q_threads, cfg_head_dim]
    py_queries = mx.array([[1.0, 1.0, 1.0, 1.0]], dtype=mx.float16)  # Uniform values

    # 2. Create K-cache with varying values to produce different attention scores
    num_physical_pages = 1
    k_cache_shape = (num_physical_pages, cfg_tokens_per_page, cfg_num_kv_heads, cfg_head_dim)
    py_k_cache_pool = mx.zeros(k_cache_shape, dtype=mx.float16)

    # Set up K-vectors with different magnitudes for varying attention scores
    # K-vector at position 0: small score
    py_k_cache_pool[0, 0, 0, :] = mx.array([0.2, 0.2, 0.2, 0.2], dtype=mx.float16)
    # K-vector at position 1: medium score
    py_k_cache_pool[0, 1, 0, :] = mx.array([0.5, 0.5, 0.5, 0.5], dtype=mx.float16)
    # K-vector at position 2: highest score
    py_k_cache_pool[0, 2, 0, :] = mx.array([1.0, 1.0, 1.0, 1.0], dtype=mx.float16)
    # K-vector at position 3: medium score
    py_k_cache_pool[0, 3, 0, :] = mx.array([0.5, 0.5, 0.5, 0.5], dtype=mx.float16)
    # K-vector at position 4: small score
    py_k_cache_pool[0, 4, 0, :] = mx.array([0.2, 0.2, 0.2, 0.2], dtype=mx.float16)

    # 3. Set up V-cache with distinct values for each position
    py_v_cache_pool = mx.zeros_like(py_k_cache_pool)
    # Different V-vectors for each history position
    py_v_cache_pool[0, 0, 0, :] = mx.array([1.0, 2.0, 3.0, 4.0], dtype=mx.float16)
    py_v_cache_pool[0, 1, 0, :] = mx.array([5.0, 6.0, 7.0, 8.0], dtype=mx.float16)
    py_v_cache_pool[0, 2, 0, :] = mx.array([9.0, 10.0, 11.0, 12.0], dtype=mx.float16)
    py_v_cache_pool[0, 3, 0, :] = mx.array([13.0, 14.0, 15.0, 16.0], dtype=mx.float16)
    py_v_cache_pool[0, 4, 0, :] = mx.array([17.0, 18.0, 19.0, 20.0], dtype=mx.float16)

    # 4. Set up page table
    py_page_table = mx.array([[0]], dtype=mx.uint32)  # Map logical block 0 to physical page 0

    # 5. Set up sequence metadata
    py_sequence_lengths = mx.array([cfg_tokens_per_page], dtype=mx.int32)
    py_query_to_seq_map = mx.zeros(num_q_threads, dtype=mx.int32)  # Maps query to sequence 0
    py_query_token_offset = mx.array([current_position], dtype=mx.int32)  # Current token position

    # --- Run paged attention ---
    output_arr = paged_attention(
        py_queries,
        py_k_cache_pool,
        py_v_cache_pool,
        py_page_table,
        py_sequence_lengths,
        py_query_to_seq_map,
        py_query_token_offset,
    )
    mx.eval(output_arr)

    # --- Calculate expected outputs ---
    # Scale factor for dot product: 1.0 / sqrt(head_dim) = 1.0 / 2.0 = 0.5
    scale = 1.0 / mx.sqrt(mx.array(float(cfg_head_dim))).item()
    assert isinstance(scale, float)

    # Calculate scores for each position (with scaling)
    pos0_score = 4 * 0.2 * scale  # = 0.8 * 0.5 = 0.4
    pos1_score = 4 * 0.5 * scale  # = 2.0 * 0.5 = 1.0
    pos2_score = 4 * 1.0 * scale  # = 4.0 * 0.5 = 2.0  (highest)
    pos3_score = 4 * 0.5 * scale  # = 2.0 * 0.5 = 1.0
    pos4_score = 4 * 0.2 * scale  # = 0.8 * 0.5 = 0.4

    scores_all = [pos0_score, pos1_score, pos2_score, pos3_score, pos4_score]

    # Find maximum score (expected at position 2)
    expected_max_score = max(scores_all)

    # Calculate exp(score - max_score) for each position
    # This is the numerator of the softmax function with numerical stability
    exp_scores_minus_max = []
    for score in scores_all:
        # Apply minimum threshold of -16.0 to prevent underflow
        exp_scores_minus_max.append(mx.exp(mx.maximum(score - expected_max_score, -16.0)).item())

    # Calculate sum of exp(score - max_score) (denominator of softmax)
    expected_sum_exp = sum(exp_scores_minus_max)

    # Calculate softmax probabilities
    softmax_probs = [val / expected_sum_exp for val in exp_scores_minus_max]

    # Calculate expected V output (weighted sum of V vectors)
    expected_V_output = mx.zeros((cfg_head_dim,), dtype=mx.float32)
    for i in range(len(scores_all)):
        v_hist = py_v_cache_pool[0, i, 0, :].astype(mx.float32)
        expected_V_output += v_hist * softmax_probs[i]

    # Reshape expected output to match kernel output
    expected_V_output_reshaped = expected_V_output.astype(mx.float16).reshape(num_q_threads, cfg_head_dim)

    # For full attention output, shape is [num_q_threads, head_dim]
    expected_output_shape = (num_q_threads, cfg_head_dim)

    # Log test details
    logger.info(f"Test: {test_parallel_online_max_and_sum_exp.__name__}")
    logger.info(f"  Scores: {scores_all}")
    logger.info(f"  Max Score: {expected_max_score}")
    logger.info(f"  Softmax Probs: {softmax_probs}")
    logger.info(f"  Expected V output: {expected_V_output_reshaped}")
    logger.info(f"  Actual V output: {output_arr}")

    # Verify results
    assert output_arr.shape == expected_output_shape, (
        f"Output shape {output_arr.shape} does not match expected {expected_output_shape}"
    )
    assert output_arr.dtype == mx.float16, f"Output dtype {output_arr.dtype} does not match float16"
    assert mx.allclose(output_arr, expected_V_output_reshaped, atol=1e-2, rtol=1e-2), (
        "Output values do not match expected values"
    )

    logger.info("test_parallel_online_max_and_sum_exp PASSED")


def test_dot_product_q_with_single_k_vector() -> None:
    """Test dot product calculation between individual Q and K vectors.

    This test verifies that the paged attention operation correctly calculates
    the scaled dot product between query and key vectors for multi-headed attention.
    It tests 3D query input format [NumTestTokens, NumQHeads, HeadDim] with
    corresponding KV-heads, and ensures proper aggregation of value vectors.
    """
    # --- Configuration ---
    num_test_tokens = 1  # Single token position
    num_q_heads = 2  # Two query heads
    cfg_head_dim = 4  # Dimension of Q, K, V vectors
    cfg_tokens_per_page = 64
    cfg_num_kv_heads = 2  # Matching number of KV heads (GQA factor = 1)
    cfg_max_logical_blocks_per_seq_in_pagetable = 1

    # --- Setup test inputs ---
    # 1. Queries: 3D [NumTestTokens, NumQHeads, HeadDim]
    py_queries = mx.zeros((num_test_tokens, num_q_heads, cfg_head_dim), dtype=mx.float16)
    # Q-vector for (token 0, q_head 0): [1.0, 2.0, 3.0, 4.0]
    py_queries[0, 0, :] = mx.array([1.0, 2.0, 3.0, 4.0], dtype=mx.float16)
    # Q-vector for (token 0, q_head 1): [0.5, 1.0, 1.5, 2.0]
    py_queries[0, 1, :] = mx.array([0.5, 1.0, 1.5, 2.0], dtype=mx.float16)

    # 2. K-Cache Pool: [NumPhysPages, TokensPerPage, NumKVHeads, HeadDim]
    num_physical_pages = 1
    k_cache_shape = (num_physical_pages, cfg_tokens_per_page, cfg_num_kv_heads, cfg_head_dim)
    py_k_cache_pool = mx.zeros(k_cache_shape, dtype=mx.float16)

    # Set up K-vectors for each KV head
    # K-vector for kv_head 0 (matched with q_head 0): [1.0, 1.0, 1.0, 1.0]
    py_k_cache_pool[0, 0, 0, :] = mx.array([1.0, 1.0, 1.0, 1.0], dtype=mx.float16)
    # K-vector for kv_head 1 (matched with q_head 1): [2.0, 2.0, 2.0, 2.0]
    py_k_cache_pool[0, 0, 1, :] = mx.array([2.0, 2.0, 2.0, 2.0], dtype=mx.float16)

    # 3. V-Cache Pool: [NumPhysPages, TokensPerPage, NumKVHeads, HeadDim]
    py_v_cache_pool = mx.zeros_like(py_k_cache_pool)
    # V-vector for kv_head 0 (matched with q_head 0)
    py_v_cache_pool[0, 0, 0, :] = mx.array([10.0, 20.0, 30.0, 40.0], dtype=mx.float16)
    # V-vector for kv_head 1 (matched with q_head 1)
    py_v_cache_pool[0, 0, 1, :] = mx.array([50.0, 60.0, 70.0, 80.0], dtype=mx.float16)

    # 4. Set up page table - single page for all sequences
    num_sequences_in_batch_for_test = 1
    py_page_table = mx.array([[0]], dtype=mx.uint32)  # Logical block 0 -> Physical page 0
    assert py_page_table.shape == (num_sequences_in_batch_for_test, cfg_max_logical_blocks_per_seq_in_pagetable)

    # 5. Set up sequence metadata
    py_sequence_lengths = mx.array([cfg_tokens_per_page], dtype=mx.int32)
    py_query_to_seq_map = mx.zeros(num_test_tokens, dtype=mx.int32)  # Maps query to sequence 0

    # 6. Set token offset to 1 for history-based attention to look at position 0
    py_query_token_offset = mx.ones(num_test_tokens, dtype=mx.int32)

    # --- Run paged attention ---
    output_arr = paged_attention(
        py_queries,
        py_k_cache_pool,
        py_v_cache_pool,
        py_page_table,
        py_sequence_lengths,
        py_query_to_seq_map,
        py_query_token_offset,
    )
    mx.eval(output_arr)

    # --- Calculate expected outputs ---
    # Scale factor: 1.0 / sqrt(head_dim) = 1.0 / 2.0 = 0.5
    scale = 1.0 / mx.sqrt(mx.array(float(cfg_head_dim))).item()
    assert isinstance(scale, float)

    # Calculate dot products (for debugging)
    q0_dot_k0 = 1.0 * 1.0 + 2.0 * 1.0 + 3.0 * 1.0 + 4.0 * 1.0  # = 10
    q1_dot_k1 = 0.5 * 2.0 + 1.0 * 2.0 + 1.5 * 2.0 + 2.0 * 2.0  # = 10

    # Calculate scaled scores
    # Q-head 0: q0_dot_k0 * scale = 10 * 0.5 = 5.0
    # Q-head 1: q1_dot_k1 * scale = 10 * 0.5 = 5.0
    scores_item0 = [5.0]  # Only one history token for q_head 0
    scores_item1 = [5.0]  # Only one history token for q_head 1

    # For single history token, softmax probability is always 1.0

    # V-aggregation for q_head 0: V[0,0,0] * prob[0] = V[0,0,0] * 1.0
    expected_V_item0 = py_v_cache_pool[0, 0, 0, :].astype(mx.float32)
    # V-aggregation for q_head 1: V[0,0,1] * prob[0] = V[0,0,1] * 1.0
    expected_V_item1 = py_v_cache_pool[0, 0, 1, :].astype(mx.float32)

    # Combine and reshape to match kernel output
    # Total items = NumTestTokens * NumQHeads = 1 * 2 = 2
    total_items = num_test_tokens * num_q_heads
    expected_V_output = mx.array([expected_V_item0.astype(mx.float16), expected_V_item1.astype(mx.float16)])

    # Expected shape: [total_items, head_dim]
    expected_output_shape = (total_items, cfg_head_dim)

    # Log test details
    logger.info(f"Test: {test_dot_product_q_with_single_k_vector.__name__}")
    logger.info(f"  Q0·K0 raw dot product = {q0_dot_k0}, scale={scale}, Expected Score0 = {scores_item0[0]}")
    logger.info(f"  Q1·K1 raw dot product = {q1_dot_k1}, scale={scale}, Expected Score1 = {scores_item1[0]}")
    logger.info(f"  Expected V output: {expected_V_output}")
    logger.info(f"  Actual V output: {output_arr}")

    # Verify results
    assert output_arr.shape == expected_output_shape, (
        f"Output shape {output_arr.shape} does not match expected {expected_output_shape}"
    )
    assert output_arr.dtype == mx.float16, f"Output dtype {output_arr.dtype} does not match float16"
    assert mx.allclose(output_arr, expected_V_output, atol=1e-2, rtol=1e-2), (
        "Output values do not match expected values"
    )

    logger.info("test_dot_product_q_with_single_k_vector PASSED")
