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
"""Tests for various attention computation scenarios.

This module contains tests that verify the paged attention operation correctly
handles different attention scenarios such as history in single or multiple blocks,
zero history, sequence length limits, and page table boundaries.
"""

import logging

import mlx.core as mx
import pytest

from proxy_attention_lab import paged_attention

logger = logging.getLogger(__name__)


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
def test_max_score_over_history_in_one_block(dtype) -> None:
    """Test full attention computation within a single block.

    This test verifies the full attention computation (max score, softmax, and V-aggregation)
    for a single query with multiple history tokens all contained within one logical block.
    """
    # --- Configuration ---
    num_q_threads = 1
    cfg_tokens_per_page = 64
    cfg_num_kv_heads = 1
    cfg_head_dim = 4
    current_position = 3  # Position of the current token (history: positions 0, 1, 2)

    # --- Setup test inputs ---
    # 1. Query vector: shape [num_q_threads, cfg_head_dim]
    py_queries = mx.array([[1.0, 2.0, 3.0, 4.0]], dtype=dtype)

    # 2. Create K-cache with values for history positions
    k_cache_shape = (1, cfg_tokens_per_page, cfg_num_kv_heads, cfg_head_dim)
    py_k_cache_pool = mx.zeros(k_cache_shape, dtype=dtype)
    # Set K-vectors for each history position
    py_k_cache_pool[0, 0, 0, :] = mx.array([1.0, 1.0, 1.0, 1.0], dtype=dtype)  # Position 0
    py_k_cache_pool[0, 1, 0, :] = mx.array([2.0, 2.0, 2.0, 2.0], dtype=dtype)  # Position 1
    py_k_cache_pool[0, 2, 0, :] = mx.array([0.5, 0.5, 0.5, 0.5], dtype=dtype)  # Position 2

    # 3. Create V-cache with distinct values for each position
    py_v_cache_pool = mx.zeros(k_cache_shape, dtype=dtype)
    py_v_cache_pool[0, 0, 0, :] = mx.array([10.0, 11.0, 12.0, 13.0], dtype=dtype)  # Position 0
    py_v_cache_pool[0, 1, 0, :] = mx.array([20.0, 21.0, 22.0, 23.0], dtype=dtype)  # Position 1
    py_v_cache_pool[0, 2, 0, :] = mx.array([30.0, 31.0, 32.0, 33.0], dtype=dtype)  # Position 2

    # 4. Set up page table and sequence metadata
    py_page_table = mx.array([[0]], dtype=mx.uint32)  # Map logical block 0 -> physical page 0
    py_sequence_lengths = mx.array([cfg_tokens_per_page], dtype=mx.int32)
    py_query_to_seq_map = mx.zeros(num_q_threads, dtype=mx.int32)
    py_query_token_offset = mx.array([current_position], dtype=mx.int32)

    mx.eval(py_queries)
    mx.eval(py_k_cache_pool)
    mx.eval(py_v_cache_pool)
    mx.eval(py_page_table)
    mx.eval(py_sequence_lengths)
    mx.eval(py_query_to_seq_map)
    mx.eval(py_query_token_offset)

    # --- Run paged attention ---
    output_arr = paged_attention(
        py_queries,
        py_k_cache_pool,
        py_v_cache_pool,
        py_page_table,
        py_sequence_lengths,
        py_query_to_seq_map,
        py_query_token_offset,
        use_fused_kernel=True,
    )
    mx.eval(output_arr)

    # --- Calculate expected output (Python reference) ---
    # Scale factor: 1.0 / sqrt(head_dim) for scaled dot-product attention
    py_scale = 1.0 / mx.sqrt(mx.array(float(cfg_head_dim))).item()

    # Calculate scores for each history position
    scores_val = []
    q_vec_py = py_queries[0].astype(mx.float32)
    for hist_idx_calc in range(current_position):
        k_vec_py = py_k_cache_pool[0, hist_idx_calc, 0, :].astype(mx.float32)
        score = (mx.sum(q_vec_py * k_vec_py) * py_scale).item()
        scores_val.append(score)

    # Find maximum score
    true_max_score = -float("inf")
    for s_val in scores_val:
        if s_val > true_max_score:
            true_max_score = s_val
    if not scores_val:
        true_max_score = 0.0  # Handle empty history case for max

    # Calculate exp(score - max_score) with numerical stability
    exp_scores_minus_max = []
    for s_val in scores_val:
        exp_scores_minus_max.append(mx.exp(mx.maximum(s_val - true_max_score, -16.0)).item())

    # Calculate sum of exponentials for softmax denominator
    true_sum_exp_score = sum(exp_scores_minus_max)
    if true_sum_exp_score == 0 and not scores_val:
        # If history was empty, sum_exp is 0
        pass
    elif true_sum_exp_score == 0:
        # If history non-empty but all scores led to sum_exp 0 (e.g. all -inf)
        true_sum_exp_score = 1.0  # Avoid division by zero, effectively making probs 0

    # Calculate softmax probabilities
    softmax_probs = []
    if true_sum_exp_score != 0:
        for val in exp_scores_minus_max:
            softmax_probs.append(val / true_sum_exp_score)
    else:  # All scores were effectively -infinity or history empty
        softmax_probs = [0.0] * len(scores_val) if scores_val else []

    # Calculate expected weighted sum of V-vectors
    expected_V_output_py = mx.zeros((cfg_head_dim,), dtype=mx.float32)
    if softmax_probs:  # Only aggregate if there are probabilities
        for i in range(current_position):
            v_hist = py_v_cache_pool[0, i, 0, :].astype(mx.float32)
            expected_V_output_py += v_hist * softmax_probs[i]

    # Reshape to match expected output format
    expected_V_output_reshaped = expected_V_output_py.astype(dtype).reshape(num_q_threads, cfg_head_dim)

    # Log test details
    logger.info(f"Test: {test_max_score_over_history_in_one_block.__name__} dtype={dtype}")
    logger.info(f"  Scores: {scores_val}")
    logger.info(f"  Max Score: {true_max_score}")
    logger.info(f"  Softmax Probs: {softmax_probs}")
    logger.info(f"  Expected V output: {expected_V_output_reshaped}")
    logger.info(f"  Actual V output: {output_arr}")

    # Verify results
    assert output_arr.shape == expected_V_output_reshaped.shape, (
        f"Output shape {output_arr.shape} does not match expected {expected_V_output_reshaped.shape}"
    )
    assert output_arr.dtype == dtype, f"Output dtype {output_arr.dtype} does not match {dtype}"
    # Increased tolerance slightly for float16/bfloat16 sum
    assert mx.allclose(output_arr, expected_V_output_reshaped, atol=1e-2, rtol=1e-2), (
        "Output values do not match expected values"
    )


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
def test_max_score_over_multi_block_history(dtype) -> None:
    """Test maximum score calculation over history spanning multiple logical blocks.

    This test verifies that the paged attention operation can correctly:
    1. Identify the current query token's logical position
    2. Loop through historical token positions from 0 up to current position
    3. Map each history position to its correct logical block and token slot
    4. Use the page table to find the physical page for each logical block
    5. Compute dot product scores with each historical K-vector
    6. Find and return the maximum score across blocks

    Unlike test_max_score_over_history_in_one_block, this test places history tokens
    in multiple logical blocks that map to different physical pages.
    """
    # --- Configuration ---
    num_q_threads = 1  # Just one query thread for this test
    cfg_tokens_per_page = 3  # Small value to ensure we span multiple blocks
    cfg_num_kv_heads = 1
    cfg_head_dim = 4

    # Current token position is 5, which spans two logical blocks:
    # - History positions 0,1,2 are in logical block 0 (positions 0,1,2 within the block)
    # - History positions 3,4 are in logical block 1 (positions 0,1 within the block)
    current_position = 5

    # --- Setup test inputs ---
    # 1. Query vector: shape [num_q_threads, cfg_head_dim]
    py_queries = mx.array([[1.0, 2.0, 3.0, 4.0]], dtype=dtype)

    # 2. Create K-cache with values for history positions in multiple blocks
    num_physical_pages = 2  # Two physical pages for the two logical blocks
    k_cache_shape = (num_physical_pages, cfg_tokens_per_page, cfg_num_kv_heads, cfg_head_dim)
    py_k_cache_pool = mx.zeros(k_cache_shape, dtype=dtype)

    # Set K-vectors in physical page 0 (logical block 0)
    py_k_cache_pool[0, 0, 0, :] = mx.array([1.0, 1.0, 1.0, 1.0], dtype=dtype)  # Position 0
    py_k_cache_pool[0, 1, 0, :] = mx.array([2.0, 2.0, 2.0, 2.0], dtype=dtype)  # Position 1
    py_k_cache_pool[0, 2, 0, :] = mx.array([0.5, 0.5, 0.5, 0.5], dtype=dtype)  # Position 2

    # Set K-vectors in physical page 1 (logical block 1)
    py_k_cache_pool[1, 0, 0, :] = mx.array([3.0, 3.0, 3.0, 3.0], dtype=dtype)  # Position 3
    py_k_cache_pool[1, 1, 0, :] = mx.array([1.5, 1.5, 1.5, 1.5], dtype=dtype)  # Position 4

    # 3. Create V-cache with distinct values for each position
    py_v_cache_pool = mx.zeros_like(py_k_cache_pool)
    # Set V-vectors in physical page 0 (logical block 0)
    py_v_cache_pool[0, 0, 0, :] = mx.array([10.0, 20.0, 30.0, 40.0], dtype=dtype)  # Position 0
    py_v_cache_pool[0, 1, 0, :] = mx.array([50.0, 60.0, 70.0, 80.0], dtype=dtype)  # Position 1
    py_v_cache_pool[0, 2, 0, :] = mx.array([5.0, 6.0, 7.0, 8.0], dtype=dtype)  # Position 2
    # Set V-vectors in physical page 1 (logical block 1)
    py_v_cache_pool[1, 0, 0, :] = mx.array([100.0, 110.0, 120.0, 130.0], dtype=dtype)  # Position 3
    py_v_cache_pool[1, 1, 0, :] = mx.array([15.0, 25.0, 35.0, 45.0], dtype=dtype)  # Position 4

    # 4. Set up page table - maps logical blocks to physical pages
    py_page_table = mx.array([[0, 1]], dtype=mx.uint32)  # Logical block 0 -> page 0, block 1 -> page 1

    # 5. Set up sequence metadata
    py_sequence_lengths = mx.array([10], dtype=mx.int32)  # More than current_position
    py_query_to_seq_map = mx.zeros(num_q_threads, dtype=mx.int32)  # Maps query to sequence 0
    py_query_token_offset = mx.array([current_position], dtype=mx.int32)  # Current token position

    mx.eval(py_queries)
    mx.eval(py_k_cache_pool)
    mx.eval(py_v_cache_pool)
    mx.eval(py_page_table)
    mx.eval(py_sequence_lengths)
    mx.eval(py_query_to_seq_map)
    mx.eval(py_query_token_offset)

    # --- Run paged attention ---
    output_arr = paged_attention(
        py_queries,
        py_k_cache_pool,
        py_v_cache_pool,
        py_page_table,
        py_sequence_lengths,
        py_query_to_seq_map,
        py_query_token_offset,
        use_fused_kernel=True,
    )
    mx.eval(output_arr)

    # --- Calculate expected output (Python reference) ---
    # Scale factor: 1.0 / sqrt(head_dim)
    scale = 1.0 / mx.sqrt(mx.array(float(cfg_head_dim))).item()
    assert isinstance(scale, float)

    # Calculate scores for each history position
    scores_val = []
    # Store V-vectors for each history position for later aggregation
    v_vectors = []

    # Loop through history positions and calculate scaled QK scores
    for hist_pos in range(current_position):
        # Map history position to logical block and token slot
        logical_block_idx = hist_pos // cfg_tokens_per_page
        token_slot_in_page = hist_pos % cfg_tokens_per_page

        # Get physical page from page table
        physical_page_id = py_page_table[0, logical_block_idx].item()

        # Get K-vector and V-vector for this historical position
        k_vec = py_k_cache_pool[physical_page_id, token_slot_in_page, 0, :].astype(mx.float32)
        v_vec = py_v_cache_pool[physical_page_id, token_slot_in_page, 0, :].astype(mx.float32)

        # Store V-vector for later use
        v_vectors.append(v_vec)

        # Compute the dot product and apply scaling
        q_vec = py_queries[0].astype(mx.float32)
        score = (mx.sum(q_vec * k_vec) * scale).item()
        scores_val.append(score)

    # Calculate detailed score values for each history position
    # Score calculation: dot_product(query, key) * scale
    score0 = (1.0 * 1.0 + 2.0 * 1.0 + 3.0 * 1.0 + 4.0 * 1.0) * scale  # = 10 * 0.5 = 5.0 (block 0, slot 0)
    score1 = (1.0 * 2.0 + 2.0 * 2.0 + 3.0 * 2.0 + 4.0 * 2.0) * scale  # = 20 * 0.5 = 10.0 (block 0, slot 1)
    score2 = (1.0 * 0.5 + 2.0 * 0.5 + 3.0 * 0.5 + 4.0 * 0.5) * scale  # = 5 * 0.5 = 2.5 (block 0, slot 2)
    score3 = (1.0 * 3.0 + 2.0 * 3.0 + 3.0 * 3.0 + 4.0 * 3.0) * scale  # = 30 * 0.5 = 15.0 (block 1, slot 0)
    score4 = (1.0 * 1.5 + 2.0 * 1.5 + 3.0 * 1.5 + 4.0 * 1.5) * scale  # = 15 * 0.5 = 7.5 (block 1, slot 1)

    # Find maximum score
    true_max_score = -float("inf")
    if scores_val:  # Ensure scores_val is not empty
        for s_val in scores_val:
            if s_val > true_max_score:
                true_max_score = s_val
    else:  # Handle case with no valid scores (e.g., empty history)
        true_max_score = 0.0

    # Calculate exp(score - max_score) with numerical stability
    exp_scores_minus_max = []
    for s_val in scores_val:
        exp_scores_minus_max.append(mx.exp(mx.maximum(s_val - true_max_score, -16.0)).item())

    # Calculate sum of exponentials for softmax denominator
    true_sum_exp_score = sum(exp_scores_minus_max)
    # Handle edge cases for sum_exp
    if not scores_val:  # If original scores_val was empty
        true_sum_exp_score = 0.0  # Based on convention for empty softmax
    elif true_sum_exp_score < 1e-9:  # If sum is effectively zero due to all scores being very low
        true_sum_exp_score = 1.0  # Prevent division by zero, probs will be ~0

    # Calculate softmax probabilities
    softmax_probs = []
    if true_sum_exp_score != 0.0:
        for val in exp_scores_minus_max:
            softmax_probs.append(val / true_sum_exp_score)
    elif scores_val:  # Scores existed but sum_exp was zero (all scores were ~ -inf)
        softmax_probs = [0.0] * len(scores_val)
    # else: softmax_probs remains empty if scores_val was empty

    # Calculate expected weighted sum of V-vectors
    expected_V_output_py = mx.zeros((cfg_head_dim,), dtype=mx.float32)
    if softmax_probs:  # Only aggregate if there are probabilities
        for i in range(len(softmax_probs)):
            expected_V_output_py += v_vectors[i] * softmax_probs[i]

    # Reshape to match expected output format
    expected_V_output_reshaped = expected_V_output_py.astype(dtype).reshape(num_q_threads, cfg_head_dim)

    # For attention output, shape is [num_q_threads, cfg_head_dim]
    expected_output_shape = (num_q_threads, cfg_head_dim)

    # Log test details
    logger.info(f"Test: {test_max_score_over_multi_block_history.__name__} dtype={dtype}")
    logger.info("Scores across multiple blocks:")
    logger.info(f"  Score for hist_pos=0 (block 0, slot 0): {score0}")
    logger.info(f"  Score for hist_pos=1 (block 0, slot 1): {score1}")
    logger.info(f"  Score for hist_pos=2 (block 0, slot 2): {score2}")
    logger.info(f"  Score for hist_pos=3 (block 1, slot 0): {score3}")
    logger.info(f"  Score for hist_pos=4 (block 1, slot 1): {score4}")
    logger.info(f"Max Score: {true_max_score}")
    logger.info(f"Softmax Probs: {softmax_probs}")
    logger.info(f"Expected V output: {expected_V_output_reshaped}")
    logger.info(f"Actual V output: {output_arr}")

    # Verify results
    assert output_arr.shape == expected_output_shape, (
        f"Output shape {output_arr.shape} does not match expected {expected_output_shape}"
    )
    assert output_arr.dtype == dtype, f"Output dtype {output_arr.dtype} does not match {dtype}"
    assert mx.allclose(output_arr, expected_V_output_reshaped, atol=1e-2, rtol=1e-2), (
        "Output values do not match expected values"
    )

    logger.info("test_max_score_over_multi_block_history PASSED")


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
def test_zero_history_returns_zero_score(dtype) -> None:
    """Test that zero history returns zero score.

    This test verifies the code path in the kernel where 'effective_history_length = 0',
    which should result in 'max_score_half = 0.0h' instead of -infinity or garbage values.
    The zero-history case occurs when query_token_offset is 0, meaning the query is
    at the first position with no history tokens before it to compute attention scores with.
    """
    # --- Configuration ---
    num_q_threads = 2  # Test with multiple threads to ensure consistency
    cfg_tokens_per_page = 64
    cfg_num_kv_heads = 1
    cfg_head_dim = 4

    # --- Setup test inputs ---
    # 1. Query vector: Simple query vectors, content doesn't matter as no dot products should occur
    py_queries = mx.array(
        [
            [1.0, 2.0, 3.0, 4.0],  # Query for thread 0
            [5.0, 6.0, 7.0, 8.0],  # Query for thread 1
        ],
        dtype=dtype,
    )

    # 2. K-Cache Pool: Minimal setup with some values (won't be accessed)
    num_physical_pages = 1
    k_cache_shape = (num_physical_pages, cfg_tokens_per_page, cfg_num_kv_heads, cfg_head_dim)
    py_k_cache_pool = mx.ones(k_cache_shape, dtype=dtype) * 10.0  # Fill with 10s

    # 3. V-Cache Pool with values to verify that these are aggregated correctly
    py_v_cache_pool = mx.zeros_like(py_k_cache_pool)
    # Set up V-cache with values to verify correct aggregation
    py_v_cache_pool[0, 0, 0, :] = mx.array([10.0, 20.0, 30.0, 40.0], dtype=dtype)  # Position 0
    py_v_cache_pool[0, 1, 0, :] = mx.array([50.0, 60.0, 70.0, 80.0], dtype=dtype)  # Position 1 (highest score)
    py_v_cache_pool[0, 2, 0, :] = mx.array([5.0, 6.0, 7.0, 8.0], dtype=dtype)  # Position 2
    # Positions 3-4 should NOT be accessed due to sequence length limit
    py_v_cache_pool[0, 3, 0, :] = mx.array([100.0, 200.0, 300.0, 400.0], dtype=dtype)  # Position 3
    # The following line is only valid if num_physical_pages > 1, so we guard it
    if num_physical_pages > 1:
        py_v_cache_pool[1, 0, 0, :] = mx.array([500.0, 600.0, 700.0, 800.0], dtype=dtype)  # Position 4

    # 4. Page Table: Simple mapping for one logical block
    py_page_table = mx.array([[0]], dtype=mx.uint32)  # Shape (1, 1)

    # 5. Sequence Lengths: One sequence with tokens
    py_sequence_lengths = mx.array([10], dtype=mx.int32)

    # 6. Query to Sequence Map: Both queries map to sequence 0
    py_query_to_seq_map = mx.zeros(num_q_threads, dtype=mx.int32)

    # 7. Query Token Offset: CRUCIAL - set to 0 to test zero-history case
    py_query_token_offset = mx.zeros(num_q_threads, dtype=mx.int32)

    mx.eval(py_queries)
    mx.eval(py_k_cache_pool)
    mx.eval(py_v_cache_pool)
    mx.eval(py_page_table)
    mx.eval(py_sequence_lengths)
    mx.eval(py_query_to_seq_map)
    mx.eval(py_query_token_offset)

    # --- Run paged attention ---
    output_arr = paged_attention(
        py_queries,
        py_k_cache_pool,
        py_v_cache_pool,
        py_page_table,
        py_sequence_lengths,
        py_query_to_seq_map,
        py_query_token_offset,
        use_fused_kernel=True,
    )
    mx.eval(output_arr)

    # --- Calculate expected output ---
    # For zero history, the kernel should return zeros for all V-vectors
    expected_v_output = mx.zeros((num_q_threads, cfg_head_dim), dtype=dtype)

    # Output is now full attention format [num_q_threads, cfg_head_dim]
    expected_output_shape = (num_q_threads, cfg_head_dim)

    logger.info(f"Test: {test_zero_history_returns_zero_score.__name__} dtype={dtype}")
    logger.info(f"  Query token offsets = {py_query_token_offset}")
    logger.info(f"  Expected V output = {expected_v_output}")
    logger.info(f"  Actual V output = {output_arr}")

    # Verify results
    assert output_arr.shape == expected_output_shape, (
        f"Output shape {output_arr.shape} does not match expected {expected_output_shape}"
    )
    assert output_arr.dtype == dtype, f"Output dtype {output_arr.dtype} does not match {dtype}"
    # Verify V output contains all zeros
    assert mx.allclose(output_arr, expected_v_output, atol=1e-3), (
        "Output values are not zero as expected for zero history"
    )

    logger.info("test_zero_history_returns_zero_score PASSED")


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
def test_history_limited_by_sequence_length(dtype) -> None:
    """Test history truncation based on sequence length limits.

    This test verifies that when a query token's logical offset implies a history that extends
    beyond the actual_sequence_length, the kernel correctly processes only up to
    actual_sequence_length - 1 and ignores positions that would be out of bounds.

    In this test, query_token_offset=5 implies 5 history tokens (positions 0-4),
    but actual_sequence_length=3 means only positions 0-2 should be considered.

    The kernel achieves this via the line:
    uint effective_history_length = min(current_q_token_logical_pos, actual_sequence_length);
    """
    # --- Configuration ---
    cfg_tokens_per_page = 4  # Small value for easier tracking
    cfg_num_kv_heads = 1
    cfg_head_dim = 4
    actual_sequence_length = 3  # Sequence contains only 3 tokens
    query_token_offset = 5  # Query position implies 5 history tokens (0-4)

    # --- Setup test inputs ---
    # 1. Query vector: Single query
    py_queries = mx.array([[1.0, 1.0, 1.0, 1.0]], dtype=dtype)

    # 2. K-Cache Pool: Set up with different values for positions 0-4
    num_physical_pages = 2  # Need enough pages for all positions
    k_cache_shape = (num_physical_pages, cfg_tokens_per_page, cfg_num_kv_heads, cfg_head_dim)
    py_k_cache_pool = mx.zeros(k_cache_shape, dtype=dtype)

    # Set up K-vectors for each history position
    # Position 0 - Score will be 2.0 after scale
    py_k_cache_pool[0, 0, 0, :] = mx.array([1.0, 1.0, 1.0, 1.0], dtype=dtype)

    # Position 1 - Score will be 6.0 after scale (should be the max within valid sequence)
    py_k_cache_pool[0, 1, 0, :] = mx.array([3.0, 3.0, 3.0, 3.0], dtype=dtype)

    # Position 2 - Score will be 4.0 after scale
    py_k_cache_pool[0, 2, 0, :] = mx.array([2.0, 2.0, 2.0, 2.0], dtype=dtype)

    # Position 3 - Beyond sequence length, should NOT be accessed
    py_k_cache_pool[0, 3, 0, :] = mx.array([8.0, 8.0, 8.0, 8.0], dtype=dtype)  # Score would be 16.0

    # Position 4 - Beyond sequence length, should NOT be accessed
    py_k_cache_pool[1, 0, 0, :] = mx.array([10.0, 10.0, 10.0, 10.0], dtype=dtype)  # Score would be 20.0

    # 3. V-Cache Pool with values to verify that these are aggregated correctly
    py_v_cache_pool = mx.zeros_like(py_k_cache_pool)
    # Set up V-cache with values to verify correct aggregation
    py_v_cache_pool[0, 0, 0, :] = mx.array([10.0, 20.0, 30.0, 40.0], dtype=dtype)  # Position 0
    py_v_cache_pool[0, 1, 0, :] = mx.array([50.0, 60.0, 70.0, 80.0], dtype=dtype)  # Position 1 (highest score)
    py_v_cache_pool[0, 2, 0, :] = mx.array([5.0, 6.0, 7.0, 8.0], dtype=dtype)  # Position 2
    # Positions 3-4 should NOT be accessed due to sequence length limit (actual_sequence_length=3)
    py_v_cache_pool[0, 3, 0, :] = mx.array([100.0, 200.0, 300.0, 400.0], dtype=dtype)  # Position 3
    py_v_cache_pool[1, 0, 0, :] = mx.array([500.0, 600.0, 700.0, 800.0], dtype=dtype)  # Position 4

    # 4. Page Table: Two logical blocks mapped to two physical pages
    py_page_table = mx.array([[0, 1]], dtype=mx.uint32)  # Shape (1, 2)

    # 5. Sequence Lengths: One sequence with only 3 tokens
    py_sequence_lengths = mx.array([actual_sequence_length], dtype=mx.int32)

    # 6. Query to Sequence Map: Single query maps to sequence 0
    py_query_to_seq_map = mx.zeros(1, dtype=mx.int32)

    # 7. Query Token Offset: Set to 5, so history would be positions 0-4
    py_query_token_offset = mx.array([query_token_offset], dtype=mx.int32)

    # --- Run paged attention ---
    output_arr = paged_attention(
        py_queries,
        py_k_cache_pool,
        py_v_cache_pool,
        py_page_table,
        py_sequence_lengths,
        py_query_to_seq_map,
        py_query_token_offset,
        use_fused_kernel=True,
    )
    mx.eval(output_arr)

    # --- Calculate expected output (Python reference) ---
    # Scale factor for dot product
    denominator = mx.sqrt(mx.array(float(cfg_head_dim))).item()
    assert isinstance(denominator, float)
    py_scale = 1.0 / denominator

    # Expected scores for positions that are within actual_sequence_length (0, 1, 2)
    score0 = 4.0 * py_scale  # = 4 * 0.5 = 2.0
    score1 = 12.0 * py_scale  # = 12 * 0.5 = 6.0
    score2 = 8.0 * py_scale  # = 8 * 0.5 = 4.0

    # Calculate expected V aggregation based on softmax over positions 0-2
    # Find maximum score from positions 0-2
    true_max_score = max(score0, score1, score2)  # Should be score1 = 6.0

    # Calculate exp(score - max_score) with clamp for each position
    exp_score0 = mx.exp(mx.maximum(score0 - true_max_score, -16.0)).item()
    exp_score1 = mx.exp(mx.maximum(score1 - true_max_score, -16.0)).item()
    exp_score2 = mx.exp(mx.maximum(score2 - true_max_score, -16.0)).item()

    # Calculate sum of exp scores
    true_sum_exp_score = exp_score0 + exp_score1 + exp_score2

    # Calculate softmax probabilities
    prob0 = exp_score0 / true_sum_exp_score
    prob1 = exp_score1 / true_sum_exp_score
    prob2 = exp_score2 / true_sum_exp_score

    # Get V-vectors for each position
    v_vec0 = py_v_cache_pool[0, 0, 0, :].astype(mx.float32)
    v_vec1 = py_v_cache_pool[0, 1, 0, :].astype(mx.float32)
    v_vec2 = py_v_cache_pool[0, 2, 0, :].astype(mx.float32)

    # Calculate weighted sum of V-vectors
    expected_V_output_py = v_vec0 * prob0 + v_vec1 * prob1 + v_vec2 * prob2

    # Reshape to match output format [num_q_threads, cfg_head_dim]
    num_q_threads = 1  # For this test
    expected_V_output = expected_V_output_py.astype(dtype).reshape(num_q_threads, cfg_head_dim)

    # Output is now full attention format [num_q_threads, cfg_head_dim]
    expected_output_shape = (num_q_threads, cfg_head_dim)

    logger.info(f"Test: {test_history_limited_by_sequence_length.__name__} dtype={dtype}")
    logger.info(f"  actual_sequence_length = {actual_sequence_length}, query_token_offset = {query_token_offset}")
    logger.info("  Scores for positions within sequence length:")
    logger.info(f"    Position 0 = {score0}")
    logger.info(f"    Position 1 = {score1} (max)")
    logger.info(f"    Position 2 = {score2}")
    logger.info(f"  Softmax probabilities = [{prob0}, {prob1}, {prob2}]")
    logger.info(f"  Expected V output = {expected_V_output}")
    logger.info(f"  Actual V output = {output_arr}")

    # Verify results
    assert output_arr.shape == expected_output_shape, (
        f"Output shape {output_arr.shape} does not match expected {expected_output_shape}"
    )
    assert output_arr.dtype == dtype, f"Output dtype {output_arr.dtype} does not match {dtype}"
    # Verify the V output contains the weighted sum of only positions 0-2,
    # not including positions 3-4 which should be excluded by sequence length limit
    assert mx.allclose(output_arr, expected_V_output, atol=1e-2, rtol=1e-2), (
        "Output values do not match expected weighted sum based on sequence length limit"
    )

    logger.info("test_history_limited_by_sequence_length PASSED")


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
def test_history_scan_stops_at_page_table_limit(dtype) -> None:
    """Test that history scan stops at page table limits.

    This test verifies that if the history scan encounters a logical_block_idx that is
    >= params->max_logical_pages_per_seq (i.e., beyond what the page table describes
    for that sequence), the kernel correctly stops scanning further history but still
    returns the max score found from valid preceding blocks.

    In this test, query_token_offset=5 implies 5 history tokens (positions 0-4), but
    the page table only describes 2 logical blocks (covering positions 0-3), so position 4
    should be ignored even though it's within sequence_length.

    The kernel achieves this via the code:
    if (logical_block_idx >= params->max_logical_pages_per_seq) {
        break; // No more valid blocks for this sequence's history
    }
    """
    # --- Configuration ---
    cfg_tokens_per_page = 2  # Small value to ensure we span multiple blocks quickly
    cfg_num_kv_heads = 1
    cfg_head_dim = 4
    max_logical_pages_per_seq_in_pagetable = 2  # Page table only describes 2 logical blocks
    query_token_offset = 5  # History positions: 0,1,2,3,4

    # --- Setup test inputs ---
    # 1. Q-vector: Single query
    py_queries = mx.array([[1.0, 1.0, 1.0, 1.0]], dtype=dtype)

    # 2. K-Cache Pool:
    num_physical_pages = 2  # Two physical pages for the two logical blocks in page table
    k_cache_shape = (num_physical_pages, cfg_tokens_per_page, cfg_num_kv_heads, cfg_head_dim)
    py_k_cache_pool = mx.zeros(k_cache_shape, dtype=dtype)

    # Position mapping with cfg_tokens_per_page = 2:
    # hist_pos 0, 1 -> logical_block_idx 0 -> physical_page 0
    # hist_pos 2, 3 -> logical_block_idx 1 -> physical_page 1
    # hist_pos 4    -> logical_block_idx 2 (beyond page_table limit) -> should not be accessed

    # K-vectors for logical block 0 (positions 0, 1)
    # Position 0 (token_slot 0 on physical_page 0) - Score will be 2.0 after scale
    py_k_cache_pool[0, 0, 0, :] = mx.array([1.0, 1.0, 1.0, 1.0], dtype=dtype)
    # Position 1 (token_slot 1 on physical_page 0) - Score will be 4.0 after scale
    py_k_cache_pool[0, 1, 0, :] = mx.array([2.0, 2.0, 2.0, 2.0], dtype=dtype)

    # K-vectors for logical block 1 (positions 2, 3)
    # Position 2 (token_slot 0 on physical_page 1) - Score will be 10.0 after scale (should be max)
    py_k_cache_pool[1, 0, 0, :] = mx.array([5.0, 5.0, 5.0, 5.0], dtype=dtype)
    # Position 3 (token_slot 1 on physical_page 1) - Score will be 6.0 after scale
    py_k_cache_pool[1, 1, 0, :] = mx.array([3.0, 3.0, 3.0, 3.0], dtype=dtype)

    # Note: Position 4 would be in logical block 2 (which is beyond page table limit)
    # We don't need to set values for it, as it should not be accessed

    # 3. V-Cache Pool with values to verify that these are aggregated correctly
    py_v_cache_pool = mx.zeros_like(py_k_cache_pool)
    # Set up V-cache with values to verify correct aggregation
    # Position mapping: physical_page = logical_block_idx, slot = hist_pos % cfg_tokens_per_page
    py_v_cache_pool[0, 0, 0, :] = mx.array([10.0, 20.0, 30.0, 40.0], dtype=dtype)  # Position 0 (page 0, slot 0)
    py_v_cache_pool[0, 1, 0, :] = mx.array([50.0, 60.0, 70.0, 80.0], dtype=dtype)  # Position 1 (page 0, slot 1)
    py_v_cache_pool[1, 0, 0, :] = mx.array(
        [500.0, 600.0, 700.0, 800.0], dtype=dtype
    )  # Position 2 (page 1, slot 0) - highest K score
    py_v_cache_pool[1, 1, 0, :] = mx.array([100.0, 200.0, 300.0, 400.0], dtype=dtype)  # Position 3 (page 1, slot 1)
    # Position 4 would be in logical block 2 (beyond page table limit) - not set

    # 4. Page Table: Maps logical blocks 0,1 to physical pages 0,1
    # Limited to max_logical_pages_per_seq_in_pagetable = 2
    py_page_table = mx.array([[0, 1]], dtype=mx.uint32)  # Shape (1, 2)

    # 5. Sequence Lengths: One sequence with plenty of tokens
    py_sequence_lengths = mx.array([10], dtype=mx.int32)  # More than query_token_offset

    # 6. Query to Sequence Map: Single query maps to sequence 0
    py_query_to_seq_map = mx.zeros(1, dtype=mx.int32)

    # 7. Query Token Offset: Set to 5, so history would be positions 0-4
    py_query_token_offset = mx.array([query_token_offset], dtype=mx.int32)

    mx.eval(py_queries)
    mx.eval(py_k_cache_pool)
    mx.eval(py_v_cache_pool)
    mx.eval(py_page_table)
    mx.eval(py_sequence_lengths)
    mx.eval(py_query_to_seq_map)
    mx.eval(py_query_token_offset)

    # --- Run paged attention ---
    output_arr = paged_attention(
        py_queries,
        py_k_cache_pool,
        py_v_cache_pool,
        py_page_table,
        py_sequence_lengths,
        py_query_to_seq_map,
        py_query_token_offset,
        use_fused_kernel=True,
    )
    mx.eval(output_arr)

    # --- Calculate expected output (Python reference) ---
    # Scale factor for dot product
    denominator = mx.sqrt(mx.array(float(cfg_head_dim))).item()
    assert isinstance(denominator, float)
    py_scale = 1.0 / denominator

    # Expected scores for positions 0-3 (logical blocks 0 and 1)
    score0 = 4.0 * py_scale  # = 4 * 0.5 = 2.0
    score1 = 8.0 * py_scale  # = 8 * 0.5 = 4.0
    score2 = 20.0 * py_scale  # = 20 * 0.5 = 10.0 (should be max)
    score3 = 12.0 * py_scale  # = 12 * 0.5 = 6.0

    # Max score should be from position 2 (score2 = 10.0)
    # Compute expected full attention output with softmax over valid positions

    # Find maximum score from valid positions 0-3 (within page table limit)
    true_max_score = max(score0, score1, score2, score3)  # Should be score2 = 10.0

    # Calculate exp(score - max_score) with clamp for each position
    exp_score0 = mx.exp(mx.maximum(score0 - true_max_score, -16.0)).item()
    exp_score1 = mx.exp(mx.maximum(score1 - true_max_score, -16.0)).item()
    exp_score2 = mx.exp(mx.maximum(score2 - true_max_score, -16.0)).item()
    exp_score3 = mx.exp(mx.maximum(score3 - true_max_score, -16.0)).item()

    # Calculate sum of exp scores
    true_sum_exp_score = exp_score0 + exp_score1 + exp_score2 + exp_score3

    # Calculate softmax probabilities
    prob0 = exp_score0 / true_sum_exp_score
    prob1 = exp_score1 / true_sum_exp_score
    prob2 = exp_score2 / true_sum_exp_score
    prob3 = exp_score3 / true_sum_exp_score

    # Get V-vectors for each position
    v_vec0 = py_v_cache_pool[0, 0, 0, :].astype(mx.float32)
    v_vec1 = py_v_cache_pool[0, 1, 0, :].astype(mx.float32)
    v_vec2 = py_v_cache_pool[1, 0, 0, :].astype(mx.float32)
    v_vec3 = py_v_cache_pool[1, 1, 0, :].astype(mx.float32)

    # Calculate weighted sum of V-vectors
    expected_V_output_py = v_vec0 * prob0 + v_vec1 * prob1 + v_vec2 * prob2 + v_vec3 * prob3

    # Output is now full attention format [num_q_threads, cfg_head_dim]
    num_q_threads = 1  # For this test
    expected_V_output = expected_V_output_py.astype(dtype).reshape(num_q_threads, cfg_head_dim)
    expected_output_shape = (num_q_threads, cfg_head_dim)

    logger.info(f"Test: {test_history_scan_stops_at_page_table_limit.__name__} dtype={dtype}")
    logger.info(
        f"  max_logical_pages_per_seq = {max_logical_pages_per_seq_in_pagetable}, query_token_offset = {query_token_offset}"
    )
    logger.info("  Scores for positions in valid blocks:")
    logger.info(f"    Position 0 (block 0, slot 0): {score0}")
    logger.info(f"    Position 1 (block 0, slot 1): {score1}")
    logger.info(f"    Position 2 (block 1, slot 0): {score2} (max)")
    logger.info(f"    Position 3 (block 1, slot 1): {score3}")
    logger.info(f"  Softmax probabilities = [{prob0}, {prob1}, {prob2}, {prob3}]")
    logger.info(f"  Expected V output = {expected_V_output}")
    logger.info(f"  Actual V output = {output_arr}")

    # Verify results
    assert output_arr.shape == expected_output_shape, (
        f"Output shape {output_arr.shape} does not match expected {expected_output_shape}"
    )
    assert output_arr.dtype == dtype, f"Output dtype {output_arr.dtype} does not match {dtype}"
    # Verify the V output contains only information from valid page table blocks
    assert mx.allclose(output_arr, expected_V_output, atol=1e-2, rtol=1e-2), (
        "Output values do not match expected weighted sum from valid page table blocks"
    )

    logger.info("test_history_scan_stops_at_page_table_limit PASSED")
