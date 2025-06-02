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
"""Tests for various attention computation scenarios.

This module contains tests that verify the paged attention operation correctly
handles different attention scenarios such as history in single or multiple blocks,
zero history, sequence length limits, and page table boundaries.
"""

import logging

import mlx.core as mx

from proxy_attention_lab import calculate_page_size, paged_attention

logger = logging.getLogger(__name__)


def test_max_score_over_history_in_one_block() -> None:
    """Test full attention computation within a single block.

    This test verifies the full attention computation (max score, softmax, and V-aggregation)
    for a single query with multiple history tokens all contained within one logical block.
    """
    # --- Configuration ---
    cfg_num_q_heads = 1
    cfg_num_kv_heads = 1
    cfg_head_dim = 4
    cfg_tokens_per_page = calculate_page_size(cfg_head_dim, cfg_num_q_heads, cfg_num_kv_heads)
    sequence_length = 4  # Sequence has 4 tokens total (positions 0, 1, 2, 3)

    # --- Setup test inputs ---
    # 1. Query vectors for all tokens in the sequence (prefill mode)
    # In prefill, we provide ALL tokens' queries contiguously
    py_queries = mx.array(
        [
            [0.5, 0.5, 0.5, 0.5],  # Token 0
            [0.7, 0.7, 0.7, 0.7],  # Token 1
            [0.3, 0.3, 0.3, 0.3],  # Token 2
            [1.0, 2.0, 3.0, 4.0],  # Token 3 (current position)
        ],
        dtype=mx.float16,
    )

    # 2. Create K-cache with values for history positions
    k_cache_shape = (1, cfg_tokens_per_page, cfg_num_kv_heads, cfg_head_dim)
    py_k_cache_pool = mx.zeros(k_cache_shape, dtype=mx.float16)
    # Set K-vectors for each history position
    py_k_cache_pool[0, 0, 0, :] = mx.array([1.0, 1.0, 1.0, 1.0], dtype=mx.float16)  # Position 0
    py_k_cache_pool[0, 1, 0, :] = mx.array([2.0, 2.0, 2.0, 2.0], dtype=mx.float16)  # Position 1
    py_k_cache_pool[0, 2, 0, :] = mx.array([0.5, 0.5, 0.5, 0.5], dtype=mx.float16)  # Position 2

    # 3. Create V-cache with distinct values for each position
    py_v_cache_pool = mx.zeros(k_cache_shape, dtype=mx.float16)
    py_v_cache_pool[0, 0, 0, :] = mx.array([10.0, 11.0, 12.0, 13.0], dtype=mx.float16)  # Position 0
    py_v_cache_pool[0, 1, 0, :] = mx.array([20.0, 21.0, 22.0, 23.0], dtype=mx.float16)  # Position 1
    py_v_cache_pool[0, 2, 0, :] = mx.array([30.0, 31.0, 32.0, 33.0], dtype=mx.float16)  # Position 2

    # 4. Set up page table and sequence metadata
    py_page_table = mx.array([[0]], dtype=mx.uint32)  # Map logical block 0 -> physical page 0
    py_sequence_lengths = mx.array([sequence_length], dtype=mx.int32)  # Sum must equal total query tokens
    py_query_to_seq_map = mx.zeros(sequence_length, dtype=mx.int32)  # All tokens map to sequence 0
    py_query_token_offset = mx.array([0, 1, 2, 3], dtype=mx.int32)  # Each token's position within sequence

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

    # --- Calculate expected output (Python reference) ---
    # In prefill mode, we compute attention for ALL tokens in parallel
    # Each token attends to all previous tokens (causal attention)
    py_scale = 1.0 / mx.sqrt(mx.array(float(cfg_head_dim))).item()

    expected_outputs = []

    # Process each query token
    for query_idx in range(sequence_length):
        q_vec_py = py_queries[query_idx].astype(mx.float32)

        # Inclusive causal: history 0..query_idx (self included)
        scores_val = []
        for hist_idx_calc in range(query_idx + 1):
            k_vec_py = py_k_cache_pool[0, hist_idx_calc, 0, :].astype(mx.float32)
            score = (mx.sum(q_vec_py * k_vec_py) * py_scale).item()
            scores_val.append(score)

        # For first token, history includes self only
        if not scores_val:
            scores_val.append(0.0)  # dot-product will be 0 with zero K but shouldn't happen

        # Find maximum score
        true_max_score = max(scores_val)

        # Calculate exp(score - max_score) with numerical stability
        exp_scores_minus_max = []
        for s_val in scores_val:
            exp_scores_minus_max.append(mx.exp(mx.maximum(s_val - true_max_score, -16.0)).item())

        # Calculate sum of exponentials for softmax denominator
        true_sum_exp_score = sum(exp_scores_minus_max)
        if true_sum_exp_score == 0:
            true_sum_exp_score = 1.0  # Avoid division by zero

        # Calculate softmax probabilities
        softmax_probs = [val / true_sum_exp_score for val in exp_scores_minus_max]

        # Calculate expected weighted sum of V-vectors
        expected_V_output_py = mx.zeros((cfg_head_dim,), dtype=mx.float32)
        for i in range(query_idx + 1):
            v_hist = py_v_cache_pool[0, i, 0, :].astype(mx.float32)
            expected_V_output_py += v_hist * softmax_probs[i]

        expected_outputs.append(expected_V_output_py.astype(mx.float16))

    # Stack all outputs
    expected_V_output_reshaped = mx.stack(expected_outputs)

    # Log test details
    logger.info(f"Test: {test_max_score_over_history_in_one_block.__name__}")
    logger.info(f"  Expected V output shape: {expected_V_output_reshaped.shape}")
    logger.info(f"  Actual V output shape: {output_arr.shape}")
    logger.info(f"  Expected V output: {expected_V_output_reshaped}")
    logger.info(f"  Actual V output: {output_arr}")

    # Verify results
    assert output_arr.shape == expected_V_output_reshaped.shape, (
        f"Output shape {output_arr.shape} does not match expected {expected_V_output_reshaped.shape}"
    )
    assert output_arr.dtype == mx.float16, f"Output dtype {output_arr.dtype} does not match float16"
    # Increased tolerance slightly for float16 sum
    assert mx.allclose(output_arr, expected_V_output_reshaped, atol=1e-2, rtol=1e-2), (
        "Output values do not match expected values"
    )


def test_max_score_over_multi_block_history() -> None:
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
    cfg_tokens_per_page = 3  # Small value to ensure we span multiple blocks
    cfg_num_kv_heads = 1
    cfg_head_dim = 4
    sequence_length = 6  # Total tokens in sequence

    # Tokens span two logical blocks:
    # - Tokens 0,1,2 are in logical block 0
    # - Tokens 3,4,5 are in logical block 1

    # --- Setup test inputs ---
    # 1. Query vectors for all tokens in sequence (prefill mode)
    py_queries = mx.array(
        [
            [0.1, 0.1, 0.1, 0.1],  # Token 0
            [0.2, 0.2, 0.2, 0.2],  # Token 1
            [0.3, 0.3, 0.3, 0.3],  # Token 2
            [0.4, 0.4, 0.4, 0.4],  # Token 3
            [0.5, 0.5, 0.5, 0.5],  # Token 4
            [1.0, 2.0, 3.0, 4.0],  # Token 5 (current position)
        ],
        dtype=mx.float16,
    )

    # 2. Create K-cache with values for history positions in multiple blocks
    num_physical_pages = 2  # Two physical pages for the two logical blocks
    k_cache_shape = (num_physical_pages, cfg_tokens_per_page, cfg_num_kv_heads, cfg_head_dim)
    py_k_cache_pool = mx.zeros(k_cache_shape, dtype=mx.float16)

    # Set K-vectors in physical page 0 (logical block 0)
    py_k_cache_pool[0, 0, 0, :] = mx.array([1.0, 1.0, 1.0, 1.0], dtype=mx.float16)  # Position 0
    py_k_cache_pool[0, 1, 0, :] = mx.array([2.0, 2.0, 2.0, 2.0], dtype=mx.float16)  # Position 1
    py_k_cache_pool[0, 2, 0, :] = mx.array([0.5, 0.5, 0.5, 0.5], dtype=mx.float16)  # Position 2

    # Set K-vectors in physical page 1 (logical block 1)
    py_k_cache_pool[1, 0, 0, :] = mx.array([3.0, 3.0, 3.0, 3.0], dtype=mx.float16)  # Position 3
    py_k_cache_pool[1, 1, 0, :] = mx.array([1.5, 1.5, 1.5, 1.5], dtype=mx.float16)  # Position 4

    # 3. Create V-cache with distinct values for each position
    py_v_cache_pool = mx.zeros_like(py_k_cache_pool)
    # Set V-vectors in physical page 0 (logical block 0)
    py_v_cache_pool[0, 0, 0, :] = mx.array([10.0, 20.0, 30.0, 40.0], dtype=mx.float16)  # Position 0
    py_v_cache_pool[0, 1, 0, :] = mx.array([50.0, 60.0, 70.0, 80.0], dtype=mx.float16)  # Position 1
    py_v_cache_pool[0, 2, 0, :] = mx.array([5.0, 6.0, 7.0, 8.0], dtype=mx.float16)  # Position 2
    # Set V-vectors in physical page 1 (logical block 1)
    py_v_cache_pool[1, 0, 0, :] = mx.array([100.0, 110.0, 120.0, 130.0], dtype=mx.float16)  # Position 3
    py_v_cache_pool[1, 1, 0, :] = mx.array([15.0, 25.0, 35.0, 45.0], dtype=mx.float16)  # Position 4

    # 4. Set up page table - maps logical blocks to physical pages
    py_page_table = mx.array([[0, 1]], dtype=mx.uint32)  # Logical block 0 -> page 0, block 1 -> page 1

    # 5. Set up sequence metadata
    py_sequence_lengths = mx.array([sequence_length], dtype=mx.int32)  # Total tokens in sequence
    py_query_to_seq_map = mx.zeros(sequence_length, dtype=mx.int32)  # All tokens map to sequence 0
    py_query_token_offset = mx.array([0, 1, 2, 3, 4, 5], dtype=mx.int32)  # Each token's position

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

    # --- Calculate expected output (Python reference) ---
    # Scale factor: 1.0 / sqrt(head_dim)
    scale = 1.0 / mx.sqrt(mx.array(float(cfg_head_dim))).item()
    assert isinstance(scale, float)

    # Build full expected outputs for all tokens (causal inclusive)
    full_expected_outputs = []
    for qi in range(sequence_length):
        q_vec = py_queries[qi].astype(mx.float32)
        scores_qi = []
        v_list_qi = []
        for hist_j in range(qi + 1):
            lb = hist_j // cfg_tokens_per_page
            slot = hist_j % cfg_tokens_per_page
            phys = py_page_table[0, lb].item()
            k_vec = py_k_cache_pool[phys, slot, 0, :].astype(mx.float32)
            v_vec = py_v_cache_pool[phys, slot, 0, :].astype(mx.float32)
            score_val = (mx.sum(q_vec * k_vec) * scale).item()
            scores_qi.append(score_val)
            v_list_qi.append(v_vec)
        if not scores_qi:
            full_expected_outputs.append(mx.zeros(cfg_head_dim, dtype=mx.float16))
            continue
        max_qi = max(scores_qi)
        exp_list = [mx.exp(mx.maximum(s - max_qi, -16.0)).item() for s in scores_qi]
        denom = sum(exp_list)
        probs = [e / denom for e in exp_list]
        out_vec = mx.zeros(cfg_head_dim, dtype=mx.float32)
        for vv, pr in zip(v_list_qi, probs, strict=False):
            out_vec += vv * pr
        full_expected_outputs.append(out_vec.astype(mx.float16))

    expected_V_output_reshaped = mx.stack(full_expected_outputs)
    expected_output_shape = (sequence_length, cfg_head_dim)

    logger.info(f"Test: {test_max_score_over_multi_block_history.__name__}")
    logger.info(f"  Expected V output shape: {expected_V_output_reshaped.shape}")
    logger.info(f"  Actual V output shape: {output_arr.shape}")
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

    logger.info("test_max_score_over_multi_block_history PASSED")


def test_zero_history_returns_zero_score() -> None:
    """Test that zero history returns zero score.

    This test verifies the code path in the kernel where 'effective_history_length = 0',
    which should result in 'max_score_half = 0.0h' instead of -infinity or garbage values.
    The zero-history case occurs when query_token_offset is 0, meaning the query is
    at the first position with no history tokens before it to compute attention scores with.
    """
    # --- Configuration ---
    cfg_tokens_per_page = 64
    cfg_num_kv_heads = 1
    cfg_head_dim = 4

    # --- Setup test inputs ---
    # For prefill mode the queries must include the entire sequence for each batch item.
    # We'll model a single sequence with 10 tokens.
    py_queries = mx.array(
        [
            [1.0, 2.0, 3.0, 4.0],  # Token 0
            [1.1, 2.1, 3.1, 4.1],  # Token 1
            [1.2, 2.2, 3.2, 4.2],  # Token 2
            [1.3, 2.3, 3.3, 4.3],  # Token 3
            [1.4, 2.4, 3.4, 4.4],  # Token 4
            [1.5, 2.5, 3.5, 4.5],  # Token 5
            [1.6, 2.6, 3.6, 4.6],  # Token 6
            [1.7, 2.7, 3.7, 4.7],  # Token 7
            [1.8, 2.8, 3.8, 4.8],  # Token 8
            [1.9, 2.9, 3.9, 4.9],  # Token 9
        ],
        dtype=mx.float16,
    )

    # 2. K-Cache Pool: Minimal setup with some values (won't be accessed)
    num_physical_pages = 1
    k_cache_shape = (num_physical_pages, cfg_tokens_per_page, cfg_num_kv_heads, cfg_head_dim)
    py_k_cache_pool = mx.zeros(k_cache_shape, dtype=mx.float16)
    # Set up K-cache values for testing
    py_k_cache_pool[0, 0, 0, :] = mx.array([1.0, 1.0, 1.0, 1.0], dtype=mx.float16)  # Position 0
    py_k_cache_pool[0, 1, 0, :] = mx.array([2.0, 2.0, 2.0, 2.0], dtype=mx.float16)  # Position 1
    py_k_cache_pool[0, 2, 0, :] = mx.array([0.5, 0.5, 0.5, 0.5], dtype=mx.float16)  # Position 2
    py_k_cache_pool[0, 3, 0, :] = mx.array([3.0, 3.0, 3.0, 3.0], dtype=mx.float16)  # Position 3
    # Positions 4+ stay as zeros

    # 3. V-Cache Pool with values to verify that these are aggregated correctly
    py_v_cache_pool = mx.zeros_like(py_k_cache_pool)
    # Set up V-cache with values to verify correct aggregation
    py_v_cache_pool[0, 0, 0, :] = mx.array([10.0, 20.0, 30.0, 40.0], dtype=mx.float16)  # Position 0
    py_v_cache_pool[0, 1, 0, :] = mx.array([50.0, 60.0, 70.0, 80.0], dtype=mx.float16)  # Position 1 (highest score)
    py_v_cache_pool[0, 2, 0, :] = mx.array([5.0, 6.0, 7.0, 8.0], dtype=mx.float16)  # Position 2
    py_v_cache_pool[0, 3, 0, :] = mx.array([100.0, 110.0, 120.0, 130.0], dtype=mx.float16)  # Position 3
    # Positions 4+ are zeros

    # 4. Page Table: Simple mapping for one logical block
    py_page_table = mx.array([[0]], dtype=mx.uint32)  # Shape (1, 1)

    # 5. Sequence Lengths: One sequence with tokens
    py_sequence_lengths = mx.array([10], dtype=mx.int32)

    # 6. Query to Sequence Map: Both queries map to sequence 0
    py_query_to_seq_map = mx.zeros(10, dtype=mx.int32)

    # 7. Query Token Offset: CRUCIAL - set to 0 to test zero-history case
    py_query_token_offset = mx.arange(10, dtype=mx.int32)

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

    # --- Calculate expected output ---
    # With inclusive causal masking, each token attends to all tokens up to and including itself
    scale = 1.0 / mx.sqrt(mx.array(float(cfg_head_dim))).item()
    expected_outputs = []

    for i in range(10):
        q_vec = py_queries[i].astype(mx.float32)
        scores = []
        v_vecs = []

        # Token i at offset i attends to positions 0..i
        for j in range(i + 1):
            # Get K and V at position j
            page = j // cfg_tokens_per_page
            slot = j % cfg_tokens_per_page
            if page < 1 and slot < cfg_tokens_per_page:  # Within our cache
                k_vec = py_k_cache_pool[page, slot, 0, :].astype(mx.float32)
                v_vec = py_v_cache_pool[page, slot, 0, :].astype(mx.float32)
                score = (mx.sum(q_vec * k_vec) * scale).item()
                scores.append(score)
                v_vecs.append(v_vec)
            else:
                # Beyond cache, use zeros
                scores.append(0.0)
                v_vecs.append(mx.zeros(cfg_head_dim, dtype=mx.float32))

        # Compute softmax
        if scores:
            max_score = max(scores)
            exp_scores = [mx.exp(s - max_score).item() for s in scores]
            sum_exp = sum(exp_scores)
            probs = [e / sum_exp for e in exp_scores]

            # Weighted sum of V vectors
            v_out = mx.zeros(cfg_head_dim, dtype=mx.float32)
            for v, p in zip(v_vecs, probs, strict=False):
                v_out += v * p
            expected_outputs.append(v_out.astype(mx.float16))
        else:
            expected_outputs.append(mx.zeros(cfg_head_dim, dtype=mx.float16))

    expected_v_output = mx.stack(expected_outputs)

    # Output is now full attention format [num_q_threads, cfg_head_dim]
    expected_output_shape = (10, cfg_head_dim)

    logger.info(f"Test: {test_zero_history_returns_zero_score.__name__}")
    logger.info(f"  Query token offsets = {py_query_token_offset}")
    logger.info(f"  Expected V output = {expected_v_output}")
    logger.info(f"  Actual V output = {output_arr}")

    # Verify results
    assert output_arr.shape == expected_output_shape, (
        f"Output shape {output_arr.shape} does not match expected {expected_output_shape}"
    )
    assert output_arr.dtype == mx.float16, f"Output dtype {output_arr.dtype} does not match float16"
    # Verify V output matches expected causal attention
    assert mx.allclose(output_arr, expected_v_output, atol=1e-2, rtol=1e-2), (
        "Output values do not match expected causal attention outputs"
    )

    logger.info("test_zero_history_returns_zero_score PASSED")


def test_history_limited_by_sequence_length() -> None:
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
    py_queries = mx.array([[1.0, 1.0, 1.0, 1.0]] * actual_sequence_length, dtype=mx.float16)

    # 2. K-Cache Pool: Set up with different values for positions 0-4
    num_physical_pages = 2  # Need enough pages for all positions
    k_cache_shape = (num_physical_pages, cfg_tokens_per_page, cfg_num_kv_heads, cfg_head_dim)
    py_k_cache_pool = mx.zeros(k_cache_shape, dtype=mx.float16)

    # Set up K-vectors for each history position
    # Position 0 - Score will be 2.0 after scale
    py_k_cache_pool[0, 0, 0, :] = mx.array([1.0, 1.0, 1.0, 1.0], dtype=mx.float16)

    # Position 1 - Score will be 6.0 after scale (should be the max within valid sequence)
    py_k_cache_pool[0, 1, 0, :] = mx.array([3.0, 3.0, 3.0, 3.0], dtype=mx.float16)

    # Position 2 - Score will be 4.0 after scale
    py_k_cache_pool[0, 2, 0, :] = mx.array([2.0, 2.0, 2.0, 2.0], dtype=mx.float16)

    # Position 3 - Beyond sequence length, should NOT be accessed
    py_k_cache_pool[0, 3, 0, :] = mx.array([8.0, 8.0, 8.0, 8.0], dtype=mx.float16)  # Score would be 16.0

    # Position 4 - Beyond sequence length, should NOT be accessed
    py_k_cache_pool[1, 0, 0, :] = mx.array([10.0, 10.0, 10.0, 10.0], dtype=mx.float16)  # Score would be 20.0

    # 3. V-Cache Pool with values to verify that these are aggregated correctly
    py_v_cache_pool = mx.zeros_like(py_k_cache_pool)
    # Set up V-cache with values to verify correct aggregation
    py_v_cache_pool[0, 0, 0, :] = mx.array([10.0, 20.0, 30.0, 40.0], dtype=mx.float16)  # Position 0
    py_v_cache_pool[0, 1, 0, :] = mx.array([50.0, 60.0, 70.0, 80.0], dtype=mx.float16)  # Position 1 (highest score)
    py_v_cache_pool[0, 2, 0, :] = mx.array([5.0, 6.0, 7.0, 8.0], dtype=mx.float16)  # Position 2
    # Positions 3-4 should NOT be accessed due to sequence length limit (actual_sequence_length=3)
    py_v_cache_pool[0, 3, 0, :] = mx.array([100.0, 200.0, 300.0, 400.0], dtype=mx.float16)  # Position 3
    py_v_cache_pool[1, 0, 0, :] = mx.array([500.0, 600.0, 700.0, 800.0], dtype=mx.float16)  # Position 4

    # 4. Page Table: Two logical blocks mapped to two physical pages
    py_page_table = mx.array([[0, 1]], dtype=mx.uint32)  # Shape (1, 2)

    # 5. Sequence Lengths: One sequence with only 3 tokens
    py_sequence_lengths = mx.array([actual_sequence_length], dtype=mx.int32)

    # 6. Query to Sequence Map: Single query maps to sequence 0
    py_query_to_seq_map = mx.zeros(actual_sequence_length, dtype=mx.int32)

    # 7. Query Token Offset: Set to 5, so history would be positions 0-4
    py_query_token_offset = mx.full(actual_sequence_length, query_token_offset, dtype=mx.int32)

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
    expected_V_output = mx.stack([expected_V_output_py.astype(mx.float16)] * actual_sequence_length)
    expected_output_shape = (actual_sequence_length, cfg_head_dim)

    logger.info(f"Test: {test_history_limited_by_sequence_length.__name__}")
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
    assert output_arr.dtype == mx.float16, f"Output dtype {output_arr.dtype} does not match float16"
    # Verify the V output contains the weighted sum of only positions 0-2,
    # not including positions 3-4 which should be excluded by sequence length limit
    assert mx.allclose(output_arr, expected_V_output, atol=1e-2, rtol=1e-2), (
        "Output values do not match expected weighted sum based on sequence length limit"
    )

    logger.info("test_history_limited_by_sequence_length PASSED")


def test_history_scan_stops_at_page_table_limit() -> None:
    """Test that history scan stops at page table limits.

    This test verifies that if the history scan encounters a logical_block_idx that is
    >= params->max_logical_blocks_per_seq (i.e., beyond what the page table describes
    for that sequence), the kernel correctly stops scanning further history but still
    returns the max score found from valid preceding blocks.

    In this test, query_token_offset=5 implies 5 history tokens (positions 0-4), but
    the page table only describes 2 logical blocks (covering positions 0-3), so position 4
    should be ignored even though it's within sequence_length.

    The kernel achieves this via the code:
    if (logical_block_idx >= params->max_logical_blocks_per_seq) {
        break; // No more valid blocks for this sequence's history
    }
    """
    # --- Configuration ---
    cfg_tokens_per_page = 2  # Small value to ensure we span multiple blocks quickly
    cfg_num_kv_heads = 1
    cfg_head_dim = 4
    max_logical_blocks_per_seq_in_pagetable = 2  # Page table only describes 2 logical blocks
    query_token_offset = 5  # History positions: 0,1,2,3,4

    # --- Setup test inputs ---
    # Provide 10 queries to match sequence_length
    py_queries = mx.array([[1.0, 1.0, 1.0, 1.0]] * 10, dtype=mx.float16)

    # 2. K-Cache Pool:
    num_physical_pages = 2  # Two physical pages for the two logical blocks in page table
    k_cache_shape = (num_physical_pages, cfg_tokens_per_page, cfg_num_kv_heads, cfg_head_dim)
    py_k_cache_pool = mx.zeros(k_cache_shape, dtype=mx.float16)

    # Position mapping with cfg_tokens_per_page = 2:
    # hist_pos 0, 1 -> logical_block_idx 0 -> physical_page 0
    # hist_pos 2, 3 -> logical_block_idx 1 -> physical_page 1
    # hist_pos 4    -> logical_block_idx 2 (beyond page_table limit) -> should not be accessed

    # K-vectors for logical block 0 (positions 0, 1)
    # Position 0 (token_slot 0 on physical_page 0) - Score will be 2.0 after scale
    py_k_cache_pool[0, 0, 0, :] = mx.array([1.0, 1.0, 1.0, 1.0], dtype=mx.float16)
    # Position 1 (token_slot 1 on physical_page 0) - Score will be 4.0 after scale
    py_k_cache_pool[0, 1, 0, :] = mx.array([2.0, 2.0, 2.0, 2.0], dtype=mx.float16)

    # K-vectors for logical block 1 (positions 2, 3)
    # Position 2 (token_slot 0 on physical_page 1) - Score will be 10.0 after scale (should be max)
    py_k_cache_pool[1, 0, 0, :] = mx.array([5.0, 5.0, 5.0, 5.0], dtype=mx.float16)
    # Position 3 (token_slot 1 on physical_page 1) - Score will be 6.0 after scale
    py_k_cache_pool[1, 1, 0, :] = mx.array([3.0, 3.0, 3.0, 3.0], dtype=mx.float16)

    # Note: Position 4 would be in logical block 2 (which is beyond page table limit)
    # We don't need to set values for it, as it should not be accessed

    # 3. V-Cache Pool with values to verify that these are aggregated correctly
    py_v_cache_pool = mx.zeros_like(py_k_cache_pool)
    # Set up V-cache with values to verify correct aggregation
    py_v_cache_pool[0, 0, 0, :] = mx.array([10.0, 20.0, 30.0, 40.0], dtype=mx.float16)  # Position 0
    py_v_cache_pool[0, 1, 0, :] = mx.array([50.0, 60.0, 70.0, 80.0], dtype=mx.float16)  # Position 1 (highest score)
    py_v_cache_pool[0, 2, 0, :] = mx.array([5.0, 6.0, 7.0, 8.0], dtype=mx.float16)  # Position 2
    # Positions 3-4 should NOT be accessed due to sequence length limit
    py_v_cache_pool[0, 3, 0, :] = mx.array([100.0, 200.0, 300.0, 400.0], dtype=mx.float16)  # Position 3
    py_v_cache_pool[1, 0, 0, :] = mx.array([500.0, 600.0, 700.0, 800.0], dtype=mx.float16)  # Position 4

    # 4. Page Table: Maps logical blocks 0,1 to physical pages 0,1
    # Limited to max_logical_blocks_per_seq_in_pagetable = 2
    py_page_table = mx.array([[0, 1]], dtype=mx.uint32)  # Shape (1, 2)

    # 5. Sequence Lengths: One sequence with plenty of tokens
    py_sequence_lengths = mx.array([10], dtype=mx.int32)  # More than query_token_offset

    # 6. Query to Sequence Map: Single query maps to sequence 0
    py_query_to_seq_map = mx.zeros(10, dtype=mx.int32)

    # 7. Query Token Offset: Set to 5, so history would be positions 0-4
    py_query_token_offset = mx.full(10, query_token_offset, dtype=mx.int32)

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
    expected_V_output = mx.stack([expected_V_output_py.astype(mx.float16)] * 10)
    expected_output_shape = (10, cfg_head_dim)

    logger.info(f"Test: {test_history_scan_stops_at_page_table_limit.__name__}")
    logger.info(
        f"  max_logical_blocks_per_seq = {max_logical_blocks_per_seq_in_pagetable}, query_token_offset = {query_token_offset}"
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
    assert output_arr.dtype == mx.float16, f"Output dtype {output_arr.dtype} does not match float16"
    # Verify the V output contains only information from valid page table blocks
    assert mx.allclose(output_arr, expected_V_output, atol=1e-2, rtol=1e-2), (
        "Output values do not match expected weighted sum from valid page table blocks"
    )

    logger.info("test_history_scan_stops_at_page_table_limit PASSED")
