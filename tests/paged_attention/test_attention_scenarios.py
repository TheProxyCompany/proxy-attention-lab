import logging

import mlx.core as mx

from proxy_attention_lab import paged_attention

logger = logging.getLogger(__name__)


def test_max_score_over_history_in_one_block():
    """
    Tests the full attention computation (max score, softmax, and V-aggregation)
    for a single item with multiple history tokens in one block.
    """
    # --- Config ---
    num_q_threads = 1
    cfg_tokens_per_page = 64
    cfg_num_kv_heads = 1
    cfg_head_dim = 4
    current_position = 3

    # --- Inputs ---
    py_queries = mx.array([[1.0, 2.0, 3.0, 4.0]], dtype=mx.float16)

    k_cache_shape = (1, cfg_tokens_per_page, cfg_num_kv_heads, cfg_head_dim)
    py_k_cache_pool = mx.zeros(k_cache_shape, dtype=mx.float16)
    py_k_cache_pool[0, 0, 0, :] = mx.array([1.0, 1.0, 1.0, 1.0], dtype=mx.float16)
    py_k_cache_pool[0, 1, 0, :] = mx.array([2.0, 2.0, 2.0, 2.0], dtype=mx.float16)
    py_k_cache_pool[0, 2, 0, :] = mx.array([0.5, 0.5, 0.5, 0.5], dtype=mx.float16)

    py_v_cache_pool = mx.zeros(k_cache_shape, dtype=mx.float16)
    py_v_cache_pool[0, 0, 0, :] = mx.array([10.0, 11.0, 12.0, 13.0], dtype=mx.float16)
    py_v_cache_pool[0, 1, 0, :] = mx.array([20.0, 21.0, 22.0, 23.0], dtype=mx.float16)
    py_v_cache_pool[0, 2, 0, :] = mx.array([30.0, 31.0, 32.0, 33.0], dtype=mx.float16)

    py_page_table = mx.array([[0]], dtype=mx.uint32)
    py_sequence_lengths = mx.array([cfg_tokens_per_page], dtype=mx.int32)
    py_query_to_seq_map = mx.zeros(num_q_threads, dtype=mx.int32)
    py_query_token_offset = mx.array([current_position], dtype=mx.int32)

    # --- Call the kernel ---
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
    py_scale = 1.0 / mx.sqrt(mx.array(float(cfg_head_dim))).item()
    scores_val = []
    q_vec_py = py_queries[0].astype(mx.float32)
    for hist_idx_calc in range(current_position):
        k_vec_py = py_k_cache_pool[0, hist_idx_calc, 0, :].astype(mx.float32)
        score = (mx.sum(q_vec_py * k_vec_py) * py_scale).item()
        scores_val.append(score)

    true_max_score = -float("inf")
    for s_val in scores_val:
        if s_val > true_max_score:
            true_max_score = s_val
    if not scores_val:
        true_max_score = 0.0  # Handle empty history case for max

    exp_scores_minus_max = []
    for s_val in scores_val:
        exp_scores_minus_max.append(mx.exp(mx.maximum(s_val - true_max_score, -16.0)).item())

    true_sum_exp_score = sum(exp_scores_minus_max)
    if true_sum_exp_score == 0 and not scores_val:  # if history was empty, sum_exp is 0
        pass
    elif true_sum_exp_score == 0:  # if history non-empty but all scores led to sum_exp 0 (e.g. all -inf)
        true_sum_exp_score = 1.0  # Avoid division by zero, effectively making probs 0 if scores were -inf

    softmax_probs = []
    if true_sum_exp_score != 0:
        for val in exp_scores_minus_max:
            softmax_probs.append(val / true_sum_exp_score)
    else:  # All scores were effectively -infinity or history empty
        softmax_probs = [0.0] * len(scores_val) if scores_val else []

    expected_V_output_py = mx.zeros((cfg_head_dim,), dtype=mx.float32)
    if softmax_probs:  # Only aggregate if there are probs
        for i in range(current_position):
            v_hist = py_v_cache_pool[0, i, 0, :].astype(mx.float32)
            expected_V_output_py += v_hist * softmax_probs[i]

    expected_V_output_reshaped = expected_V_output_py.astype(mx.float16).reshape(num_q_threads, cfg_head_dim)

    logger.info(f"Test ('{test_max_score_over_history_in_one_block.__name__}'):")
    logger.info(f"  Scores: {scores_val}")
    logger.info(f"  Max Score: {true_max_score}")
    logger.info(f"  Sum Exp Score: {true_sum_exp_score}")
    logger.info(f"  Softmax Probs: {softmax_probs}")
    logger.info(f"  Expected V output: {expected_V_output_reshaped}")
    logger.info(f"  Actual V output: {output_arr}")

    assert output_arr.shape == expected_V_output_reshaped.shape
    assert output_arr.dtype == mx.float16
    assert mx.allclose(
        output_arr, expected_V_output_reshaped, atol=1e-2, rtol=1e-2
    )  # Increased tolerance slightly for float16 sum


def test_max_score_over_multi_block_history():
    """
    Tests the kernel's ability to find the maximum score over a history window
    that spans multiple logical blocks.

    This test verifies that the kernel can:
    1. Identify the current Q token's logical position
    2. Loop through historical token positions from 0 up to current position
    3. Map each history position to its correct logical block and token slot
    4. Use the page table to find the physical page for each logical block
    5. Compute dot product scores with each historical K-vector
    6. Find and return the maximum score across blocks

    Unlike test_max_score_over_history_in_one_block, this test places history tokens
    in multiple logical blocks that map to different physical pages.
    """
    # --- Config ---
    num_q_threads = 1  # Just one query thread for this test
    cfg_tokens_per_page = 3  # Small value to ensure we span multiple blocks
    cfg_num_kv_heads = 1
    cfg_head_dim = 4

    # Current token position is 5, which spans two logical blocks:
    # - History positions 0,1,2 are in logical block 0 (positions 0,1,2 within the block)
    # - History positions 3,4 are in logical block 1 (positions 0,1 within the block)
    current_position = 5

    # --- Inputs ---
    # 1. Q-vector: Shape [num_q_threads, cfg_head_dim]
    py_queries = mx.array([[1.0, 2.0, 3.0, 4.0]], dtype=mx.float16)

    # 2. K-Cache Pool: [NumPhysPages, TokensPerPage, NumKVHeads, HeadDim]
    num_physical_pages = 2  # We need two physical pages for the two logical blocks
    k_cache_shape = (num_physical_pages, cfg_tokens_per_page, cfg_num_kv_heads, cfg_head_dim)
    py_k_cache_pool = mx.zeros(k_cache_shape, dtype=mx.float16)

    # K-vectors in physical page 0 (logical block 0)
    # K-vector at position 0
    py_k_cache_pool[0, 0, 0, :] = mx.array([1.0, 1.0, 1.0, 1.0], dtype=mx.float16)  # Score will be 5.0
    # K-vector at position 1
    py_k_cache_pool[0, 1, 0, :] = mx.array([2.0, 2.0, 2.0, 2.0], dtype=mx.float16)  # Score will be 10.0
    # K-vector at position 2
    py_k_cache_pool[0, 2, 0, :] = mx.array([0.5, 0.5, 0.5, 0.5], dtype=mx.float16)  # Score will be 2.5

    # K-vectors in physical page 1 (logical block 1)
    # K-vector at position 0 (logical position 3)
    py_k_cache_pool[1, 0, 0, :] = mx.array([3.0, 3.0, 3.0, 3.0], dtype=mx.float16)  # Score will be 15.0
    # K-vector at position 1 (logical position 4)
    py_k_cache_pool[1, 1, 0, :] = mx.array([1.5, 1.5, 1.5, 1.5], dtype=mx.float16)  # Score will be 7.5

    # 3. V-Cache Pool with distinct values for each position
    py_v_cache_pool = mx.zeros_like(py_k_cache_pool)
    # V-vectors in physical page 0 (logical block 0)
    py_v_cache_pool[0, 0, 0, :] = mx.array([10.0, 20.0, 30.0, 40.0], dtype=mx.float16)  # Position 0
    py_v_cache_pool[0, 1, 0, :] = mx.array([50.0, 60.0, 70.0, 80.0], dtype=mx.float16)  # Position 1
    py_v_cache_pool[0, 2, 0, :] = mx.array([5.0, 6.0, 7.0, 8.0], dtype=mx.float16)  # Position 2
    # V-vectors in physical page 1 (logical block 1)
    py_v_cache_pool[1, 0, 0, :] = mx.array([100.0, 110.0, 120.0, 130.0], dtype=mx.float16)  # Position 3
    py_v_cache_pool[1, 1, 0, :] = mx.array([15.0, 25.0, 35.0, 45.0], dtype=mx.float16)  # Position 4

    # 4. Page Table: Maps logical blocks to physical pages
    # - Logical block 0 -> Physical page 0
    # - Logical block 1 -> Physical page 1
    py_page_table = mx.array([[0, 1]], dtype=mx.uint32)  # Shape (1, 2)

    # 5. Sequence Lengths: One sequence with enough tokens
    py_sequence_lengths = mx.array([10], dtype=mx.int32)  # More than current_position

    # 6. Query to Sequence Map: Maps our single query to sequence 0
    py_query_to_seq_map = mx.zeros(num_q_threads, dtype=mx.int32)

    # 7. Query Token Offset: Current position
    py_query_token_offset = mx.array([current_position], dtype=mx.int32)

    # --- Call the kernel ---
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

    # Calculate scores for each history position
    scores_val = []
    # Store V-vectors for each history position
    v_vectors = []

    # Loop through history positions and calculate scaled QK scores
    for hist_pos in range(current_position):
        # Calculate logical block and token slot
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
        score = (mx.sum(q_vec * k_vec) * py_scale).item()
        scores_val.append(score)

    # Calculate detailed score values for debugging and verification
    score0 = (1.0 * 1.0 + 2.0 * 1.0 + 3.0 * 1.0 + 4.0 * 1.0) * py_scale  # = 10 * 0.5 = 5.0
    score1 = (1.0 * 2.0 + 2.0 * 2.0 + 3.0 * 2.0 + 4.0 * 2.0) * py_scale  # = 20 * 0.5 = 10.0
    score2 = (1.0 * 0.5 + 2.0 * 0.5 + 3.0 * 0.5 + 4.0 * 0.5) * py_scale  # = 5 * 0.5 = 2.5
    score3 = (1.0 * 3.0 + 2.0 * 3.0 + 3.0 * 3.0 + 4.0 * 3.0) * py_scale  # = 30 * 0.5 = 15.0
    score4 = (1.0 * 1.5 + 2.0 * 1.5 + 3.0 * 1.5 + 4.0 * 1.5) * py_scale  # = 15 * 0.5 = 7.5

    # Find maximum score
    true_max_score = -float("inf")
    if scores_val:  # Ensure scores_val is not empty
        for s_val in scores_val:
            if s_val > true_max_score:
                true_max_score = s_val
    else:  # Handle case with no valid scores (e.g., empty history)
        true_max_score = 0.0

    # Calculate exp(score - max_score) with clamp
    exp_scores_minus_max = []
    for s_val in scores_val:
        exp_scores_minus_max.append(mx.exp(mx.maximum(s_val - true_max_score, -16.0)).item())

    true_sum_exp_score = sum(exp_scores_minus_max)
    # Handle sum_exp being zero to avoid division by zero
    if not scores_val:  # If original scores_val was empty
        true_sum_exp_score = 0.0  # Or based on spec for empty softmax
    elif true_sum_exp_score < 1e-9:  # If sum is effectively zero due to all scores being very low
        true_sum_exp_score = 1.0  # Prevent division by zero, probs will be ~0

    softmax_probs = []
    if true_sum_exp_score != 0.0:
        for val in exp_scores_minus_max:
            softmax_probs.append(val / true_sum_exp_score)
    elif scores_val:  # scores existed but sum_exp was zero (all scores were ~ -inf)
        softmax_probs = [0.0] * len(scores_val)
    # else: softmax_probs remains empty if scores_val was empty

    # Calculate expected weighted sum of V-vectors
    expected_V_output_py = mx.zeros((cfg_head_dim,), dtype=mx.float32)
    if softmax_probs:  # Only aggregate if there are probabilities
        for i in range(len(softmax_probs)):  # Iterate up to the number of actual scores/probs
            expected_V_output_py += v_vectors[i] * softmax_probs[i]

    expected_V_output_reshaped = expected_V_output_py.astype(mx.float16).reshape(num_q_threads, cfg_head_dim)

    logger.info(f"Score for hist_pos=0 (block 0, slot 0): {score0}")
    logger.info(f"Score for hist_pos=1 (block 0, slot 1): {score1}")
    logger.info(f"Score for hist_pos=2 (block 0, slot 2): {score2}")
    logger.info(f"Score for hist_pos=3 (block 1, slot 0): {score3}")
    logger.info(f"Score for hist_pos=4 (block 1, slot 1): {score4}")
    logger.info(f"Max Score: {true_max_score}")
    logger.info(f"Sum Exp Score: {true_sum_exp_score}")
    logger.info(f"Softmax Probs: {softmax_probs}")
    logger.info(f"Expected V output: {expected_V_output_reshaped}")

    # Output is now full attention format [num_q_threads, cfg_head_dim]
    expected_output_shape = (num_q_threads, cfg_head_dim)

    logger.info(f"Test: Expected output shape: {expected_output_shape}")
    logger.info(f"Test: Actual output shape: {output_arr.shape}")
    logger.info(f"Test: Expected V output: {expected_V_output_reshaped}")
    logger.info(f"Test: Actual V output: {output_arr}")

    assert output_arr.shape == expected_output_shape
    assert output_arr.dtype == mx.float16
    assert mx.allclose(output_arr, expected_V_output_reshaped, atol=1e-2, rtol=1e-2)

    logger.info("test_max_score_over_multi_block_history PASSED")


def test_zero_history_returns_zero_score():
    """
    Tests that the kernel returns a score of 0.0 when there is no history to process.

    This test verifies the code path in the kernel where 'effective_history_length = 0',
    which should result in 'max_score_half = 0.0h' instead of -infinity or garbage values.

    The zero-history case occurs when query_token_offset is 0, meaning the query is
    at the first position with no history tokens before it to compute attention scores with.
    """
    # --- Config ---
    num_q_threads = 2  # Test with multiple threads to ensure consistency
    cfg_tokens_per_page = 64
    cfg_num_kv_heads = 1
    cfg_head_dim = 4

    # --- Inputs ---
    # 1. Q-vector: Simple query vectors, content doesn't matter as no dot products should occur
    py_queries = mx.array(
        [
            [1.0, 2.0, 3.0, 4.0],  # Query for thread 0
            [5.0, 6.0, 7.0, 8.0],  # Query for thread 1
        ],
        dtype=mx.float16,
    )

    # 2. K-Cache Pool: Minimal setup with some values (won't be accessed)
    num_physical_pages = 1
    k_cache_shape = (num_physical_pages, cfg_tokens_per_page, cfg_num_kv_heads, cfg_head_dim)
    py_k_cache_pool = mx.ones(k_cache_shape, dtype=mx.float16) * 10.0  # Fill with 10s

    # 3. V-Cache Pool with values to verify that these are aggregated correctly
    py_v_cache_pool = mx.zeros_like(py_k_cache_pool)
    # Set up V-cache with values to verify correct aggregation
    py_v_cache_pool[0, 0, 0, :] = mx.array([10.0, 20.0, 30.0, 40.0], dtype=mx.float16)  # Position 0
    py_v_cache_pool[0, 1, 0, :] = mx.array([50.0, 60.0, 70.0, 80.0], dtype=mx.float16)  # Position 1 (highest score)
    py_v_cache_pool[0, 2, 0, :] = mx.array([5.0, 6.0, 7.0, 8.0], dtype=mx.float16)  # Position 2
    # Positions 3-4 should NOT be accessed due to sequence length limit
    py_v_cache_pool[0, 3, 0, :] = mx.array([100.0, 200.0, 300.0, 400.0], dtype=mx.float16)  # Position 3
    py_v_cache_pool[1, 0, 0, :] = mx.array([500.0, 600.0, 700.0, 800.0], dtype=mx.float16)  # Position 4

    # 4. Page Table: Simple mapping for one logical block
    py_page_table = mx.array([[0]], dtype=mx.uint32)  # Shape (1, 1)

    # 5. Sequence Lengths: One sequence with tokens
    py_sequence_lengths = mx.array([10], dtype=mx.int32)

    # 6. Query to Sequence Map: Both queries map to sequence 0
    py_query_to_seq_map = mx.zeros(num_q_threads, dtype=mx.int32)

    # 7. Query Token Offset: CRUCIAL - set to 0 to test zero-history case
    py_query_token_offset = mx.zeros(num_q_threads, dtype=mx.int32)

    # --- Call the kernel ---
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

    # --- Expected output ---
    # For zero history, the kernel should return zeros for all V-vectors
    expected_v_output = mx.zeros((num_q_threads, cfg_head_dim), dtype=mx.float16)

    # Output is now full attention format [num_q_threads, cfg_head_dim]
    expected_output_shape = (num_q_threads, cfg_head_dim)

    logger.info(f"Test Zero History: Query token offsets = {py_query_token_offset}")
    logger.info(f"Test Zero History: Expected output shape = {expected_output_shape}")
    logger.info(f"Test Zero History: Actual output shape = {output_arr.shape}")
    logger.info(f"Test Zero History: Expected V output = {expected_v_output}")
    logger.info(f"Test Zero History: Actual V output = {output_arr}")

    # Verify output shape and type
    assert output_arr.shape == expected_output_shape
    assert output_arr.dtype == mx.float16

    # Verify V output contains all zeros
    assert mx.allclose(output_arr, expected_v_output, atol=1e-3)

    logger.info("test_zero_history_returns_zero_score PASSED")


def test_history_limited_by_sequence_length():
    """
    Tests that when query_token_offset > sequence_length, the kernel correctly limits
    the effective history to only the valid sequence length.

    This test verifies that when a query token's logical offset implies a history that extends
    beyond the actual_sequence_length, the kernel correctly processes only up to
    actual_sequence_length - 1 and ignores positions that would be out of bounds.

    The kernel achieves this via the line:
    uint effective_history_length = min(current_q_token_logical_pos, actual_sequence_length);
    """
    # --- Config ---
    cfg_tokens_per_page = 4  # Small value for easier tracking
    cfg_num_kv_heads = 1
    cfg_head_dim = 4
    actual_sequence_length = 3  # Sequence contains only 3 tokens
    query_token_offset = 5  # Query position implies 5 history tokens (0-4)

    # --- Inputs ---
    # 1. Q-vector: Single query
    py_queries = mx.array([[1.0, 1.0, 1.0, 1.0]], dtype=mx.float16)

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
    # Positions 3-4 should NOT be accessed due to sequence length limit
    py_v_cache_pool[0, 3, 0, :] = mx.array([100.0, 200.0, 300.0, 400.0], dtype=mx.float16)  # Position 3
    py_v_cache_pool[1, 0, 0, :] = mx.array([500.0, 600.0, 700.0, 800.0], dtype=mx.float16)  # Position 4

    # 4. Page Table: Two logical blocks mapped to two physical pages
    py_page_table = mx.array([[0, 1]], dtype=mx.uint32)  # Shape (1, 2)

    # 5. Sequence Lengths: One sequence with only 3 tokens
    py_sequence_lengths = mx.array([actual_sequence_length], dtype=mx.int32)

    # 6. Query to Sequence Map: Single query maps to sequence 0
    py_query_to_seq_map = mx.zeros(1, dtype=mx.int32)

    # 7. Query Token Offset: Set to 5, so history would be positions 0-4
    py_query_token_offset = mx.array([query_token_offset], dtype=mx.int32)

    # --- Call the kernel ---
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
    expected_V_output = expected_V_output_py.astype(mx.float16).reshape(num_q_threads, cfg_head_dim)

    # Output is now full attention format [num_q_threads, cfg_head_dim]
    expected_output_shape = (num_q_threads, cfg_head_dim)

    logger.info(f"Test History Limit: actual_sequence_length = {actual_sequence_length}")
    logger.info(f"Test History Limit: query_token_offset = {query_token_offset}")
    logger.info(f"Test History Limit: Score for position 0 = {score0}")
    logger.info(f"Test History Limit: Score for position 1 = {score1} (max)")
    logger.info(f"Test History Limit: Score for position 2 = {score2}")
    logger.info(f"Test History Limit: Softmax probabilities = [{prob0}, {prob1}, {prob2}]")
    logger.info(f"Test History Limit: Expected V output = {expected_V_output}")
    logger.info(f"Test History Limit: Actual V output = {output_arr}")

    # Verify output shape and type
    assert output_arr.shape == expected_output_shape
    assert output_arr.dtype == mx.float16

    # Verify the V output contains the weighted sum of only positions 0-2,
    # not including positions 3-4 which should be excluded by sequence length limit
    assert mx.allclose(output_arr, expected_V_output, atol=1e-2, rtol=1e-2)

    logger.info("test_history_limited_by_sequence_length PASSED")


def test_history_scan_stops_at_page_table_limit():
    """
    Tests that the history scan stops at the page table limit but still returns the
    correct max score from valid blocks processed before the limit.

    This test verifies that if the history scan encounters a logical_block_idx that is
    >= params->max_logical_blocks_per_seq (i.e., beyond what the page table describes
    for that sequence), the kernel correctly stops scanning further history but still
    returns the max score found from valid preceding blocks.

    The kernel achieves this via the code:
    if (logical_block_idx >= params->max_logical_blocks_per_seq) {
        break; // No more valid blocks for this sequence's history
    }
    """
    # --- Config ---
    cfg_tokens_per_page = 2  # Small value to ensure we span multiple blocks quickly
    cfg_num_kv_heads = 1
    cfg_head_dim = 4
    max_logical_blocks_per_seq_in_pagetable = 2  # Page table only describes 2 logical blocks
    query_token_offset = 5  # History positions: 0,1,2,3,4

    # --- Inputs ---
    # 1. Q-vector: Single query
    py_queries = mx.array([[1.0, 1.0, 1.0, 1.0]], dtype=mx.float16)

    # 2. K-Cache Pool:
    num_physical_pages = 2  # Two physical pages for the two logical blocks in page table
    k_cache_shape = (num_physical_pages, cfg_tokens_per_page, cfg_num_kv_heads, cfg_head_dim)
    py_k_cache_pool = mx.zeros(k_cache_shape, dtype=mx.float16)

    # Position mapping with cfg_tokens_per_page = 2:
    # hist_pos 0, 1 -> logical_block_idx 0 -> physical_page 0
    # hist_pos 2, 3 -> logical_block_idx 1 -> physical_page 1
    # hist_pos 4    -> logical_block_idx 2 (beyond page_table) -> should not be accessed

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
    py_query_to_seq_map = mx.zeros(1, dtype=mx.int32)

    # 7. Query Token Offset: Set to 5, so history would be positions 0-4
    py_query_token_offset = mx.array([query_token_offset], dtype=mx.int32)

    # --- Call the kernel ---
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
    expected_V_output = expected_V_output_py.astype(mx.float16).reshape(num_q_threads, cfg_head_dim)
    expected_output_shape = (num_q_threads, cfg_head_dim)

    logger.info(f"Test Page Table Limit: max_logical_blocks_per_seq = {max_logical_blocks_per_seq_in_pagetable}")
    logger.info(f"Test Page Table Limit: query_token_offset = {query_token_offset}")
    logger.info("Test Page Table Limit: Scores for positions in valid blocks:")
    logger.info(f"  Position 0 (block 0, slot 0): {score0}")
    logger.info(f"  Position 1 (block 0, slot 1): {score1}")
    logger.info(f"  Position 2 (block 1, slot 0): {score2} (max)")
    logger.info(f"  Position 3 (block 1, slot 1): {score3}")
    logger.info(f"Test Page Table Limit: Softmax probabilities = [{prob0}, {prob1}, {prob2}, {prob3}]")
    logger.info(f"Test Page Table Limit: Expected V output = {expected_V_output}")
    logger.info(f"Test Page Table Limit: Actual V output = {output_arr}")

    # Verify output shape and type
    assert output_arr.shape == expected_output_shape
    assert output_arr.dtype == mx.float16

    # Verify the V output contains only information from valid page table blocks
    assert mx.allclose(output_arr, expected_V_output, atol=1e-2, rtol=1e-2)

    logger.info("test_history_scan_stops_at_page_table_limit PASSED")
