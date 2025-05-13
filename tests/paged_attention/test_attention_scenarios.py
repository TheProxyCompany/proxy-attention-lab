import logging

import mlx.core as mx

from proxy_attention_lab import paged_attention

logger = logging.getLogger(__name__)


def test_max_score_over_history_in_one_block():
    """
    Tests the kernel's ability to find the maximum score over a history window.

    This test verifies that the kernel can:
    1. Identify the current Q token's logical position
    2. Loop through historical token positions from 0 up to current position
    3. Compute dot product scores with each historical K-vector
    4. Find and return the maximum score

    All token positions in this test are within the same logical block 0.
    """
    # --- Config ---
    num_q_threads = 1  # Just one query thread for this test
    cfg_tokens_per_page = 64
    cfg_num_kv_heads = 1
    cfg_head_dim = 4

    # Current token position is 3, so we'll attend to history positions 0, 1, and 2
    current_position = 3

    # --- Inputs ---
    # 1. Q-vector: Shape [num_q_threads, cfg_head_dim]
    py_queries = mx.array([[1.0, 2.0, 3.0, 4.0]], dtype=mx.float16)

    # 2. K-Cache Pool: [NumPhysPages, TokensPerPage, NumKVHeads, HeadDim]
    num_physical_pages = 1
    k_cache_shape = (num_physical_pages, cfg_tokens_per_page, cfg_num_kv_heads, cfg_head_dim)
    py_k_cache_pool = mx.zeros(k_cache_shape, dtype=mx.float16)

    # Historical K-vectors with different values to produce different scores
    # K-vector at position 0
    py_k_cache_pool[0, 0, 0, :] = mx.array([1.0, 1.0, 1.0, 1.0], dtype=mx.float16)
    # K-vector at position 1
    py_k_cache_pool[0, 1, 0, :] = mx.array([2.0, 2.0, 2.0, 2.0], dtype=mx.float16)
    # K-vector at position 2
    py_k_cache_pool[0, 2, 0, :] = mx.array([0.5, 0.5, 0.5, 0.5], dtype=mx.float16)

    # 3. V-Cache Pool (not used for this test)
    py_v_cache_pool = mx.zeros_like(py_k_cache_pool)

    # 4. Page Table: Maps logical block 0 to physical page 0
    py_page_table = mx.array([[0]], dtype=mx.uint32)  # Shape (1, 1)

    # 5. Sequence Lengths: One sequence with enough tokens
    py_sequence_lengths = mx.array([cfg_tokens_per_page], dtype=mx.int32)

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

    # --- Calculate expected output ---
    # Scale factor for dot product
    denominator = mx.sqrt(mx.array(float(cfg_head_dim))).item()
    assert isinstance(denominator, float)
    py_scale = 1.0 / denominator

    # Initialize list to store scores for each historical position
    scores = []

    # Loop through history positions
    for hist_pos in range(current_position):
        # For this test, token_slot_in_page is the same as hist_pos (all in logical block 0)
        token_slot_in_page = hist_pos

        # Get K-vector for this historical position
        k_vec = py_k_cache_pool[0, token_slot_in_page, 0, :]

        # Compute the dot product
        q_vec = py_queries[0]
        dot_product = mx.sum(q_vec * k_vec).item()
        assert isinstance(dot_product, float)

        # Apply scaling
        score = dot_product * py_scale
        scores.append(score)

    # Find maximum score
    expected_max_score = max(scores)

    # Calculate detailed score values for debugging and verification
    score0 = (1.0 * 1.0 + 2.0 * 1.0 + 3.0 * 1.0 + 4.0 * 1.0) * py_scale  # = 10 * 0.5 = 5.0
    score1 = (1.0 * 2.0 + 2.0 * 2.0 + 3.0 * 2.0 + 4.0 * 2.0) * py_scale  # = 20 * 0.5 = 10.0
    score2 = (1.0 * 0.5 + 2.0 * 0.5 + 3.0 * 0.5 + 4.0 * 0.5) * py_scale  # = 5 * 0.5 = 2.5

    logger.info(f"Score for hist_pos=0: {score0}")
    logger.info(f"Score for hist_pos=1: {score1}")
    logger.info(f"Score for hist_pos=2: {score2}")
    logger.info(f"Expected max score: {expected_max_score}")

    # The expected max score should be from hist_pos=1 (score1 = 10.0)
    expected_output = mx.array([expected_max_score], dtype=mx.float16)

    logger.info(f"Test: Expected output: {expected_output}")
    logger.info(f"Test: Actual output: {output_arr}")

    assert output_arr.shape == (num_q_threads,)
    assert output_arr.dtype == mx.float16
    assert mx.allclose(output_arr, expected_output, atol=1e-2, rtol=1e-2)

    logger.info("test_max_score_over_history_in_one_block PASSED")


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

    # 3. V-Cache Pool (not used for this test)
    py_v_cache_pool = mx.zeros_like(py_k_cache_pool)

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

    # --- Calculate expected output ---
    # Scale factor for dot product
    denominator = mx.sqrt(mx.array(float(cfg_head_dim))).item()
    assert isinstance(denominator, float)
    py_scale = 1.0 / denominator

    # Initialize list to store scores for each historical position
    scores = []

    # Loop through history positions
    for hist_pos in range(current_position):
        # Calculate logical block and token slot
        logical_block_idx = hist_pos // cfg_tokens_per_page
        token_slot_in_page = hist_pos % cfg_tokens_per_page

        # Get physical page from page table
        physical_page_id = py_page_table[0, logical_block_idx].item()

        # Get K-vector for this historical position
        k_vec = py_k_cache_pool[physical_page_id, token_slot_in_page, 0, :]

        # Compute the dot product
        q_vec = py_queries[0]
        dot_product = mx.sum(q_vec * k_vec).item()
        assert isinstance(dot_product, float)

        # Apply scaling
        score = dot_product * py_scale
        scores.append(score)

    # Find maximum score
    expected_max_score = max(scores)

    # Calculate detailed score values for debugging and verification
    score0 = (1.0 * 1.0 + 2.0 * 1.0 + 3.0 * 1.0 + 4.0 * 1.0) * py_scale  # = 10 * 0.5 = 5.0
    score1 = (1.0 * 2.0 + 2.0 * 2.0 + 3.0 * 2.0 + 4.0 * 2.0) * py_scale  # = 20 * 0.5 = 10.0
    score2 = (1.0 * 0.5 + 2.0 * 0.5 + 3.0 * 0.5 + 4.0 * 0.5) * py_scale  # = 5 * 0.5 = 2.5
    score3 = (1.0 * 3.0 + 2.0 * 3.0 + 3.0 * 3.0 + 4.0 * 3.0) * py_scale  # = 30 * 0.5 = 15.0
    score4 = (1.0 * 1.5 + 2.0 * 1.5 + 3.0 * 1.5 + 4.0 * 1.5) * py_scale  # = 15 * 0.5 = 7.5

    logger.info(f"Score for hist_pos=0 (block 0, slot 0): {score0}")
    logger.info(f"Score for hist_pos=1 (block 0, slot 1): {score1}")
    logger.info(f"Score for hist_pos=2 (block 0, slot 2): {score2}")
    logger.info(f"Score for hist_pos=3 (block 1, slot 0): {score3}")
    logger.info(f"Score for hist_pos=4 (block 1, slot 1): {score4}")
    logger.info(f"Expected max score: {expected_max_score}")

    # The expected max score should be from hist_pos=3 (score3 = 15.0)
    expected_output = mx.array([expected_max_score], dtype=mx.float16)

    logger.info(f"Test: Expected output: {expected_output}")
    logger.info(f"Test: Actual output: {output_arr}")

    assert output_arr.shape == (num_q_threads,)
    assert output_arr.dtype == mx.float16
    assert mx.allclose(output_arr, expected_output, atol=1e-2, rtol=1e-2)

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

    # 3. V-Cache Pool (not used for this test)
    py_v_cache_pool = mx.zeros_like(py_k_cache_pool)

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
    # For zero history, the kernel should return 0.0 for all threads
    expected_output = mx.zeros(num_q_threads, dtype=mx.float16)

    logger.info(f"Test Zero History: Query token offsets = {py_query_token_offset}")
    logger.info(f"Test Zero History: Expected output = {expected_output}")
    logger.info(f"Test Zero History: Actual output = {output_arr}")

    # Verify output shape and type
    assert output_arr.shape == (num_q_threads,)
    assert output_arr.dtype == mx.float16

    # Verify all outputs are 0.0
    assert mx.allclose(output_arr, expected_output, atol=1e-3)

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

    # 3. V-Cache Pool (not used for this test)
    py_v_cache_pool = mx.zeros_like(py_k_cache_pool)

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

    # Max score should be from position 1 (score1 = 6.0)
    expected_max_score = score1
    expected_output = mx.array([expected_max_score], dtype=mx.float16)

    logger.info(f"Test History Limit: actual_sequence_length = {actual_sequence_length}")
    logger.info(f"Test History Limit: query_token_offset = {query_token_offset}")
    logger.info(f"Test History Limit: Score for position 0 = {score0}")
    logger.info(f"Test History Limit: Score for position 1 = {score1} (max)")
    logger.info(f"Test History Limit: Score for position 2 = {score2}")
    logger.info(f"Test History Limit: Expected max score = {expected_max_score}")
    logger.info(f"Test History Limit: Actual output = {output_arr}")

    # Verify output shape and type
    assert output_arr.shape == (1,)
    assert output_arr.dtype == mx.float16

    # Verify the max score is from position 1, not from positions 3 or 4
    # which should not be accessed due to sequence length limit
    assert mx.allclose(output_arr, expected_output, atol=1e-2, rtol=1e-2)

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

    # 3. V-Cache Pool (not used for this test)
    py_v_cache_pool = mx.zeros_like(py_k_cache_pool)

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
    expected_max_score = score2
    expected_output = mx.array([expected_max_score], dtype=mx.float16)

    logger.info(f"Test Page Table Limit: max_logical_blocks_per_seq = {max_logical_blocks_per_seq_in_pagetable}")
    logger.info(f"Test Page Table Limit: query_token_offset = {query_token_offset}")
    logger.info("Test Page Table Limit: Scores for positions in valid blocks:")
    logger.info(f"  Position 0 (block 0, slot 0): {score0}")
    logger.info(f"  Position 1 (block 0, slot 1): {score1}")
    logger.info(f"  Position 2 (block 1, slot 0): {score2} (max)")
    logger.info(f"  Position 3 (block 1, slot 1): {score3}")
    logger.info(f"Test Page Table Limit: Expected max score = {expected_max_score}")
    logger.info(f"Test Page Table Limit: Actual output = {output_arr}")

    # Verify output shape and type
    assert output_arr.shape == (1,)
    assert output_arr.dtype == mx.float16

    # Verify the max score is from position 2, which is within the valid page table range
    assert mx.allclose(output_arr, expected_output, atol=1e-2, rtol=1e-2)

    logger.info("test_history_scan_stops_at_page_table_limit PASSED")
