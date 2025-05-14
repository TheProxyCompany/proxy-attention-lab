import logging

import mlx.core as mx

from proxy_attention_lab import paged_attention

logger = logging.getLogger(__name__)


def test_fetch_k_vector_element_for_first_token_of_sequence():
    """Sequence parallel fetch; expect dot product of Q with K with scale applied."""
    num_q_threads = 2
    cfg_tokens_per_page = 64
    cfg_num_kv_heads = 1
    cfg_head_dim = 4
    cfg_max_logical_blocks_per_seq_in_pagetable = 2

    # Create 2D queries with shape [num_q_threads, cfg_head_dim]
    py_queries = mx.zeros((num_q_threads, cfg_head_dim), dtype=mx.float16)
    # Set Q-vectors: For thread 0, use [100.0, 0.0, 0.0, 0.0]
    py_queries[0, 0] = 100.0
    # For thread 1, use [200.0, 0.0, 0.0, 0.0]
    py_queries[1, 0] = 200.0

    num_physical_pages = 2
    k_cache_shape = (num_physical_pages, cfg_tokens_per_page, cfg_num_kv_heads, cfg_head_dim)
    py_k_cache_pool = mx.zeros(k_cache_shape, dtype=mx.float16)
    py_k_cache_pool[0, 0, 0, 0] = 11.0
    for i in range(1, cfg_head_dim):
        py_k_cache_pool[0, 0, 0, i] = 0.0
    py_k_cache_pool[1, 0, 0, 0] = 22.0
    for i in range(1, cfg_head_dim):
        py_k_cache_pool[1, 0, 0, i] = 0.0

    # Set up V-cache with distinct values
    py_v_cache_pool = mx.zeros_like(py_k_cache_pool)
    py_v_cache_pool[0, 0, 0, :] = mx.array([10.0, 20.0, 30.0, 40.0], dtype=mx.float16)
    py_v_cache_pool[1, 0, 0, :] = mx.array([100.0, 200.0, 300.0, 400.0], dtype=mx.float16)

    py_page_table = mx.array(
        [
            [0, 99],
            [1, 88],
        ],
        dtype=mx.uint32,
    )
    assert py_page_table.shape == (num_q_threads, cfg_max_logical_blocks_per_seq_in_pagetable)
    py_sequence_lengths = mx.array([10, 5], dtype=mx.int32)
    py_query_to_seq_map = mx.array([0, 1], dtype=mx.int32)

    # Modified for history-based attention:
    # With the new kernel looking at history from 0 to query_token_offset - 1,
    # we need to set query_token_offset to 1 to make the kernel look at position 0
    py_query_token_offset = mx.array([1, 1], dtype=mx.int32)

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
    1.0 / mx.sqrt(mx.array(float(cfg_head_dim))).item()

    # Calculate scores for each item
    # Item 0: Q[0] dot K[0,0,0] * scale = 100.0 * 11.0 * 0.5 = 550.0
    # Item 1: Q[1] dot K[1,0,0] * scale = 200.0 * 22.0 * 0.5 = 2200.0

    # Calculate softmax probs - for single history token, prob is always 1.0

    # Calculate expected V outputs
    expected_V_item0 = mx.zeros((cfg_head_dim,), dtype=mx.float32)
    expected_V_item1 = mx.zeros((cfg_head_dim,), dtype=mx.float32)

    # V-aggregation for item 0: V[0,0,0] * prob[0] = V[0,0,0] * 1.0
    expected_V_item0 = py_v_cache_pool[0, 0, 0, :].astype(mx.float32)

    # V-aggregation for item 1: V[1,0,0] * prob[0] = V[1,0,0] * 1.0
    expected_V_item1 = py_v_cache_pool[1, 0, 0, :].astype(mx.float32)

    # Combine and reshape to match kernel output
    expected_V_output = mx.array([expected_V_item0.astype(mx.float16), expected_V_item1.astype(mx.float16)])

    # Debug the actual output shape
    logger.info(f"DEBUG: output_arr shape = {output_arr.shape}, type = {type(output_arr)}")
    logger.info(f"DEBUG: expected_V_output shape = {expected_V_output.shape}")

    # For 2D queries with new full attention output, the shape is [num_q_threads, head_dim]
    expected_output_shape = (num_q_threads, cfg_head_dim)

    logger.info(f"Test: Expected output shape: {expected_output_shape}")
    logger.info(f"Test: Actual output shape: {output_arr.shape}")
    logger.info(f"Test: Expected V output: {expected_V_output}")
    logger.info(f"Test: Actual V output: {output_arr}")

    assert output_arr.shape == expected_output_shape
    assert output_arr.dtype == mx.float16

    # Check that output V-vectors match the expected values
    assert mx.allclose(output_arr, expected_V_output, atol=1e-2, rtol=1e-2)


def test_fetch_entire_k_vector_for_specific_token_slot():
    """Computes dot product between Q and K vectors."""
    num_q_threads = 2
    cfg_tokens_per_page = 64
    cfg_num_kv_heads = 1
    cfg_head_dim = 4
    cfg_max_logical_blocks_per_seq_in_pagetable = 2

    # Create 2D queries with shape [num_q_threads, cfg_head_dim]
    py_queries = mx.zeros((num_q_threads, cfg_head_dim), dtype=mx.float16)
    # Fill each query vector with the same value to simulate the old behavior
    py_queries[0, :] = 100.0
    py_queries[1, :] = 200.0

    num_physical_pages = 2
    k_cache_shape = (num_physical_pages, cfg_tokens_per_page, cfg_num_kv_heads, cfg_head_dim)
    py_k_cache_pool = mx.zeros(k_cache_shape, dtype=mx.float16)
    for i in range(cfg_head_dim):
        py_k_cache_pool[0, 0, 0, i] = float(i + 1)
        py_k_cache_pool[1, 0, 0, i] = float(i + 5)

    # Set up V-cache with distinct values
    py_v_cache_pool = mx.zeros_like(py_k_cache_pool)
    # For item 0 history position 0
    py_v_cache_pool[0, 0, 0, :] = mx.array([10.0, 20.0, 30.0, 40.0], dtype=mx.float16)
    # For item 1 history position 0
    py_v_cache_pool[1, 0, 0, :] = mx.array([50.0, 60.0, 70.0, 80.0], dtype=mx.float16)

    py_page_table = mx.array(
        [
            [0, 99],
            [1, 88],
        ],
        dtype=mx.uint32,
    )
    assert py_page_table.shape == (num_q_threads, cfg_max_logical_blocks_per_seq_in_pagetable)
    py_sequence_lengths = mx.array([10, 5], dtype=mx.int32)
    py_query_to_seq_map = mx.array([0, 1], dtype=mx.int32)

    # Modified for history-based attention:
    # Set token offset to 1 so history includes position 0
    py_query_token_offset = mx.array([1, 1], dtype=mx.int32)

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
    1.0 / mx.sqrt(mx.array(float(cfg_head_dim))).item()

    # Calculate scores for each item
    # Item 0: Q[0] dot K[0,0,0] * scale = 100.0 * (1+2+3+4) * 0.5 = 100.0 * 10 * 0.5 = 500.0
    # Item 1: Q[1] dot K[1,0,0] * scale = 200.0 * (5+6+7+8) * 0.5 = 200.0 * 26 * 0.5 = 2600.0

    # Calculate softmax probs - for single history token, prob is always 1.0

    # Calculate expected V outputs
    # V-aggregation for item 0: V[0,0,0] * prob[0] = V[0,0,0] * 1.0
    expected_V_item0 = py_v_cache_pool[0, 0, 0, :].astype(mx.float32)

    # V-aggregation for item 1: V[1,0,0] * prob[0] = V[1,0,0] * 1.0
    expected_V_item1 = py_v_cache_pool[1, 0, 0, :].astype(mx.float32)

    # Combine and reshape to match kernel output
    expected_V_output = mx.array([expected_V_item0.astype(mx.float16), expected_V_item1.astype(mx.float16)])

    # For 2D queries with new full attention output, the shape is [num_q_threads, head_dim]
    expected_output_shape = (num_q_threads, cfg_head_dim)

    logger.info(f"Test: Expected output shape: {expected_output_shape}")
    logger.info(f"Test: Actual output shape: {output_arr.shape}")
    logger.info(f"Test: Expected V output: {expected_V_output}")
    logger.info(f"Test: Actual V output: {output_arr}")

    assert output_arr.shape == expected_output_shape
    assert output_arr.dtype == mx.float16

    # Check that output V-vectors match the expected values
    assert mx.allclose(output_arr, expected_V_output, atol=1e-2, rtol=1e-2)


def test_fetch_k_vector_from_variable_token_slot_in_first_logical_block():
    """Variable token slot access; computes dot product between Q and K vectors."""
    num_q_threads = 2
    cfg_tokens_per_page = 64
    cfg_num_kv_heads = 1
    cfg_head_dim = 4
    cfg_max_logical_blocks_per_seq_in_pagetable = 2

    # Create 2D queries with shape [num_q_threads, cfg_head_dim]
    py_queries = mx.zeros((num_q_threads, cfg_head_dim), dtype=mx.float16)
    # Fill each query vector with the same value to simulate the old behavior
    py_queries[0, :] = 100.0
    py_queries[1, :] = 200.0

    num_physical_pages = 2
    k_cache_shape = (num_physical_pages, cfg_tokens_per_page, cfg_num_kv_heads, cfg_head_dim)
    py_k_cache_pool = mx.zeros(k_cache_shape, dtype=mx.float16)

    # Adjust the test to work with the new kernel's history mechanism:
    # For thread 0 with token_offset = 4, place the K-vector at position 3 (history)
    # For thread 1 with token_offset = 8, place the K-vector at position 7 (history)
    for i in range(cfg_head_dim):
        py_k_cache_pool[0, 3, 0, i] = float(i + 1)
        py_k_cache_pool[0, 7, 0, i] = float(i + 5)

    # Set up V-cache with distinct values
    py_v_cache_pool = mx.zeros_like(py_k_cache_pool)
    # For item 0 (thread 0) history position 3 (the only history token)
    py_v_cache_pool[0, 3, 0, :] = mx.array([10.0, 20.0, 30.0, 40.0], dtype=mx.float16)
    # For item 1 (thread 1) history position 7 (the only history token)
    py_v_cache_pool[0, 7, 0, :] = mx.array([50.0, 60.0, 70.0, 80.0], dtype=mx.float16)

    py_page_table = mx.array(
        [
            [0, 99],
            [1, 88],
        ],
        dtype=mx.uint32,
    )
    assert py_page_table.shape == (2, cfg_max_logical_blocks_per_seq_in_pagetable)
    py_sequence_lengths = mx.array([64, 32], dtype=mx.int32)
    py_query_to_seq_map = mx.array([0, 0], dtype=mx.int32)

    # Add 1 to each offset so that the kernel looks at the token
    # we're interested in as part of the history window
    py_query_token_offset = mx.array([4, 8], dtype=mx.int32)

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
    1.0 / mx.sqrt(mx.array(float(cfg_head_dim))).item()

    # Calculate scores for each item
    # Item 0 (thread 0): Q[0] dot K[0,3,0] * scale = 100.0 * (1+2+3+4) * 0.5 = 100.0 * 10 * 0.5 = 500.0
    # Item 1 (thread 1): Q[1] dot K[0,7,0] * scale = 200.0 * (5+6+7+8) * 0.5 = 200.0 * 26 * 0.5 = 2600.0

    # Calculate softmax probs - for single history token, prob is always 1.0

    # Calculate expected V outputs
    # V-aggregation for item 0: V[0,3,0] * prob[0] = V[0,3,0] * 1.0
    expected_V_item0 = py_v_cache_pool[0, 3, 0, :].astype(mx.float32)

    # V-aggregation for item 1: V[0,7,0] * prob[0] = V[0,7,0] * 1.0
    expected_V_item1 = py_v_cache_pool[0, 7, 0, :].astype(mx.float32)

    # Combine and reshape to match kernel output
    expected_V_output = mx.array([expected_V_item0.astype(mx.float16), expected_V_item1.astype(mx.float16)])

    # For 2D queries with new full attention output, the shape is [num_q_threads, head_dim]
    expected_output_shape = (num_q_threads, cfg_head_dim)

    logger.info(f"Test: Expected output shape: {expected_output_shape}")
    logger.info(f"Test: Actual output shape: {output_arr.shape}")
    logger.info(f"Test: Expected V output: {expected_V_output}")
    logger.info(f"Test: Actual V output: {output_arr}")

    assert output_arr.shape == expected_output_shape
    assert output_arr.dtype == mx.float16

    # Check that output V-vectors match the expected values
    assert mx.allclose(output_arr, expected_V_output, atol=1e-2, rtol=1e-2)


def test_correct_token_processing_for_2d_queries_variable_offsets():
    """Regression test for 2D queries with variable token offsets.

    Tests that different query tokens correctly map to different K-vectors in the cache
    when using 2D query input format [num_q_threads, cfg_head_dim].

    This specifically verifies the fix for the bug where num_q_heads was incorrectly
    derived for 2D queries, causing incorrect token_idx calculation in the kernel.
    """
    num_q_threads = 2
    cfg_tokens_per_page = 64
    cfg_num_kv_heads = 1
    cfg_head_dim = 4

    # Queries: 2D [num_q_threads, cfg_head_dim]
    py_queries = mx.zeros((num_q_threads, cfg_head_dim), dtype=mx.float16)
    py_queries[0, :] = mx.array([1.0, 2.0, 1.0, 2.0], dtype=mx.float16)  # Q for thread 0
    py_queries[1, :] = mx.array([3.0, 4.0, 3.0, 4.0], dtype=mx.float16)  # Q for thread 1

    num_physical_pages = 1
    k_cache_shape = (num_physical_pages, cfg_tokens_per_page, cfg_num_kv_heads, cfg_head_dim)
    py_k_cache_pool = mx.zeros(k_cache_shape, dtype=mx.float16)

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

    py_page_table = mx.array([[0]], dtype=mx.uint32)  # Seq 0, LogBlock 0 -> PhysPage 0. Shape (1,1)
    # C++ primitive will set params.max_logical_blocks_per_seq = 1
    # And params.num_sequences_in_batch = 1

    # All threads map to sequence 0, but target different token offsets
    py_sequence_lengths = mx.array([cfg_tokens_per_page], dtype=mx.int32)
    py_query_to_seq_map = mx.zeros(num_q_threads, dtype=mx.int32)  # All map to seq 0

    # Modified for history-based attention:
    # Thread 0 will look at position 0 by setting offset to 1
    # Thread 1 will look at position 1 by setting offset to 2
    py_query_token_offset = mx.array([1, 2], dtype=mx.int32)

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
    1.0 / mx.sqrt(mx.array(float(cfg_head_dim))).item()

    # Calculate scores for each item
    # Item 0: Dot(Q[0], K[0,0,0]) * scale = (1+2+1+2) * 0.5 = 6 * 0.5 = 3.0
    # Item 1: Dot(Q[1], K[0,1,0]) * scale = (3+4+3+4)*(2) * 0.5 = 28 * 0.5 = 14.0

    # Calculate softmax probs - for single history token, prob is always 1.0

    # Calculate expected V outputs
    # V-aggregation for item 0: V[0,0,0] * prob[0] = V[0,0,0] * 1.0
    expected_V_item0 = py_v_cache_pool[0, 0, 0, :].astype(mx.float32)

    # V-aggregation for item 1: V[0,1,0] * prob[0] = V[0,1,0] * 1.0
    expected_V_item1 = py_v_cache_pool[0, 1, 0, :].astype(mx.float32)

    # Combine and reshape to match kernel output
    expected_V_output = mx.array([expected_V_item0.astype(mx.float16), expected_V_item1.astype(mx.float16)])

    # For 2D queries with new full attention output, the shape is [num_q_threads, head_dim]
    expected_output_shape = (num_q_threads, cfg_head_dim)

    logger.info(f"DEBUG 2D regression test: output_arr shape = {output_arr.shape}, values = {output_arr}")
    logger.info(f"DEBUG 2D regression test: expected shape = {expected_V_output.shape}, values = {expected_V_output}")

    assert output_arr.shape == expected_output_shape
    assert output_arr.dtype == mx.float16

    # Check that output V-vectors match the expected values
    assert mx.allclose(output_arr, expected_V_output, atol=1e-2, rtol=1e-2)

    logger.info("test_correct_token_processing_for_2d_queries_variable_offsets PASSED")


def test_parallel_online_max_and_sum_exp():
    """Test the parallel online max and sum-exp computation."""
    num_q_threads = 1  # Just one query thread for this test
    cfg_tokens_per_page = 64
    cfg_num_kv_heads = 1
    cfg_head_dim = 4

    # Current token position is 5, so we'll attend to history positions 0, 1, 2, 3, 4
    current_position = 5

    # --- Inputs ---
    # 1. Q-vector: Shape [num_q_threads, cfg_head_dim]
    py_queries = mx.array([[1.0, 1.0, 1.0, 1.0]], dtype=mx.float16)

    # 2. K-Cache Pool: [NumPhysPages, TokensPerPage, NumKVHeads, HeadDim]
    num_physical_pages = 1
    k_cache_shape = (num_physical_pages, cfg_tokens_per_page, cfg_num_kv_heads, cfg_head_dim)
    py_k_cache_pool = mx.zeros(k_cache_shape, dtype=mx.float16)

    # Historical K-vectors with different values to produce different scores
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

    # --- Calculate expected outputs ---
    # Scale factor for dot product
    denominator = mx.sqrt(mx.array(float(cfg_head_dim))).item()
    assert isinstance(denominator, float)
    scale = 1.0 / denominator

    # The scores for each position (with scaling)
    pos0_score = 4 * 0.2 * scale  # = 0.8 * 0.5 = 0.4
    pos1_score = 4 * 0.5 * scale  # = 2.0 * 0.5 = 1.0
    pos2_score = 4 * 1.0 * scale  # = 4.0 * 0.5 = 2.0
    pos3_score = 4 * 0.5 * scale  # = 2.0 * 0.5 = 1.0
    pos4_score = 4 * 0.2 * scale  # = 0.8 * 0.5 = 0.4

    scores_all = [pos0_score, pos1_score, pos2_score, pos3_score, pos4_score]

    # Expected max score (from position 2)
    expected_max_score = max(scores_all)

    # Calculate exp(score - max_score) for each position
    exp_scores_minus_max = []
    for score in scores_all:
        exp_scores_minus_max.append(mx.exp(mx.maximum(score - expected_max_score, -16.0)).item())

    # Calculate sum of exp(score - max_score)
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

    # For new full attention output, shape is [num_q_threads, head_dim]
    expected_output_shape = (num_q_threads, cfg_head_dim)

    logger.info(f"Test online Log-Sum-Exp: Expected output shape: {expected_output_shape}")
    logger.info(f"Test online Log-Sum-Exp: Actual output shape: {output_arr.shape}")
    logger.info(f"Test online Log-Sum-Exp: Scores: {scores_all}")
    logger.info(f"Test online Log-Sum-Exp: Max Score: {expected_max_score}")
    logger.info(f"Test online Log-Sum-Exp: Sum Exp: {expected_sum_exp}")
    logger.info(f"Test online Log-Sum-Exp: Softmax Probs: {softmax_probs}")
    logger.info(f"Test online Log-Sum-Exp: Expected V output: {expected_V_output_reshaped}")
    logger.info(f"Test online Log-Sum-Exp: Actual V output: {output_arr}")

    assert output_arr.shape == expected_output_shape
    assert output_arr.dtype == mx.float16

    # Check final V-vectors match expected values
    assert mx.allclose(output_arr, expected_V_output_reshaped, atol=1e-2, rtol=1e-2)

    logger.info("test_parallel_online_max_and_sum_exp PASSED")


def test_dot_product_q_with_single_k_vector():
    """
    Tests Q.K^T * scale for a single Q-vector and a single K-vector.
    Kernel fetches full Q-vector and full K-vector (from logical_block_0, token_slot_0, kv_head_0)
    and computes their scaled dot product.
    Output shape will be [NumTestTokens * NumQHeads, HeadDim] with full V-aggregation.
    """
    # --- Config ---
    num_test_tokens = 1  # Test with 1 token position for Q
    num_q_heads = 2  # Test with 2 Q heads for this token
    cfg_head_dim = 4  # Dimension of Q, K, V vectors

    cfg_tokens_per_page = 64
    cfg_num_kv_heads = 2  # K-pool has 2 KV heads
    # For this test, let Q-head 0 map to KV-head 0, Q-head 1 to KV-head 1 (GQA factor = 1)
    cfg_max_logical_blocks_per_seq_in_pagetable = 1  # Only one logical block needed for K

    # --- Inputs ---
    # 1. Queries: 3D [NumTestTokens, NumQHeads, HeadDim]
    py_queries = mx.zeros((num_test_tokens, num_q_heads, cfg_head_dim), dtype=mx.float16)
    # Q-vector for (token 0, q_head 0)
    py_queries[0, 0, :] = mx.array([1.0, 2.0, 3.0, 4.0], dtype=mx.float16)
    # Q-vector for (token 0, q_head 1)
    py_queries[0, 1, :] = mx.array([0.5, 1.0, 1.5, 2.0], dtype=mx.float16)

    # 2. K-Cache Pool: [NumPhysPages, TokensPerPage, NumKVHeads, HeadDim]
    num_physical_pages = 1
    k_cache_shape = (num_physical_pages, cfg_tokens_per_page, cfg_num_kv_heads, cfg_head_dim)
    py_k_cache_pool = mx.zeros(k_cache_shape, dtype=mx.float16)

    # K-vector for (phys_page 0, token_slot 0, kv_head 0) - targeted by Q-head 0
    py_k_cache_pool[0, 0, 0, :] = mx.array([1.0, 1.0, 1.0, 1.0], dtype=mx.float16)  # Dot with Q[0,0] = 1+2+3+4 = 10
    # K-vector for (phys_page 0, token_slot 0, kv_head 1) - targeted by Q-head 1
    py_k_cache_pool[0, 0, 1, :] = mx.array([2.0, 2.0, 2.0, 2.0], dtype=mx.float16)  # Dot with Q[0,1] = 1+2+3+4 = 10

    # 3. V-Cache Pool with distinct values for each KV head
    py_v_cache_pool = mx.zeros_like(py_k_cache_pool)
    # V-vector for (phys_page 0, token_slot 0, kv_head 0) - targeted by Q-head 0
    py_v_cache_pool[0, 0, 0, :] = mx.array([10.0, 20.0, 30.0, 40.0], dtype=mx.float16)
    # V-vector for (phys_page 0, token_slot 0, kv_head 1) - targeted by Q-head 1
    py_v_cache_pool[0, 0, 1, :] = mx.array([50.0, 60.0, 70.0, 80.0], dtype=mx.float16)

    # 4. Page Table: [NumBatchSequences, MaxLogicalBlocksPerSeq]
    #    One sequence in batch for this test. Logical block 0 maps to physical page 0.
    num_sequences_in_batch_for_test = 1
    py_page_table = mx.array([[0]], dtype=mx.uint32)  # Shape (1,1)
    assert py_page_table.shape == (num_sequences_in_batch_for_test, cfg_max_logical_blocks_per_seq_in_pagetable)

    # 5. sequence_lengths: [NumBatchSequences]
    py_sequence_lengths = mx.array([cfg_tokens_per_page], dtype=mx.int32)  # Seq 0 has enough tokens

    # 6. query_to_seq_map: [NumTestTokens]. All map to sequence 0.
    # With updated validation, this must match the number of tokens
    py_query_to_seq_map = mx.zeros(num_test_tokens, dtype=mx.int32)

    # 7. query_token_offset: [NumTestTokens].
    #    Modified to work with history-based attention by setting to 1 so the kernel
    #    will look at history position 0
    py_query_token_offset = mx.ones(num_test_tokens, dtype=mx.int32)

    # --- Call Op ---
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
    # Scale = 1 / sqrt(cfg_head_dim)
    denominator = mx.sqrt(mx.array(float(cfg_head_dim))).item()
    assert isinstance(denominator, float)
    py_scale = 1.0 / denominator

    # For each Q-head, calculate the score with its corresponding K-vector
    # Item 0 (Q-head 0): Dot(Q[0,0], K[0,0,0]) * scale = (1+2+3+4) * 0.5 = 10 * 0.5 = 5.0
    # Item 1 (Q-head 1): Dot(Q[0,1], K[0,0,1]) * scale = (0.5+1+1.5+2)*2 * 0.5 = 10 * 0.5 = 5.0
    scores_item0 = [5.0]  # Only one history token for item 0
    scores_item1 = [5.0]  # Only one history token for item 1

    # Calculate softmax probs - for single history token, prob is always 1.0

    # Calculate expected V outputs
    # V-aggregation for item 0 (Q-head 0): V[0,0,0] * prob[0] = V[0,0,0] * 1.0
    expected_V_item0 = py_v_cache_pool[0, 0, 0, :].astype(mx.float32)

    # V-aggregation for item 1 (Q-head 1): V[0,0,1] * prob[0] = V[0,0,1] * 1.0
    expected_V_item1 = py_v_cache_pool[0, 0, 1, :].astype(mx.float32)

    # Combine and reshape to match kernel output
    # For 3D query input with shape [NumTestTokens, NumQHeads, HeadDim],
    # total items is NumTestTokens * NumQHeads = 1 * 2 = 2
    total_items = num_test_tokens * num_q_heads
    expected_V_output = mx.array([expected_V_item0.astype(mx.float16), expected_V_item1.astype(mx.float16)])

    # Expected shape for new full attention output: [total_items, head_dim]
    expected_output_shape = (total_items, cfg_head_dim)

    # Calculate raw dot products for debugging
    q0_dot_k0 = 1.0 * 1.0 + 2.0 * 1.0 + 3.0 * 1.0 + 4.0 * 1.0  # = 10
    q1_dot_k1 = 0.5 * 2.0 + 1.0 * 2.0 + 1.5 * 2.0 + 2.0 * 2.0  # = 10

    logger.info(f"DEBUG: output_arr shape = {output_arr.shape}, type = {type(output_arr)}")
    logger.info(f"Test: Q0: {py_queries[0, 0, :]}, K0 chosen: {py_k_cache_pool[0, 0, 0, :]}")
    logger.info(f"Test: Q0·K0 raw dot product = {q0_dot_k0}, scale={py_scale}, Expected Score0 = {scores_item0[0]}")
    logger.info(f"Test: Q1: {py_queries[0, 1, :]}, K1 chosen: {py_k_cache_pool[0, 0, 1, :]}")
    logger.info(f"Test: Q1·K1 raw dot product = {q1_dot_k1}, scale={py_scale}, Expected Score1 = {scores_item1[0]}")

    logger.info(f"Test: Expected output shape: {expected_output_shape}")
    logger.info(f"Test: Actual output shape: {output_arr.shape}")
    logger.info(f"Test: Expected V output: {expected_V_output}")
    logger.info(f"Test: Actual V output: {output_arr}")

    assert output_arr.shape == expected_output_shape, f"Shape: {output_arr.shape} vs {expected_output_shape}"
    assert output_arr.dtype == mx.float16

    # Check V-vectors match expected values
    assert mx.allclose(output_arr, expected_V_output, atol=1e-2, rtol=1e-2)

    logger.info("test_dot_product_q_with_single_k_vector PASSED")
