import logging

import mlx.core as mx
import pytest

from proxy_attention_lab import paged_attention

logger = logging.getLogger(__name__)


def test_fetch_k_vector_from_multiple_kv_heads():
    """GQA: multiple Q heads map to KV heads; computes dot products."""
    num_tokens = 1
    num_q_heads = 2
    cfg_tokens_per_page = 64
    cfg_num_kv_heads = 2
    cfg_head_dim = 4
    cfg_max_logical_blocks_per_seq_in_pagetable = 2
    py_queries = mx.zeros((num_tokens, num_q_heads, cfg_head_dim), dtype=mx.float16)
    py_queries[0, 0, :] = 100.0
    py_queries[0, 1, :] = 200.0
    num_physical_pages = 1
    k_cache_shape = (num_physical_pages, cfg_tokens_per_page, cfg_num_kv_heads, cfg_head_dim)
    py_k_cache_pool = mx.zeros(k_cache_shape, dtype=mx.float16)
    token_slot = 5
    for i in range(cfg_head_dim):
        py_k_cache_pool[0, token_slot, 0, i] = float(i + 1)
        py_k_cache_pool[0, token_slot, 1, i] = float(i + 5)
    py_v_cache_pool = mx.zeros_like(py_k_cache_pool)
    py_page_table = mx.array(
        [
            [0, 99],
        ],
        dtype=mx.uint32,
    )
    assert py_page_table.shape == (1, cfg_max_logical_blocks_per_seq_in_pagetable)
    py_sequence_lengths = mx.array([64], dtype=mx.int32)
    py_query_to_seq_map = mx.array([0, 0], dtype=mx.int32)

    # With the new kernel looking at history, we need token_slot + 1
    # to make it look at just the token we want
    py_query_token_offset = mx.array([token_slot + 1, token_slot + 1], dtype=mx.int32)

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

    # For token 0, q_head 0 -> k_head 0:
    # Q[0,0,:] = [100.0, 100.0, 100.0, 100.0], K = [1.0, 2.0, 3.0, 4.0]
    # Dot product = 100.0 * 1.0 + 100.0 * 2.0 + 100.0 * 3.0 + 100.0 * 4.0 = 1000.0
    # Scaled = 1000.0 * py_scale = 1000.0 / 2.0 = 500.0

    # For token 0, q_head 1 -> k_head 1:
    # Q[0,1,:] = [200.0, 200.0, 200.0, 200.0], K = [5.0, 6.0, 7.0, 8.0]
    # Dot product = 200.0 * 5.0 + 200.0 * 6.0 + 200.0 * 7.0 + 200.0 * 8.0 = 5200.0
    # Scaled = 5200.0 * py_scale = 5200.0 / 2.0 = 2600.0

    # For 3D queries [num_tokens, num_q_heads, cfg_head_dim], output should be [num_tokens, num_q_heads]
    expected_output_value = mx.array([500.0, 2600.0], dtype=mx.float16)
    expected_output_shape = (num_tokens, num_q_heads)
    expected_output = expected_output_value.reshape(expected_output_shape)

    logger.info(f"Test: Expected output: {expected_output}")
    logger.info(f"Test: Actual output  : {output_arr}")
    assert output_arr.shape == expected_output_shape
    assert output_arr.dtype == mx.float16
    assert mx.allclose(output_arr, expected_output, atol=1e-3)


def test_invalid_gqa_configuration():
    """Non multiple GQA config raises exception."""
    num_tokens = 1
    num_q_heads = 3
    cfg_tokens_per_page = 64
    cfg_num_kv_heads = 2
    cfg_head_dim = 4
    py_queries = mx.zeros((num_tokens, num_q_heads, cfg_head_dim), dtype=mx.float16)
    py_queries[0, 0, :] = 100.0
    py_queries[0, 1, :] = 200.0
    py_queries[0, 2, :] = 300.0
    num_physical_pages = 1
    k_cache_shape = (num_physical_pages, cfg_tokens_per_page, cfg_num_kv_heads, cfg_head_dim)
    py_k_cache_pool = mx.zeros(k_cache_shape, dtype=mx.float16)
    py_v_cache_pool = mx.zeros_like(py_k_cache_pool)
    py_page_table = mx.array(
        [
            [0, 99],
        ],
        dtype=mx.uint32,
    )
    py_sequence_lengths = mx.array([64], dtype=mx.int32)
    py_query_to_seq_map = mx.array([0, 0, 0], dtype=mx.int32)
    py_query_token_offset = mx.array([0, 0, 0], dtype=mx.int32)

    with pytest.raises((RuntimeError, ValueError), match="num_q_heads must be an integer multiple of num_kv_heads"):
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


def test_mqa_kv_head_selection():
    """
    Tests Multi-Query Attention (MQA) configuration where num_q_heads < num_kv_heads.

    This test verifies that the kernel correctly maps query heads to KV heads
    when there are fewer query heads than KV heads, ensuring each query head
    attends to the correct KV head according to the mapping logic.

    Specifically, with num_q_heads=1 and num_kv_heads=2, the test confirms
    that queries use KV head 0 as specified in the kernel's MQA logic.
    """
    # MQA configuration: fewer query heads than KV heads
    num_tokens = 1
    num_q_heads = 1  # Only one query head
    cfg_tokens_per_page = 64
    cfg_num_kv_heads = 2  # Two KV heads
    cfg_head_dim = 4

    # Create 3D queries with shape [num_tokens, num_q_heads, cfg_head_dim]
    py_queries = mx.zeros((num_tokens, num_q_heads, cfg_head_dim), dtype=mx.float16)
    # Q-vector for the single query head
    py_queries[0, 0, :] = mx.array([1.0, 2.0, 3.0, 4.0], dtype=mx.float16)

    # Create K-cache pool with different K-vectors in each KV head
    num_physical_pages = 1
    k_cache_shape = (num_physical_pages, cfg_tokens_per_page, cfg_num_kv_heads, cfg_head_dim)
    py_k_cache_pool = mx.zeros(k_cache_shape, dtype=mx.float16)

    # K-vector for KV head 0 - this is the one that should be used
    py_k_cache_pool[0, 0, 0, :] = mx.array([1.0, 1.0, 1.0, 1.0], dtype=mx.float16)
    # K-vector for KV head 1 - should NOT be used by the single query head
    py_k_cache_pool[0, 0, 1, :] = mx.array([2.0, 2.0, 2.0, 2.0], dtype=mx.float16)

    py_v_cache_pool = mx.zeros_like(py_k_cache_pool)
    py_page_table = mx.array([[0]], dtype=mx.uint32)  # Logical block 0 -> Physical page 0
    py_sequence_lengths = mx.array([cfg_tokens_per_page], dtype=mx.int32)

    # Map our single query to sequence 0
    py_query_to_seq_map = mx.zeros(num_q_heads, dtype=mx.int32)

    # Set token offset to 1 to look at history position 0
    py_query_token_offset = mx.ones(num_q_heads, dtype=mx.int32)

    # Call the kernel
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

    # Calculate expected results
    # The kernel should use K-vector from KV head 0 for the query
    denominator = mx.sqrt(mx.array(float(cfg_head_dim))).item()
    assert isinstance(denominator, float)
    py_scale = 1.0 / denominator

    # Q=[1,2,3,4] with K=[1,1,1,1] from kv_head=0 gives dot product = 10
    expected_score = (1.0 * 1.0 + 2.0 * 1.0 + 3.0 * 1.0 + 4.0 * 1.0) * py_scale  # = 10 * 0.5 = 5.0

    # If the kernel incorrectly used kv_head=1 with K=[2,2,2,2], we'd get:
    incorrect_score = (1.0 * 2.0 + 2.0 * 2.0 + 3.0 * 2.0 + 4.0 * 2.0) * py_scale  # = 20 * 0.5 = 10.0

    # For 3D queries, output shape is [num_tokens, num_q_heads]
    expected_output = mx.array([[expected_score]], dtype=mx.float16)

    logger.info(f"Test MQA: Q = {py_queries[0, 0, :]}")
    logger.info(f"Test MQA: K (KV head 0) = {py_k_cache_pool[0, 0, 0, :]}")
    logger.info(f"Test MQA: K (KV head 1) = {py_k_cache_pool[0, 0, 1, :]}")
    logger.info(f"Test MQA: Expected score (using KV head 0) = {expected_score}")
    logger.info(f"Test MQA: Incorrect score (would use KV head 1) = {incorrect_score}")
    logger.info(f"Test MQA: Actual output = {output_arr}")

    assert output_arr.shape == expected_output.shape
    assert output_arr.dtype == mx.float16
    assert mx.allclose(output_arr, expected_output, atol=1e-2, rtol=1e-2)

    # Also explicitly verify we're not getting the incorrect score from KV head 1
    incorrect_output = mx.array([[incorrect_score]], dtype=mx.float16)
    assert not mx.allclose(output_arr, incorrect_output, atol=1e-2, rtol=1e-2)

    logger.info("test_mqa_kv_head_selection PASSED")


def test_mqa_multi_token_kv_head_selection_2d_query():
    """
    Tests MQA (Multi-Query Attention) head mapping for multiple tokens with 2D queries.

    This test ensures that for 2D queries (where num_q_heads is effectively 1 per thread)
    and num_kv_heads > 1, the kernel consistently uses the same KV-head (expected to be
    KV-head 0) for all query tokens, rather than cycling through KV-heads.

    The test will FAIL with incorrect KV-head selection logic for 2D queries in MQA mode,
    where the kernel might use different KV-heads for different tokens.
    """
    # Test configuration
    num_tokens = 5  # Multiple tokens to test consistent KV-head selection
    cfg_head_dim = 4
    cfg_tokens_per_page = 64
    cfg_num_kv_heads = 4  # Multiple KV heads

    # Create 2D queries with shape [num_tokens, cfg_head_dim]
    # For 2D queries, the C++ primitive sets params->num_q_heads = 1 internally
    py_queries = mx.array([[1.0] * cfg_head_dim] * num_tokens, dtype=mx.float16)

    # Create K-cache pool with drastically different values for each KV head
    num_physical_pages = 1
    k_cache_shape = (num_physical_pages, cfg_tokens_per_page, cfg_num_kv_heads, cfg_head_dim)
    py_k_cache_pool = mx.zeros(k_cache_shape, dtype=mx.float16)

    # Set up KV heads with distinctive values to detect which head is used
    # KV-head 0: All 1.0's
    py_k_cache_pool[0, 0, 0, :] = mx.array([1.0, 1.0, 1.0, 1.0], dtype=mx.float16)
    # KV-head 1: All 100.0's
    py_k_cache_pool[0, 0, 1, :] = mx.array([100.0, 100.0, 100.0, 100.0], dtype=mx.float16)
    # KV-head 2: All 10000.0's
    py_k_cache_pool[0, 0, 2, :] = mx.array([10000.0, 10000.0, 10000.0, 10000.0], dtype=mx.float16)
    # KV-head 3: All 1000000.0's
    py_k_cache_pool[0, 0, 3, :] = mx.array([1000000.0, 1000000.0, 1000000.0, 1000000.0], dtype=mx.float16)

    py_v_cache_pool = mx.zeros_like(py_k_cache_pool)
    py_page_table = mx.array([[0]], dtype=mx.uint32)  # One logical block mapped to physical page 0
    py_sequence_lengths = mx.array([cfg_tokens_per_page], dtype=mx.int32)  # Plenty of tokens in the sequence

    # All query tokens map to sequence 0
    py_query_to_seq_map = mx.zeros(num_tokens, dtype=mx.int32)

    # All query tokens look at history position 0
    py_query_token_offset = mx.ones(num_tokens, dtype=mx.int32)

    # Call the kernel
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

    # Calculate scale factor
    denominator = mx.sqrt(mx.array(float(cfg_head_dim))).item()
    assert isinstance(denominator, float)
    py_scale = 1.0 / denominator

    # Expected scores if each token uses the same KV-head (KV-head 0)
    # Q=[1,1,1,1] with K=[1,1,1,1] from kv_head=0 gives dot product = 4
    expected_score = 4.0 * py_scale
    expected_output = mx.array([expected_score] * num_tokens, dtype=mx.float16)

    # Alternative scores if tokens would incorrectly use different KV-heads
    alt_scores = [
        4.0 * py_scale,  # KV-head 0: [1,1,1,1]
        400.0 * py_scale,  # KV-head 1: [100,100,100,100]
        40000.0 * py_scale,  # KV-head 2: [10000,10000,10000,10000]
        4000000.0 * py_scale,  # KV-head 3: [1000000,1000000,1000000,1000000]
    ]

    logger.info(f"Test MQA Multi-Token: Queries shape = {py_queries.shape}")
    logger.info(f"Test MQA Multi-Token: Expected score from KV-head 0 = {expected_score}")
    logger.info(f"Test MQA Multi-Token: Possible scores from different KV-heads = {alt_scores}")
    logger.info(f"Test MQA Multi-Token: Actual output = {output_arr}")

    # Check shape and type
    assert output_arr.shape == (num_tokens,)
    assert output_arr.dtype == mx.float16

    # Verification: All tokens should get the same score by using KV-head 0 consistently
    assert mx.allclose(output_arr, expected_output, atol=1e-2, rtol=1e-2)

    # Extra verification: output should not contain any scores from other KV-heads
    for i, alt_score in enumerate(alt_scores[1:], 1):
        incorrect_output = mx.array([alt_score], dtype=mx.float16)
        # Check that none of the output elements match this incorrect score
        for token_idx in range(num_tokens):
            assert not mx.allclose(output_arr[token_idx : token_idx + 1], incorrect_output, atol=1e-2, rtol=1e-2), (
                f"Token {token_idx} appears to be using KV-head {i} incorrectly"
            )

    logger.info("test_mqa_multi_token_kv_head_selection_2d_query PASSED")
