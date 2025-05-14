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

    # Set up V-cache pool with distinct values for each K-vector position
    # Values for KV head 0 (used by Q head 0)
    py_v_cache_pool[0, token_slot, 0, :] = mx.array([10.0, 11.0, 12.0, 13.0], dtype=mx.float16)
    # Values for KV head 1 (used by Q head 1)
    py_v_cache_pool[0, token_slot, 1, :] = mx.array([20.0, 21.0, 22.0, 23.0], dtype=mx.float16)

    py_page_table = mx.array(
        [
            [0, 99],
        ],
        dtype=mx.uint32,
    )
    assert py_page_table.shape == (1, cfg_max_logical_blocks_per_seq_in_pagetable)
    py_sequence_lengths = mx.array([64], dtype=mx.int32)

    # Need query_to_seq_map with one entry per token, not per head
    py_query_to_seq_map = mx.array([0], dtype=mx.int32)

    # With the new kernel looking at history, we need token_slot + 1
    # to make it look at just the token we want
    py_query_token_offset = mx.array([token_slot + 1], dtype=mx.int32)

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

    # Output is now full attention format [num_q_threads, cfg_head_dim]
    # For 3D queries, shape is [num_tokens * num_q_heads, cfg_head_dim]
    total_items = num_tokens * num_q_heads
    expected_output_shape = (total_items, cfg_head_dim)

    # Get the entire output array
    logger.info(f"DEBUG: FULL OUTPUT ARRAY: {output_arr}")

    # Calculate expected V-output for each query head
    # Since we only have one history token per query, each softmax prob is 1.0
    # Therefore V-output for each query head is exactly the corresponding V-vector

    # For Q head 0 -> KV head 0: V-vector should be [10, 11, 12, 13]
    expected_v_head0 = mx.array([10.0, 11.0, 12.0, 13.0], dtype=mx.float16)

    # For Q head 1 -> KV head 1: V-vector should be [20, 21, 22, 23]
    expected_v_head1 = mx.array([20.0, 21.0, 22.0, 23.0], dtype=mx.float16)

    # Combine into expected V-output array
    expected_v_output = mx.stack([expected_v_head0, expected_v_head1])

    logger.info(f"Test: Expected output shape: {expected_output_shape}")
    logger.info(f"Test: Actual output shape: {output_arr.shape}")
    logger.info(f"Test: Expected V output: {expected_v_output}")
    logger.info(f"Test: Actual V output: {output_arr}")

    assert output_arr.shape == expected_output_shape
    assert output_arr.dtype == mx.float16

    # Check V output vectors
    assert mx.allclose(output_arr, expected_v_output, atol=1e-2, rtol=1e-2)


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
    # Set up V-cache pool with distinct values for each KV head
    # For KV head 0 (which will be used in MQA)
    py_v_cache_pool[0, 0, 0, :] = mx.array([10.0, 20.0, 30.0, 40.0], dtype=mx.float16)
    # For KV head 1 (which would be incorrect to use)
    py_v_cache_pool[0, 0, 1, :] = mx.array([50.0, 60.0, 70.0, 80.0], dtype=mx.float16)

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
    (1.0 * 1.0 + 2.0 * 1.0 + 3.0 * 1.0 + 4.0 * 1.0) * py_scale  # = 10 * 0.5 = 5.0

    # If the kernel incorrectly used kv_head=1 with K=[2,2,2,2], we'd get:
    (1.0 * 2.0 + 2.0 * 2.0 + 3.0 * 2.0 + 4.0 * 2.0) * py_scale  # = 20 * 0.5 = 10.0

    # Output is now full attention format [num_q_threads, cfg_head_dim]
    total_items = num_tokens * num_q_heads
    expected_output_shape = (total_items, cfg_head_dim)

    # Since we only have one history token, the softmax prob is 1.0
    # Therefore, expected V output should be exactly the V vector from KV head 0
    expected_v_output = py_v_cache_pool[0, 0, 0, :].reshape(1, cfg_head_dim)

    # Incorrect V output would be using KV head 1's V vector
    incorrect_v_output = py_v_cache_pool[0, 0, 1, :].reshape(1, cfg_head_dim)

    logger.info(f"Test MQA: Q = {py_queries[0, 0, :]}")
    logger.info(f"Test MQA: K (KV head 0) = {py_k_cache_pool[0, 0, 0, :]}")
    logger.info(f"Test MQA: K (KV head 1) = {py_k_cache_pool[0, 0, 1, :]}")
    logger.info(f"Test MQA: V (KV head 0) = {py_v_cache_pool[0, 0, 0, :]}")
    logger.info(f"Test MQA: V (KV head 1) = {py_v_cache_pool[0, 0, 1, :]}")
    logger.info(f"Test MQA: Expected output shape = {expected_output_shape}")
    logger.info(f"Test MQA: Actual output shape = {output_arr.shape}")
    logger.info(f"Test MQA: Expected V output = {expected_v_output}")
    logger.info(f"Test MQA: Actual V output = {output_arr}")

    assert output_arr.shape == expected_output_shape
    assert output_arr.dtype == mx.float16

    # Verify that the kernel is correctly using KV head 0 for the query by checking the V output
    assert mx.allclose(output_arr, expected_v_output, atol=1e-2, rtol=1e-2)

    # Also explicitly verify we're not getting the incorrect V-vector from KV head 1
    assert not mx.allclose(output_arr, incorrect_v_output, atol=1e-2, rtol=1e-2)

    logger.info("test_mqa_kv_head_selection PASSED")


def test_mqa_multi_token_kv_head_selection_2d_query():
    """
    PARAMETER STRUCT DEBUGGING TEST
    This test is specifically configured to diagnose struct layout/marshalling issues
    between C++ and Metal. It only checks the first 5 parameters in PagedAttentionParams.
    """
    # Test configuration
    num_tokens = 5  # Multiple tokens to test consistent KV-head selection
    cfg_head_dim = 4
    cfg_tokens_per_page = 64
    cfg_num_kv_heads = 4  # Multiple KV heads

    # Create 2D queries with shape [num_tokens, cfg_head_dim]
    # For 2D queries, the C++ primitive sets params->num_q_heads = 1 internally
    py_queries = mx.array([[1.0] * cfg_head_dim] * num_tokens, dtype=mx.float16)

    # Create K-cache pool with zeros (not used by debug kernel)
    num_physical_pages = 1
    k_cache_shape = (num_physical_pages, cfg_tokens_per_page, cfg_num_kv_heads, cfg_head_dim)
    py_k_cache_pool = mx.zeros(k_cache_shape, dtype=mx.float16)
    py_v_cache_pool = mx.zeros_like(py_k_cache_pool)

    # Add V-cache with distinct values for each token position
    # All queries use KV-head 0 in MQA mode when queries are 2D
    py_v_cache_pool[0, 0, 0, :] = mx.array([10.0, 20.0, 30.0, 40.0], dtype=mx.float16)

    py_page_table = mx.array([[0]], dtype=mx.uint32)  # Simple page table
    py_sequence_lengths = mx.array([cfg_tokens_per_page], dtype=mx.int32)  # Plenty of tokens in the sequence

    # All query tokens map to sequence 0
    py_query_to_seq_map = mx.zeros(num_tokens, dtype=mx.int32)

    # All query tokens look at history position 0
    py_query_token_offset = mx.ones(num_tokens, dtype=mx.int32)

    # Calculate expected scale factor for verification
    denominator = mx.sqrt(mx.array(float(cfg_head_dim))).item()
    assert isinstance(denominator, float)
    1.0 / denominator

    # Call the kernel with our debug version
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

    # This test was originally for parameter marshalling debug, but now we have actual output

    # Output is now full attention format [num_tokens, cfg_head_dim]
    expected_output_shape = (num_tokens, cfg_head_dim)

    # For each query token, we expect the V-vector from KV head 0
    # Since we only have one history token per query item, softmax prob is 1.0
    # So the output should be exactly the V-vector from KV head 0
    # Create a stack of the same V-vector for each token
    single_v = mx.array([10.0, 20.0, 30.0, 40.0], dtype=mx.float16)
    expected_v_output = mx.stack([single_v] * num_tokens)

    logger.info(f"Test MQA 2D: Output shape: {output_arr.shape}")
    logger.info(f"Test MQA 2D: Expected shape: {expected_output_shape}")
    logger.info(f"Test MQA 2D: Expected output: {expected_v_output}")
    logger.info(f"Test MQA 2D: Actual output: {output_arr}")

    # Check shape and values
    assert output_arr.shape == expected_output_shape
    assert mx.allclose(output_arr, expected_v_output, atol=1e-2, rtol=1e-2)

    # Only verify dtype - this test is purely for debug information
    assert output_arr.dtype == mx.float16

    logger.info("test_mqa_multi_token_kv_head_selection_2d_query - PARAMETER DEBUG TEST COMPLETED")
