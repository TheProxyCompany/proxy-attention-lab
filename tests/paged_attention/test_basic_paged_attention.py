import logging

import mlx.core as mx

from proxy_attention_lab import paged_attention

logger = logging.getLogger(__name__)

"""
Unit tests for Proxy Attention Lab (PAL) paged attention kernel.

Covered scenarios
-----------------
• Correct K vector fetch and reduction for first token of distinct sequences
• Full vector sum for identical logical block across sequences
• Variable token slot access within same logical block
• GQA mapping: multiple Q heads → KV heads
• Runtime bounds checks (invalid page id, negative offset, invalid seq index)
• Primitive level GQA configuration validation
"""


def test_fetch_k_vector_element_for_first_token_of_sequence():
    """Seq parallel fetch; expect dot product of Q with K with scale applied."""
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
    py_v_cache_pool = mx.zeros_like(py_k_cache_pool)
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
    py_query_token_offset = mx.array([0, 0], dtype=mx.int32)
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

    # Debug the actual output shape
    logger.info(f"DEBUG: output_arr shape = {output_arr.shape}, type = {type(output_arr)}")
    # For 2D queries, the output is [num_q_threads] (scalar per thread)
    expected_output_value = mx.array([550.0, 2200.0], dtype=mx.float16)
    expected_output_shape = (num_q_threads,)

    logger.info(f"Test: Expected output values: {expected_output_value}")
    logger.info(f"Test: Actual output  : {output_arr}")
    assert output_arr.shape == expected_output_shape
    assert output_arr.dtype == mx.float16

    # Check each scalar output value
    assert mx.allclose(output_arr, expected_output_value, atol=1e-3)


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
    py_v_cache_pool = mx.zeros_like(py_k_cache_pool)
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
    py_query_token_offset = mx.array([0, 0], dtype=mx.int32)
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

    # For 2D queries, the output is [num_q_threads] (scalar per thread)
    expected_output_value = mx.array([500.0, 2600.0], dtype=mx.float16)
    expected_output_shape = (num_q_threads,)

    logger.info(f"Test: Expected output values: {expected_output_value}")
    logger.info(f"Test: Actual output  : {output_arr}")
    assert output_arr.shape == expected_output_shape
    assert output_arr.dtype == mx.float16

    # Check each scalar output value
    assert mx.allclose(output_arr, expected_output_value, atol=1e-3)


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
    for i in range(cfg_head_dim):
        py_k_cache_pool[0, 3, 0, i] = float(i + 1)
        py_k_cache_pool[0, 7, 0, i] = float(i + 5)
    py_v_cache_pool = mx.zeros_like(py_k_cache_pool)
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
    py_query_token_offset = mx.array([3, 7], dtype=mx.int32)
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

    # Scale = 1 / sqrt(cfg_head_dim)
    numerator = mx.array(float(cfg_head_dim)).item()
    assert isinstance(numerator, float)

    # For 2D queries, the output is [num_q_threads] (scalar per thread)
    expected_output_value = mx.array([500.0, 2600.0], dtype=mx.float16)
    expected_output_shape = (num_q_threads,)

    logger.info(f"Test: Expected output values: {expected_output_value}")
    logger.info(f"Test: Actual output  : {output_arr}")
    assert output_arr.shape == expected_output_shape
    assert output_arr.dtype == mx.float16

    # Check each scalar output value
    assert mx.allclose(output_arr, expected_output_value, atol=1e-3)


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
    py_query_token_offset = mx.array([token_slot, token_slot], dtype=mx.int32)
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


def test_invalid_physical_page_id_in_page_table():
    """Invalid page id returns zero for affected thread."""
    cfg_tokens_per_page = 64
    cfg_num_kv_heads = 1
    cfg_head_dim = 4

    # Create 2D queries with shape [num_q_threads, cfg_head_dim]
    py_queries = mx.zeros((2, cfg_head_dim), dtype=mx.float16)
    py_queries[0, :] = 100.0
    py_queries[1, :] = 200.0

    num_physical_pages = 2
    k_cache_shape = (num_physical_pages, cfg_tokens_per_page, cfg_num_kv_heads, cfg_head_dim)
    py_k_cache_pool = mx.zeros(k_cache_shape, dtype=mx.float16)
    py_v_cache_pool = mx.zeros_like(py_k_cache_pool)
    py_page_table = mx.array(
        [
            [0, 99],
            [2, 88],
        ],
        dtype=mx.uint32,
    )
    py_sequence_lengths = mx.array([10, 5], dtype=mx.int32)
    py_query_to_seq_map = mx.array([0, 1], dtype=mx.int32)
    py_query_token_offset = mx.array([0, 0], dtype=mx.int32)
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
    expected_thread0_output = 0.0  # Dot product with all zeros in K
    expected_thread1_output = 0.0  # Early exit, zero output
    expected_output_value = mx.array([expected_thread0_output, expected_thread1_output], dtype=mx.float16)
    expected_output_shape = (2,)  # For 2D queries, output is 1D (scalar per thread)

    logger.info(f"Test: Expected output values: {expected_output_value}")
    logger.info(f"Test: Actual output  : {output_arr}")
    assert output_arr.shape == expected_output_shape
    assert output_arr.dtype == mx.float16

    # Check each scalar output value
    assert mx.allclose(output_arr, expected_output_value, atol=1e-3)


def test_negative_query_token_offset():
    """Negative token offset returns zero for affected thread."""
    cfg_tokens_per_page = 64
    cfg_num_kv_heads = 1
    cfg_head_dim = 4

    # Create 2D queries with shape [num_q_threads, cfg_head_dim]
    py_queries = mx.zeros((2, cfg_head_dim), dtype=mx.float16)
    py_queries[0, :] = 100.0
    py_queries[1, :] = 200.0

    num_physical_pages = 2
    k_cache_shape = (num_physical_pages, cfg_tokens_per_page, cfg_num_kv_heads, cfg_head_dim)
    py_k_cache_pool = mx.zeros(k_cache_shape, dtype=mx.float16)
    py_v_cache_pool = mx.zeros_like(py_k_cache_pool)
    py_page_table = mx.array(
        [
            [0, 99],
            [1, 88],
        ],
        dtype=mx.uint32,
    )
    py_sequence_lengths = mx.array([10, 5], dtype=mx.int32)
    py_query_to_seq_map = mx.array([0, 1], dtype=mx.int32)
    py_query_token_offset = mx.array([-1, 0], dtype=mx.int32)
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
    expected_thread0_output = 0.0  # Early exit, zero output
    expected_thread1_output = 0.0  # Dot product with all zeros in K
    expected_output_value = mx.array([expected_thread0_output, expected_thread1_output], dtype=mx.float16)
    expected_output_shape = (2,)  # For 2D queries, output is 1D (scalar per thread)

    logger.info(f"Test: Expected output values: {expected_output_value}")
    logger.info(f"Test: Actual output  : {output_arr}")
    assert output_arr.shape == expected_output_shape
    assert output_arr.dtype == mx.float16

    # Check each scalar output value
    assert mx.allclose(output_arr, expected_output_value, atol=1e-3)


def test_invalid_seq_idx_in_query_map():
    """Invalid seq idx in query map returns zero for affected thread."""
    cfg_tokens_per_page = 64
    cfg_num_kv_heads = 1
    cfg_head_dim = 4

    # Create 2D queries with shape [num_q_threads, cfg_head_dim]
    py_queries = mx.zeros((2, cfg_head_dim), dtype=mx.float16)
    py_queries[0, :] = 100.0
    py_queries[1, :] = 200.0

    num_physical_pages = 2
    k_cache_shape = (num_physical_pages, cfg_tokens_per_page, cfg_num_kv_heads, cfg_head_dim)
    py_k_cache_pool = mx.zeros(k_cache_shape, dtype=mx.float16)
    py_v_cache_pool = mx.zeros_like(py_k_cache_pool)
    py_page_table = mx.array(
        [
            [0, 99],
            [1, 88],
        ],
        dtype=mx.uint32,
    )
    py_sequence_lengths = mx.array([10, 5], dtype=mx.int32)
    py_query_to_seq_map = mx.array([0, 2], dtype=mx.int32)
    py_query_token_offset = mx.array([0, 0], dtype=mx.int32)
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
    expected_thread0_output = 0.0  # Dot product with all zeros in K
    expected_thread1_output = 0.0  # Early exit, zero output
    expected_output_value = mx.array([expected_thread0_output, expected_thread1_output], dtype=mx.float16)
    expected_output_shape = (2,)  # For 2D queries, output is 1D (scalar per thread)

    logger.info(f"Test: Expected output values: {expected_output_value}")
    logger.info(f"Test: Actual output  : {output_arr}")
    assert output_arr.shape == expected_output_shape
    assert output_arr.dtype == mx.float16

    # Check each scalar output value
    assert mx.allclose(output_arr, expected_output_value, atol=1e-3)


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

    py_v_cache_pool = mx.zeros_like(py_k_cache_pool)
    py_page_table = mx.array([[0]], dtype=mx.uint32)  # Seq 0, LogBlock 0 -> PhysPage 0. Shape (1,1)
    # C++ primitive will set params.max_logical_blocks_per_seq = 1
    # And params.num_sequences_in_batch = 1

    # All threads map to sequence 0, but target different token offsets
    py_sequence_lengths = mx.array([cfg_tokens_per_page], dtype=mx.int32)
    py_query_to_seq_map = mx.zeros(num_q_threads, dtype=mx.int32)  # All map to seq 0
    py_query_token_offset = mx.array([0, 1], dtype=mx.int32)  # Thread 0 targets tok 0, Thread 1 targets tok 1

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

    denominator = mx.sqrt(mx.array(float(cfg_head_dim))).item()
    assert isinstance(denominator, float)
    py_scale = 1.0 / denominator
    score0 = (1 * 1 + 2 * 1 + 1 * 1 + 2 * 1) * py_scale  # = 6 * 0.5 = 3.0
    score1 = (3 * 2 + 4 * 2 + 3 * 2 + 4 * 2) * py_scale  # = 28 * 0.5 = 14.0

    expected_output = mx.array([score0, score1], dtype=mx.float16)  # Output is 1D [num_q_threads]

    logger.info(f"DEBUG 2D regression test: output_arr shape = {output_arr.shape}, values = {output_arr}")
    logger.info(f"DEBUG 2D regression test: expected shape = {expected_output.shape}, values = {expected_output}")

    assert output_arr.shape == expected_output.shape
    assert mx.allclose(output_arr, expected_output, atol=1e-2, rtol=1e-2)

    logger.info("test_correct_token_processing_for_2d_queries_variable_offsets PASSED")


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
    import pytest

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


def test_dot_product_q_with_single_k_vector():
    """
    Tests Q.K^T * scale for a single Q-vector and a single K-vector.
    Kernel fetches full Q-vector and full K-vector (from logical_block_0, token_slot_0, kv_head_0)
    and computes their scaled dot product.
    Output shape will be [NumTestTokens, NumQHeads] (scalar score per Q-head).
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

    # 3. V-Cache Pool (not used yet)
    py_v_cache_pool = mx.zeros_like(py_k_cache_pool)

    # 4. Page Table: [NumBatchSequences, MaxLogicalBlocksPerSeq]
    #    One sequence in batch for this test. Logical block 0 maps to physical page 0.
    num_sequences_in_batch_for_test = 1
    py_page_table = mx.array([[0]], dtype=mx.uint32)  # Shape (1,1)
    assert py_page_table.shape == (num_sequences_in_batch_for_test, cfg_max_logical_blocks_per_seq_in_pagetable)

    # 5. sequence_lengths: [NumBatchSequences]
    py_sequence_lengths = mx.array([cfg_tokens_per_page], dtype=mx.int32)  # Seq 0 has enough tokens

    # 6. query_to_seq_map: [NumTestTokens * NumQHeads]. All map to sequence 0.
    # Dispatch grid will be (NumTestTokens * NumQHeads) threads.
    num_dispatch_threads = num_test_tokens * num_q_heads
    py_query_to_seq_map = mx.zeros(num_dispatch_threads, dtype=mx.int32)

    # 7. query_token_offset: [NumTestTokens * NumQHeads].
    #    All threads target K-data from token_slot 0 of the sequence.
    py_query_token_offset = mx.zeros(num_dispatch_threads, dtype=mx.int32)

    # --- Call Op ---
    # Output shape will be [NumTestTokens, NumQHeads]
    # Based on the output_shapes() method in PagedAttentionPrimitive
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

    # --- Expected Output ---
    # Scale = 1 / sqrt(cfg_head_dim)
    denominator = mx.sqrt(mx.array(float(cfg_head_dim))).item()
    assert isinstance(denominator, float)
    py_scale = 1.0 / denominator

    # Q_head0: [1,2,3,4], K_head0: [1,1,1,1] -> dot = 10. Score = 10 * scale
    score0 = (1.0 * 1.0 + 2.0 * 1.0 + 3.0 * 1.0 + 4.0 * 1.0) * py_scale
    # Q_head1: [0.5,1,1.5,2], K_head1: [2,2,2,2] -> dot = (0.5+1+1.5+2)*2 = 5*2 = 10. Score = 10 * scale
    score1 = (0.5 * 2.0 + 1.0 * 2.0 + 1.5 * 2.0 + 2.0 * 2.0) * py_scale

    # Output shape should be (num_test_tokens, num_q_heads)
    # For 3D queries [NumTokens, NumQHeads, HeadDim], output is [NumTokens, NumQHeads]
    expected_output_shape = (num_test_tokens, num_q_heads)
    expected_output = mx.array([score0, score1], dtype=mx.float16).reshape(expected_output_shape)

    # Calculate expected dot product manually for clarity
    q0_dot_k0 = 1.0 * 1.0 + 2.0 * 1.0 + 3.0 * 1.0 + 4.0 * 1.0  # = 10
    q1_dot_k1 = 0.5 * 2.0 + 1.0 * 2.0 + 1.5 * 2.0 + 2.0 * 2.0  # = 10

    logger.info(f"DEBUG: output_arr shape = {output_arr.shape}, type = {type(output_arr)}")
    logger.info(f"Test: Q0: {py_queries[0, 0, :]}, K0 chosen: {py_k_cache_pool[0, 0, 0, :]}")
    logger.info(f"Test: Q0·K0 raw dot product = {q0_dot_k0}, scale={py_scale}, Expected Score0 = {score0}")
    logger.info(f"Test: Q1: {py_queries[0, 1, :]}, K1 chosen: {py_k_cache_pool[0, 0, 1, :]}")
    logger.info(f"Test: Q1·K1 raw dot product = {q1_dot_k1}, scale={py_scale}, Expected Score1 = {score1}")
    logger.info(f"Test: Expected output: {expected_output}")
    logger.info(f"Test: Actual output: {output_arr}")

    assert output_arr.shape == expected_output_shape, f"Shape: {output_arr.shape} vs {expected_output_shape}"
    assert output_arr.dtype == expected_output.dtype
    assert mx.allclose(output_arr, expected_output, atol=1e-2, rtol=1e-2)  # Increased atol/rtol slightly for float sums

    logger.info("test_dot_product_q_with_single_k_vector PASSED")
