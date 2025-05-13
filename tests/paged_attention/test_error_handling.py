import logging

import mlx.core as mx

from proxy_attention_lab import paged_attention

logger = logging.getLogger(__name__)


def test_invalid_physical_page_id_in_page_table():
    """
    Tests handling of invalid physical page ID in the page table.

    When the page table contains a physical page ID that exceeds the bounds of
    the available physical pages in the cache pool, the kernel should safely
    skip those positions and return a zero score.
    """
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

    # Page table with invalid physical page ID (2) for the second sequence
    # Valid range is [0, 1] since num_physical_pages = 2
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
    """
    Tests handling of negative query token offset.

    When a query token has a negative logical offset, the kernel should safely
    return a zero score for that query without accessing any K-vectors.
    """
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

    # First query has a negative token offset
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

    expected_thread0_output = 0.0  # Early exit due to negative offset, zero output
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
    """
    Tests handling of invalid sequence index in query map.

    When a query maps to a sequence index that exceeds the number of sequences
    in the batch, the kernel should safely return a zero score for that query
    without accessing any K-vectors.
    """
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

    # Second query maps to an invalid sequence index (2)
    # Valid range is [0, 1] since we have only 2 sequences in the batch
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
    expected_thread1_output = 0.0  # Early exit due to invalid sequence index, zero output
    expected_output_value = mx.array([expected_thread0_output, expected_thread1_output], dtype=mx.float16)
    expected_output_shape = (2,)  # For 2D queries, output is 1D (scalar per thread)

    logger.info(f"Test: Expected output values: {expected_output_value}")
    logger.info(f"Test: Actual output  : {output_arr}")
    assert output_arr.shape == expected_output_shape
    assert output_arr.dtype == mx.float16

    # Check each scalar output value
    assert mx.allclose(output_arr, expected_output_value, atol=1e-3)
