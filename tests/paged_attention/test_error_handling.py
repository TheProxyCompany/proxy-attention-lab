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

    total_items = 2  # Two threads in this test
    expected_output_shape = (total_items * 2,)  # With the new format, we have [items * 2]

    # Extract the max scores from the first plane of the output
    max_scores = output_arr[:total_items]

    # Expect all zeros for max scores
    expected_max_scores = mx.zeros(total_items, dtype=mx.float16)

    logger.info(f"Test: Expected output shape: {expected_output_shape}")
    logger.info(f"Test: Actual output shape: {output_arr.shape}")
    logger.info(f"Test: Expected max scores: {expected_max_scores}")
    logger.info(f"Test: Actual max scores: {max_scores}")

    assert output_arr.shape == expected_output_shape
    assert output_arr.dtype == mx.float16

    # Check just the max scores
    assert mx.allclose(max_scores, expected_max_scores, atol=1e-3)


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

    total_items = 2  # Two threads in this test
    expected_output_shape = (total_items * 2,)  # With the new format, we have [items * 2]

    # Extract the max scores from the first plane of the output
    max_scores = output_arr[:total_items]

    # Expect all zeros for max scores
    expected_max_scores = mx.zeros(total_items, dtype=mx.float16)

    logger.info(f"Test: Expected output shape: {expected_output_shape}")
    logger.info(f"Test: Actual output shape: {output_arr.shape}")
    logger.info(f"Test: Expected max scores: {expected_max_scores}")
    logger.info(f"Test: Actual max scores: {max_scores}")

    assert output_arr.shape == expected_output_shape
    assert output_arr.dtype == mx.float16

    # Check just the max scores
    assert mx.allclose(max_scores, expected_max_scores, atol=1e-3)


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

    total_items = 2  # Two threads in this test
    expected_output_shape = (total_items * 2,)  # With the new format, we have [items * 2]

    # Extract the max scores and sum_exp scores
    max_scores = output_arr[:total_items]
    sum_exp_scores = output_arr[total_items:]

    # Expect all zeros for both max scores and sum_exp scores
    expected_zeros = mx.zeros(total_items, dtype=mx.float16)

    logger.info(f"Test: Expected output shape: {expected_output_shape}")
    logger.info(f"Test: Actual output shape: {output_arr.shape}")
    logger.info(f"Test: Expected max scores: {expected_zeros}")
    logger.info(f"Test: Actual max scores: {max_scores}")
    logger.info(f"Test: Expected sum_exp scores: {expected_zeros}")
    logger.info(f"Test: Actual sum_exp scores: {sum_exp_scores}")

    assert output_arr.shape == expected_output_shape
    assert output_arr.dtype == mx.float16

    # Check the max scores
    assert mx.allclose(max_scores, expected_zeros, atol=1e-3)

    # Check the sum_exp scores
    assert mx.allclose(sum_exp_scores, expected_zeros, atol=1e-3)


def test_large_head_dimension():
    """
    Tests paged attention with a head dimension larger than 128.

    With the dynamic threadgroup memory allocation, head dimensions
    larger than the previous hard-coded limit of 128 should now work.
    """
    # Use head_dim = 192, which exceeds the previous limit of 128
    # Also ensure it's a multiple of 4 as required by the vectorized kernel
    cfg_head_dim = 192
    cfg_tokens_per_page = 64
    cfg_num_kv_heads = 1

    # Create 2D queries with shape [num_q_threads, cfg_head_dim]
    num_queries = 4
    py_queries = mx.random.normal((num_queries, cfg_head_dim), dtype=mx.float16)

    num_physical_pages = 2
    k_cache_shape = (num_physical_pages, cfg_tokens_per_page, cfg_num_kv_heads, cfg_head_dim)
    py_k_cache_pool = mx.random.normal(k_cache_shape, dtype=mx.float16)
    py_v_cache_pool = mx.random.normal(k_cache_shape, dtype=mx.float16)

    # Simple page table with two sequences
    py_page_table = mx.array(
        [
            [0],  # Sequence 0 uses physical page 0
            [1],  # Sequence 1 uses physical page 1
        ],
        dtype=mx.uint32,
    )
    py_sequence_lengths = mx.array([32, 32], dtype=mx.int32)

    # Map queries to sequences
    py_query_to_seq_map = mx.array([0, 0, 1, 1], dtype=mx.int32)
    py_query_token_offset = mx.array([0, 1, 0, 1], dtype=mx.int32)

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

    # Verify output shape and data types
    total_items = num_queries
    expected_output_shape = (total_items * 2,)  # [items * 2] for planar layout

    assert output_arr.shape == expected_output_shape
    assert output_arr.dtype == mx.float16

    # Extract the max scores and sum_exp scores
    max_scores = output_arr[:total_items]
    sum_exp_scores = output_arr[total_items:]

    # Verify scores are finite and sum_exp scores are non-negative
    assert mx.isfinite(max_scores).all(), "Max scores should be finite"
    assert mx.isfinite(sum_exp_scores).all(), "Sum-exp scores should be finite"
    assert (sum_exp_scores >= 0).all(), "Sum-exp scores should be non-negative"

    logger.info(f"Large head dimension test passed with head_dim={cfg_head_dim}")
