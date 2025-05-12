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
    """Seq parallel fetch; expect 111 / 222 outputs."""
    num_q_threads = 2
    cfg_tokens_per_page = 64
    cfg_num_kv_heads = 1
    cfg_head_dim = 4
    cfg_max_logical_blocks_per_seq_in_pagetable = 2
    py_queries = mx.array([100.0, 200.0], dtype=mx.float16)
    assert py_queries.shape[0] == num_q_threads
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
    expected_output = mx.array([111.0, 222.0], dtype=mx.float16)
    logger.info(f"Test: Expected output: {expected_output}")
    logger.info(f"Test: Actual output  : {output_arr}")
    assert output_arr.shape == expected_output.shape
    assert output_arr.dtype == expected_output.dtype
    assert mx.allclose(output_arr, expected_output, atol=1e-3)


def test_fetch_entire_k_vector_for_specific_token_slot():
    """Sum all K elements per token; expect 110 / 226 outputs."""
    num_q_threads = 2
    cfg_tokens_per_page = 64
    cfg_num_kv_heads = 1
    cfg_head_dim = 4
    cfg_max_logical_blocks_per_seq_in_pagetable = 2
    py_queries = mx.array([100.0, 200.0], dtype=mx.float16)
    assert py_queries.shape[0] == num_q_threads
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
    expected_output = mx.array([110.0, 226.0], dtype=mx.float16)
    logger.info(f"Test: Expected output: {expected_output}")
    logger.info(f"Test: Actual output  : {output_arr}")
    assert output_arr.shape == expected_output.shape
    assert output_arr.dtype == expected_output.dtype
    assert mx.allclose(output_arr, expected_output, atol=1e-3)


def test_fetch_k_vector_from_variable_token_slot_in_first_logical_block():
    """Variable token slot access; expect 110 / 226 outputs."""
    num_q_threads = 2
    cfg_tokens_per_page = 64
    cfg_num_kv_heads = 1
    cfg_head_dim = 4
    cfg_max_logical_blocks_per_seq_in_pagetable = 2
    py_queries = mx.array([100.0, 200.0], dtype=mx.float16)
    assert py_queries.shape[0] == num_q_threads
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
    expected_output = mx.array([110.0, 226.0], dtype=mx.float16)
    logger.info(f"Test: Expected output: {expected_output}")
    logger.info(f"Test: Actual output  : {output_arr}")
    assert output_arr.shape == expected_output.shape
    assert output_arr.dtype == expected_output.dtype
    assert mx.allclose(output_arr, expected_output, atol=1e-3)


def test_fetch_k_vector_from_multiple_kv_heads():
    """GQA: multiple Q heads map to KV heads; expect 110/226 in output."""
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
    expected_output = mx.zeros((num_tokens, num_q_heads, cfg_head_dim), dtype=mx.float16)
    expected_output[0, 0, :] = 110.0
    expected_output[0, 1, :] = 226.0
    logger.info(f"Test: Expected output: {expected_output}")
    logger.info(f"Test: Actual output  : {output_arr}")
    assert output_arr.shape == expected_output.shape
    assert output_arr.dtype == expected_output.dtype
    assert mx.allclose(output_arr, expected_output, atol=1e-3)


def test_invalid_physical_page_id_in_page_table():
    """Invalid page id returns zero for affected thread."""
    cfg_tokens_per_page = 64
    cfg_num_kv_heads = 1
    cfg_head_dim = 4
    py_queries = mx.array([100.0, 200.0], dtype=mx.float16)
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
    expected_thread0_output = 100.0 + 0.0
    expected_thread1_output = 0.0
    expected_output = mx.array([expected_thread0_output, expected_thread1_output], dtype=mx.float16)
    logger.info(f"Test: Expected output: {expected_output}")
    logger.info(f"Test: Actual output  : {output_arr}")
    assert output_arr.shape == expected_output.shape
    assert output_arr.dtype == expected_output.dtype
    assert mx.allclose(output_arr[1], mx.array([expected_thread1_output], dtype=mx.float16), atol=1e-3)


def test_negative_query_token_offset():
    """Negative token offset returns zero for affected thread."""
    cfg_tokens_per_page = 64
    cfg_num_kv_heads = 1
    cfg_head_dim = 4
    py_queries = mx.array([100.0, 200.0], dtype=mx.float16)
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
    expected_thread0_output = 0.0
    expected_thread1_output = 200.0 + 0.0
    expected_output = mx.array([expected_thread0_output, expected_thread1_output], dtype=mx.float16)
    logger.info(f"Test: Expected output: {expected_output}")
    logger.info(f"Test: Actual output  : {output_arr}")
    assert output_arr.shape == expected_output.shape
    assert output_arr.dtype == expected_output.dtype
    assert mx.allclose(output_arr[0], mx.array([expected_thread0_output], dtype=mx.float16), atol=1e-3)


def test_invalid_seq_idx_in_query_map():
    """Invalid seq idx in query map returns zero for affected thread."""
    cfg_tokens_per_page = 64
    cfg_num_kv_heads = 1
    cfg_head_dim = 4
    py_queries = mx.array([100.0, 200.0], dtype=mx.float16)
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
    expected_thread0_output = 100.0 + 0.0
    expected_thread1_output = 0.0
    expected_output = mx.array([expected_thread0_output, expected_thread1_output], dtype=mx.float16)
    logger.info(f"Test: Expected output: {expected_output}")
    logger.info(f"Test: Actual output  : {output_arr}")
    assert output_arr.shape == expected_output.shape
    assert output_arr.dtype == expected_output.dtype
    assert mx.allclose(output_arr[1], mx.array([expected_thread1_output], dtype=mx.float16), atol=1e-3)


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
