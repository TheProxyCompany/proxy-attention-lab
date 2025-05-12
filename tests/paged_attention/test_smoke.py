import logging

import mlx.core as mx

from proxy_attention_lab import paged_attention

logger = logging.getLogger(__name__)


def test_paged_attention_smoke():
    num_queries = 4
    num_q_heads = 2
    head_dim = 8

    num_total_pages = 2
    tokens_per_page = 64
    num_kv_heads = 2

    num_sequences_in_batch = 1  # For this smoke test, let's keep it simple with one sequence
    max_logical_blocks_per_seq_val = 1  # Sequence uses at most 1 logical block for this simple test

    mock_queries = mx.random.normal((num_queries, num_q_heads, head_dim)).astype(mx.float16)
    mock_k_cache_pool = mx.random.normal((num_total_pages, tokens_per_page, num_kv_heads, head_dim)).astype(mx.float16)
    mock_v_cache_pool = mx.random.normal((num_total_pages, tokens_per_page, num_kv_heads, head_dim)).astype(mx.float16)

    # Shape: [num_sequences_in_batch, max_logical_blocks_per_seq_val]
    # Content: physical page IDs. For this smoke test, just point to physical page 0.
    page_table_content = [[0] * max_logical_blocks_per_seq_val for _ in range(num_sequences_in_batch)]
    mock_page_table = mx.array(page_table_content, dtype=mx.uint32)
    # Expected shape e.g., (1, 1) if num_sequences_in_batch=1, max_logical_blocks_per_seq_val=1

    # 4. Metadata Arrays
    mock_sequence_lengths = mx.array([tokens_per_page // 2] * num_sequences_in_batch, dtype=mx.int32)
    # query_to_seq_map should have num_queries elements, all pointing to seq 0 if num_sequences_in_batch is 1
    mock_query_to_seq_map = mx.zeros(num_queries, dtype=mx.int32)
    mock_query_token_offset = mx.arange(num_queries, dtype=mx.int32)

    logger.info(f"Smoke test inputs prepared: Q_shape={mock_queries.shape}, PageTable_shape={mock_page_table.shape}")

    try:
        out = paged_attention(
            mock_queries,
            mock_k_cache_pool,
            mock_v_cache_pool,
            mock_page_table,
            mock_sequence_lengths,
            mock_query_to_seq_map,
            mock_query_token_offset,
        )
        mx.eval(out)

        assert out.shape == mock_queries.shape, (
            f"Output shape {out.shape} does not match query shape {mock_queries.shape}"
        )
        assert out.dtype == mock_queries.dtype, (
            f"Output dtype {out.dtype} does not match query dtype {mock_queries.dtype}"
        )
        logger.info(f"Paged attention smoke test passed. Output shape: {out.shape}, dtype: {out.dtype}")

    except Exception as e:
        logger.error(f"Paged attention smoke test failed: {e}", exc_info=True)
        raise
