import logging

import mlx.core as mx

from proxy_attention_lab import paged_attention

logger = logging.getLogger(__name__)


def test_paged_attention_smoke():
    """
    Basic smoke test for the paged_attention operation.
    Checks if it can be called with valid shapes and dtypes,
    and produces an output of the expected shape and dtype without crashing.
    """
    # Define some reasonably small but valid dimensions for a smoke test
    num_queries = 4
    num_q_heads = 2  # For Q shape
    head_dim = 8  # For Q shape and K/V shape

    num_total_pages = 2
    tokens_per_page = 64  # Should match what the kernel/params might expect
    num_kv_heads = 2  # For K/V shape

    num_sequences_in_batch = 1  # Simplest case for metadata arrays
    max_logical_blocks_per_seq = 1  # Simplest case for page_table

    # 1. Queries: [TotalQueryTokens, NumQHeads, HeadDim]
    #    For this smoke test, TotalQueryTokens can be simple.
    #    Let's assume queries are already projected and shaped.
    #    If your primitive expects a different Q shape (e.g. flat), adjust here.
    #    Given current primitive logic: q.shape(0) is grid_dim_x,
    #    params.head_dim = q.shape(-1), params.num_q_heads = q.shape(-2) or 1.
    #    Let's make Q [num_queries, num_q_heads, head_dim]
    mock_queries = mx.random.normal((num_queries, num_q_heads, head_dim)).astype(mx.float16)

    # 2. K/V Cache Pools: [NumTotalPages, TokensPerPage, NumKVHeads, HeadDim]
    mock_k_cache_pool = mx.random.normal((num_total_pages, tokens_per_page, num_kv_heads, head_dim)).astype(mx.float16)
    mock_v_cache_pool = mx.random.normal((num_total_pages, tokens_per_page, num_kv_heads, head_dim)).astype(mx.float16)

    # 3. Page Table: [NumSequencesInBatch, MaxLogicalBlocksPerSequence] -> physical_page_id
    #    For the smoke test, content doesn't hugely matter as long as indices are valid (0 or 1 for num_total_pages=2)
    #    The primitive derives params.max_logical_blocks_per_seq from page_table.size() if 1D, or shape(1) if 2D.
    #    Let's make it 1D for simplicity in this smoke test, matching primitive's current derivation.
    mock_page_table = mx.array([0] * max_logical_blocks_per_seq, dtype=mx.uint32)  # All point to page 0

    # 4. Metadata Arrays (shapes based on num_sequences_in_batch or num_queries)
    #    sequence_lengths: [NumSequencesInBatch]
    mock_sequence_lengths = mx.array([tokens_per_page // 2] * num_sequences_in_batch, dtype=mx.int32)  # Half full
    #    query_to_seq_map: [TotalQueryTokens] -> maps to seq_idx (0 for this test)
    mock_query_to_seq_map = mx.zeros(num_queries, dtype=mx.int32)
    #    query_token_offset: [TotalQueryTokens] -> logical offset within sequence
    mock_query_token_offset = mx.arange(num_queries, dtype=mx.int32)

    logger.info(f"Smoke test inputs prepared: Q_shape={mock_queries.shape}")

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
        mx.eval(out)  # Ensure kernel execution

        # Basic assertions: shape and dtype
        assert out.shape == mock_queries.shape, (
            f"Output shape {out.shape} does not match query shape {mock_queries.shape}"
        )
        assert out.dtype == mock_queries.dtype, (
            f"Output dtype {out.dtype} does not match query dtype {mock_queries.dtype}"
        )

        logger.info(f"Paged attention smoke test passed. Output shape: {out.shape}, dtype: {out.dtype}")

    except Exception as e:
        logger.error(f"Paged attention smoke test failed: {e}", exc_info=True)
        raise  # Re-raise the exception to fail the test
