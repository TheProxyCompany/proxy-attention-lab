import logging

import mlx.core as mx
import numpy as np
import pytest

from proxy_attention_lab import calculate_page_size, paged_attention

logger = logging.getLogger(__name__)


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
def test_paged_attention_determinism_prefill(dtype) -> None:
    """Test that paged_attention prefill output is deterministic for identical inputs.

    This test configures a moderately complex prefill scenario with multiple sequences
    and varying lengths, calls paged_attention twice with the same inputs,
    and asserts that the resulting output arrays are identical.
    """
    logger.info(f"Test: {test_paged_attention_determinism_prefill.__name__} (dtype={dtype})")

    # --- Configuration ---
    num_q_heads = 32
    num_kv_heads = 16
    head_dim = 128

    # Use the PAL helper to get the optimal tokens_per_page (D_s for prefill)
    tokens_per_page = calculate_page_size(head_dim, num_q_heads, num_kv_heads)
    logger.info(f"  Calculated tokens_per_page (D_s): {tokens_per_page}")

    # Batch setup for prefill
    py_sequence_lengths_list = [
        tokens_per_page * 2,  # Uses 2 full pages/Q-blocks
        tokens_per_page // 2 + 1,  # Uses 1 page/Q-block, not full
        1,  # Minimal case
        tokens_per_page,  # Uses 1 full page/Q-block
    ]

    if tokens_per_page > 32:
        py_sequence_lengths_list = [min(35, tokens_per_page + 3), min(10, tokens_per_page - 2), 1]
    else:
        py_sequence_lengths_list = [tokens_per_page * 2, tokens_per_page // 2 + 1, 1]

    py_sequence_lengths = mx.array(py_sequence_lengths_list, dtype=mx.int32)
    num_sequences_in_batch = len(py_sequence_lengths_list)
    total_query_tokens_in_batch = int(mx.sum(py_sequence_lengths).item())

    logger.info(f"  Batch setup: {num_sequences_in_batch} sequences, lengths: {py_sequence_lengths_list}")
    logger.info(f"  Total query tokens in batch: {total_query_tokens_in_batch}")

    # Max logical blocks needed for any sequence in the batch
    max_logical_blocks_per_seq = 0
    if total_query_tokens_in_batch > 0:
        max_logical_blocks_per_seq = int(mx.ceil(mx.max(py_sequence_lengths) / tokens_per_page).item())
    if max_logical_blocks_per_seq == 0:
        max_logical_blocks_per_seq = 1

    # Seed for reproducibility
    mx.random.seed(11)

    # --- Setup Test Inputs (Identical for both calls) ---

    # 1. Queries: Shape [TotalNumQueryTokensInBatch, NumQHeads, HeadDim]
    if total_query_tokens_in_batch == 0:
        py_queries = mx.zeros((0, num_q_heads, head_dim), dtype=dtype)
    else:
        queries_shape = (total_query_tokens_in_batch, num_q_heads, head_dim)
        py_queries = mx.random.normal(queries_shape, dtype=dtype)

    # 2. K/V Cache Pools
    num_total_physical_pages = 0
    for length in py_sequence_lengths_list:
        num_total_physical_pages += (length + tokens_per_page - 1) // tokens_per_page
    if num_total_physical_pages == 0:
        num_total_physical_pages = 1

    kv_cache_shape = (num_total_physical_pages, tokens_per_page, num_kv_heads, head_dim)
    py_k_cache_pool = mx.random.normal(kv_cache_shape, dtype=dtype)
    py_v_cache_pool = mx.random.normal(kv_cache_shape, dtype=dtype)

    # 3. Page Table: [NumSequencesInBatch, MaxLogicalBlocksPerSeq]
    py_page_table_list = []
    current_physical_page_idx = 0
    for i in range(num_sequences_in_batch):
        seq_len = py_sequence_lengths_list[i]
        num_logical_pages_for_seq = (seq_len + tokens_per_page - 1) // tokens_per_page

        pages_for_this_seq = []
        for _ in range(num_logical_pages_for_seq):
            pages_for_this_seq.append(current_physical_page_idx)
            current_physical_page_idx += 1
        while len(pages_for_this_seq) < max_logical_blocks_per_seq:
            pages_for_this_seq.append(0)
        py_page_table_list.append(pages_for_this_seq)

    if not py_page_table_list:
        py_page_table = mx.zeros((0, max_logical_blocks_per_seq), dtype=mx.uint32)
    else:
        py_page_table = mx.array(py_page_table_list, dtype=mx.uint32)

    # 4. Sequence Lengths: py_sequence_lengths is already defined.

    # 5. Query to Sequence Map: [TotalNumQueryTokensInBatch]
    py_query_to_seq_map_list = []
    for i in range(num_sequences_in_batch):
        py_query_to_seq_map_list.extend([i] * py_sequence_lengths_list[i])
    py_query_to_seq_map = mx.array(py_query_to_seq_map_list, dtype=mx.int32)

    # 6. Query Token Offset: [TotalNumQueryTokensInBatch]
    py_query_token_offset_list = []
    for i in range(num_sequences_in_batch):
        py_query_token_offset_list.extend(list(range(py_sequence_lengths_list[i])))
    py_query_token_offset = mx.array(py_query_token_offset_list, dtype=mx.int32)

    # Ensure all inputs are evaluated before calling the kernel
    mx.eval(
        py_queries,
        py_k_cache_pool,
        py_v_cache_pool,
        py_page_table,
        py_sequence_lengths,
        py_query_to_seq_map,
        py_query_token_offset,
    )

    # --- Call paged_attention the first time ---
    logger.info("  First call to paged_attention (prefill mode)...")
    output1 = paged_attention(
        py_queries,
        py_k_cache_pool,
        py_v_cache_pool,
        py_page_table,
        py_sequence_lengths,
        py_query_to_seq_map,
        py_query_token_offset,
        use_fused_kernel=False,
    )
    mx.eval(output1)

    logger.info("  Second call to paged_attention (prefill mode) with identical inputs...")
    output2 = paged_attention(
        py_queries,
        py_k_cache_pool,
        py_v_cache_pool,
        py_page_table,
        py_sequence_lengths,
        py_query_to_seq_map,
        py_query_token_offset,
        use_fused_kernel=False,
    )
    mx.eval(output2)

    # --- Assertions ---
    assert output1.shape == output2.shape, f"Output shapes differ: {output1.shape} vs {output2.shape}"
    assert output1.dtype == output2.dtype, f"Output dtypes differ: {output1.dtype} vs {output2.dtype}"

    # For debugging, print if they are not equal
    if not mx.array_equal(output1, output2).item():
        logger.error("Non-deterministic output detected!")
        logger.error(f"Output 1 sample: {output1[0, : min(output1.shape[1], 4)] if output1.size > 0 else 'empty'}")
        logger.error(f"Output 2 sample: {output2[0, : min(output2.shape[1], 4)] if output2.size > 0 else 'empty'}")
        if output1.size > 0 and output2.size > 0 and output1.size == output2.size:
            diff_mask: mx.array = mx.abs(output1 - output2) > 1e-5
            diff_loc = np.nonzero(diff_mask.tolist())
            if diff_loc[0].size > 0:
                first_diff_idx_flat = int(diff_loc[0][0])
                total_items_dim0 = total_query_tokens_in_batch * num_q_heads
                if total_items_dim0 > 0:
                    row_idx = first_diff_idx_flat // head_dim
                    col_idx = first_diff_idx_flat % head_dim
                    output1_val = output1[row_idx, col_idx].item()
                    output2_val = output2[row_idx, col_idx].item()
                    logger.error(f"First difference at index [{row_idx}, {col_idx}] (flat: {first_diff_idx_flat})")
                    logger.error(f"  Output1[{row_idx}, {col_idx}]: {output1_val}")
                    logger.error(f"  Output2[{row_idx}, {col_idx}]: {output2_val}")

    assert mx.array_equal(output1, output2).item(), (
        f"Paged attention prefill output is not deterministic for dtype={dtype}. Outputs differ between two identical calls."
    )

    logger.info("  Result: Outputs are identical - prefill determinism verified.")
