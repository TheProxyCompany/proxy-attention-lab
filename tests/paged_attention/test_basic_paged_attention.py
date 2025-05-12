import logging

import mlx.core as mx

from proxy_attention_lab import paged_attention

logger = logging.getLogger(__name__)


def test_fetch_k_vector_element_for_first_token_of_sequence():
    # --- Config ---
    num_q_threads = 2  # Test with 2 query threads, each belonging to a different sequence
    # These define the geometry the C++ primitive will tell the kernel via PagedAttentionParams
    # The kernel will use these to calculate strides and offsets.
    # C++ primitive will derive some of these from actual input shapes.
    cfg_tokens_per_page = 64
    cfg_num_kv_heads = 1
    cfg_head_dim = 4  # Let K vectors be of dimension 4 for this test
    # For a page_table [NumBatchSeq, MaxLogBlocks], C++ will set params.max_logical_blocks_per_seq
    # If page_table is [2, 2], then params.max_logical_blocks_per_seq = 2
    cfg_max_logical_blocks_per_seq_in_pagetable = 2

    # --- Inputs ---
    # 1. Queries: 1D [num_q_threads]. Each element is a dummy query value.
    py_queries = mx.array([100.0, 200.0], dtype=mx.float16)
    assert py_queries.shape[0] == num_q_threads

    # 2. K-Cache Pool: [NumPhysPages, TokensPerPage, NumKVHeads, HeadDim]
    # We need at least 2 physical pages if page_table maps to phys_page 0 and 1.
    num_physical_pages = 2
    k_cache_shape = (num_physical_pages, cfg_tokens_per_page, cfg_num_kv_heads, cfg_head_dim)
    py_k_cache_pool = mx.zeros(k_cache_shape, dtype=mx.float16)
    # We want to fetch K from: logical_block 0, token_slot 0, kv_head 0, element 0.
    # Seq 0, logical_block 0 -> maps to physical_page 0.
    # Seq 1, logical_block 0 -> maps to physical_page 1.
    py_k_cache_pool[0, 0, 0, 0] = 11.0  # K-value for Seq 0, Token 0, Head 0, Element 0
    py_k_cache_pool[1, 0, 0, 0] = 22.0  # K-value for Seq 1, Token 0, Head 0, Element 0

    # 3. V-Cache Pool (not used by kernel yet for output, but signature requires it)
    py_v_cache_pool = mx.zeros_like(py_k_cache_pool)

    # 4. Page Table: [NumBatchSequences, MaxLogicalBlocksPerSequenceInTable]
    #    py_page_table[seq_idx, logical_block_idx] = physical_page_id
    #    Thread 0 -> seq 0. Wants logical block 0 of seq 0. page_table[0,0] = phys_page 0.
    #    Thread 1 -> seq 1. Wants logical block 0 of seq 1. page_table[1,0] = phys_page 1.
    py_page_table = mx.array(
        [
            [0, 99],  # Seq 0: log_block 0 -> phys_page 0, log_block 1 -> phys_page 99 (dummy)
            [1, 88],
        ],  # Seq 1: log_block 0 -> phys_page 1, log_block 1 -> phys_page 88 (dummy)
        dtype=mx.uint32,
    )  # Shape (2, 2)
    assert py_page_table.shape == (num_q_threads, cfg_max_logical_blocks_per_seq_in_pagetable)

    # 5. sequence_lengths: [NumBatchSequences]
    py_sequence_lengths = mx.array([10, 5], dtype=mx.int32)  # Seq 0 has 10 tokens, Seq 1 has 5

    # 6. query_to_seq_map: [num_q_threads]. Maps each query thread to a sequence index.
    #    Thread 0 processes query for seq 0. Thread 1 processes query for seq 1.
    py_query_to_seq_map = mx.array([0, 1], dtype=mx.int32)

    # 7. query_token_offset: [num_q_threads]. Logical position of Q within its sequence.
    #    Not directly used to fetch K for *this specific test's target K (token 0)*,
    #    but needed by signature. Let's say thread 0 is for Q at pos 5 of seq 0,
    #    and thread 1 is for Q at pos 2 of seq 1.
    py_query_token_offset = mx.array([5, 2], dtype=mx.int32)

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

    # --- Expected Output ---
    # Kernel Output = queries_in[tid] + k_element_val_for_seq_token_0_head_0_element_0
    # Thread 0 (for seq 0): query=100.0. Fetches K from phys_page 0, token 0, head 0, el 0 => 11.0. Output = 100.0 + 11.0 = 111.0
    # Thread 1 (for seq 1): query=200.0. Fetches K from phys_page 1, token 0, head 0, el 0 => 22.0. Output = 200.0 + 22.0 = 222.0
    expected_output = mx.array([111.0, 222.0], dtype=mx.float16)

    logger.info(f"Test: Expected output: {expected_output}")
    logger.info(f"Test: Actual output  : {output_arr}")
    assert output_arr.shape == expected_output.shape
    assert output_arr.dtype == expected_output.dtype
    assert mx.allclose(output_arr, expected_output, atol=1e-3)  # Looser tolerance for float math
