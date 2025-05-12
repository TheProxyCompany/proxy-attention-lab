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
    # We want to fetch K from: logical_block 0, token_slot 0, kv_head 0.
    # Seq 0, logical_block 0 -> maps to physical_page 0.
    # Seq 1, logical_block 0 -> maps to physical_page 1.
    #
    # Since our general kernel now sums all elements in the K-vector,
    # we need to set up the values so that the sum of the vector is 11.0 or 22.0
    # to match the expected output.

    # For Seq 0: Put 11.0 in first element, 0.0 in others
    py_k_cache_pool[0, 0, 0, 0] = 11.0  # K-value for Seq 0, Token 0, Head 0, Element 0
    for i in range(1, cfg_head_dim):
        py_k_cache_pool[0, 0, 0, i] = 0.0

    # For Seq 1: Put 22.0 in first element, 0.0 in others
    py_k_cache_pool[1, 0, 0, 0] = 22.0  # K-value for Seq 1, Token 0, Head 0, Element 0
    for i in range(1, cfg_head_dim):
        py_k_cache_pool[1, 0, 0, i] = 0.0

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
    # For this test, we need to set these to 0 to target token_slot 0
    py_query_token_offset = mx.array([0, 0], dtype=mx.int32)

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


def test_fetch_entire_k_vector_for_specific_token_slot():
    # --- Config ---
    num_q_threads = 2  # Test with 2 query threads, each belonging to a different sequence
    cfg_tokens_per_page = 64
    cfg_num_kv_heads = 1
    cfg_head_dim = 4  # Let K vectors be of dimension 4 for this test
    cfg_max_logical_blocks_per_seq_in_pagetable = 2

    # --- Inputs ---
    # 1. Queries: 1D [num_q_threads]. Each element is a dummy query value.
    py_queries = mx.array([100.0, 200.0], dtype=mx.float16)
    assert py_queries.shape[0] == num_q_threads

    # 2. K-Cache Pool: [NumPhysPages, TokensPerPage, NumKVHeads, HeadDim]
    num_physical_pages = 2
    k_cache_shape = (num_physical_pages, cfg_tokens_per_page, cfg_num_kv_heads, cfg_head_dim)
    py_k_cache_pool = mx.zeros(k_cache_shape, dtype=mx.float16)

    # Populate entire K vectors for token_slot 0, kv_head 0 in both physical pages
    # For physical_page 0 (sequence 0), fill with values 1.0, 2.0, 3.0, 4.0
    # For physical_page 1 (sequence 1), fill with values 5.0, 6.0, 7.0, 8.0
    for i in range(cfg_head_dim):
        py_k_cache_pool[0, 0, 0, i] = float(i + 1)  # Values 1.0, 2.0, 3.0, 4.0
        py_k_cache_pool[1, 0, 0, i] = float(i + 5)  # Values 5.0, 6.0, 7.0, 8.0

    # 3. V-Cache Pool (not used by kernel yet for output, but signature requires it)
    py_v_cache_pool = mx.zeros_like(py_k_cache_pool)

    # 4. Page Table: [NumBatchSequences, MaxLogicalBlocksPerSequenceInTable]
    py_page_table = mx.array(
        [
            [0, 99],  # Seq 0: log_block 0 -> phys_page 0, log_block 1 -> phys_page 99 (dummy)
            [1, 88],  # Seq 1: log_block 0 -> phys_page 1, log_block 1 -> phys_page 88 (dummy)
        ],
        dtype=mx.uint32,
    )
    assert py_page_table.shape == (num_q_threads, cfg_max_logical_blocks_per_seq_in_pagetable)

    # 5. sequence_lengths: [NumBatchSequences]
    py_sequence_lengths = mx.array([10, 5], dtype=mx.int32)  # Seq 0 has 10 tokens, Seq 1 has 5

    # 6. query_to_seq_map: [num_q_threads]. Maps each query thread to a sequence index.
    py_query_to_seq_map = mx.array([0, 1], dtype=mx.int32)

    # 7. query_token_offset: [num_q_threads]. Logical position of Q within its sequence.
    # For this test, we need to set these to 0 to target token_slot 0
    py_query_token_offset = mx.array([0, 0], dtype=mx.int32)

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
    # Expected behavior: The kernel adds the query value to the sum of all elements in the K-vector
    # Thread 0 (seq 0): query=100.0, K-vector sum = 1.0+2.0+3.0+4.0 = 10.0, Output = 100.0 + 10.0 = 110.0
    # Thread 1 (seq 1): query=200.0, K-vector sum = 5.0+6.0+7.0+8.0 = 26.0, Output = 200.0 + 26.0 = 226.0
    expected_output = mx.array([110.0, 226.0], dtype=mx.float16)

    logger.info(f"Test: Expected output: {expected_output}")
    logger.info(f"Test: Actual output  : {output_arr}")
    assert output_arr.shape == expected_output.shape
    assert output_arr.dtype == expected_output.dtype
    assert mx.allclose(output_arr, expected_output, atol=1e-3)  # Looser tolerance for float math


def test_fetch_k_vector_from_variable_token_slot_in_first_logical_block():
    # --- Config ---
    num_q_threads = 2  # Test with 2 query threads, each accessing different token slots
    cfg_tokens_per_page = 64
    cfg_num_kv_heads = 1
    cfg_head_dim = 4  # Let K vectors be of dimension 4 for this test
    cfg_max_logical_blocks_per_seq_in_pagetable = 2

    # --- Inputs ---
    # 1. Queries: 1D [num_q_threads]. Each element is a dummy query value.
    py_queries = mx.array([100.0, 200.0], dtype=mx.float16)
    assert py_queries.shape[0] == num_q_threads

    # 2. K-Cache Pool: [NumPhysPages, TokensPerPage, NumKVHeads, HeadDim]
    num_physical_pages = 2
    k_cache_shape = (num_physical_pages, cfg_tokens_per_page, cfg_num_kv_heads, cfg_head_dim)
    py_k_cache_pool = mx.zeros(k_cache_shape, dtype=mx.float16)

    # Populate K vectors for different token slots in physical page 0 (for sequence 0)
    # Token slot 3: values 1.0, 2.0, 3.0, 4.0 (sum = 10.0)
    # Token slot 7: values 5.0, 6.0, 7.0, 8.0 (sum = 26.0)
    for i in range(cfg_head_dim):
        py_k_cache_pool[0, 3, 0, i] = float(i + 1)  # Values 1.0, 2.0, 3.0, 4.0
        py_k_cache_pool[0, 7, 0, i] = float(i + 5)  # Values 5.0, 6.0, 7.0, 8.0

    # 3. V-Cache Pool (not used by kernel yet for output, but signature requires it)
    py_v_cache_pool = mx.zeros_like(py_k_cache_pool)

    # 4. Page Table: [NumBatchSequences, MaxLogicalBlocksPerSequenceInTable]
    # Both threads will target sequence 0, which has logical block 0 mapped to physical page 0
    py_page_table = mx.array(
        [
            [0, 99],  # Seq 0: log_block 0 -> phys_page 0, log_block 1 -> phys_page 99 (dummy)
            [1, 88],  # Seq 1: log_block 0 -> phys_page 1, log_block 1 -> phys_page 88 (dummy) (not used in this test)
        ],
        dtype=mx.uint32,
    )
    assert py_page_table.shape == (2, cfg_max_logical_blocks_per_seq_in_pagetable)  # Keep 2 sequences in page table

    # 5. sequence_lengths: [NumBatchSequences]
    py_sequence_lengths = mx.array([64, 32], dtype=mx.int32)  # Seq 0 has 64 tokens, Seq 1 has 32 (not used)

    # 6. query_to_seq_map: [num_q_threads]. Map both threads to sequence 0.
    py_query_to_seq_map = mx.array([0, 0], dtype=mx.int32)

    # 7. query_token_offset: [num_q_threads]. Key parameter for this test.
    # Thread 0 will target token position 3 in sequence 0 (within logical block 0)
    # Thread 1 will target token position 7 in sequence 0 (within logical block 0)
    # Both must be < cfg_tokens_per_page to ensure we stay within logical block 0
    py_query_token_offset = mx.array([3, 7], dtype=mx.int32)

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
    # Expected behavior: The kernel adds the query value to the sum of all elements in the K-vector
    # for the token slot specified by query_token_offset
    # Thread 0: targets seq 0, token pos 3 => query=100.0, K-vector sum = 1.0+2.0+3.0+4.0 = 10.0, Output = 100.0 + 10.0 = 110.0
    # Thread 1: targets seq 0, token pos 7 => query=200.0, K-vector sum = 5.0+6.0+7.0+8.0 = 26.0, Output = 200.0 + 26.0 = 226.0
    expected_output = mx.array([110.0, 226.0], dtype=mx.float16)

    logger.info(f"Test: Expected output: {expected_output}")
    logger.info(f"Test: Actual output  : {output_arr}")
    assert output_arr.shape == expected_output.shape
    assert output_arr.dtype == expected_output.dtype
    assert mx.allclose(output_arr, expected_output, atol=1e-3)  # Looser tolerance for float math


def test_fetch_k_vector_from_multiple_kv_heads():
    # --- Config ---
    num_tokens = 1  # Test with 1 token position
    num_q_heads = 2  # Test with 2 query heads, one per KV head (GQA factor = 1)
    cfg_tokens_per_page = 64
    cfg_num_kv_heads = 2  # Important: We now have 2 KV heads
    cfg_head_dim = 4  # Let K vectors be of dimension 4 for this test
    cfg_max_logical_blocks_per_seq_in_pagetable = 2

    # --- Inputs ---
    # 1. Queries: 3D [num_tokens, num_q_heads, head_dim]
    # First dimension: token index
    # Second dimension: query head index
    # Third dimension: head dimension
    # Shape: (1, 2, 4) - 1 token with 2 query heads, each with 4-dimensional embedding
    # For q_head 0, set all elements to 100.0
    # For q_head 1, set all elements to 200.0
    py_queries = mx.zeros((num_tokens, num_q_heads, cfg_head_dim), dtype=mx.float16)
    # Fill each query head with a distinct value for clear testing
    py_queries[0, 0, :] = 100.0  # First query head all 100.0
    py_queries[0, 1, :] = 200.0  # Second query head all 200.0

    # 2. K-Cache Pool: [NumPhysPages, TokensPerPage, NumKVHeads, HeadDim]
    num_physical_pages = 1  # We only need one physical page for this test
    k_cache_shape = (num_physical_pages, cfg_tokens_per_page, cfg_num_kv_heads, cfg_head_dim)
    py_k_cache_pool = mx.zeros(k_cache_shape, dtype=mx.float16)

    # Populate K vectors for token_slot 5, for both KV heads in physical page 0
    # KV head 0: values 1.0, 2.0, 3.0, 4.0 (sum = 10.0)
    # KV head 1: values 5.0, 6.0, 7.0, 8.0 (sum = 26.0)
    token_slot = 5
    for i in range(cfg_head_dim):
        py_k_cache_pool[0, token_slot, 0, i] = float(i + 1)  # Values 1.0, 2.0, 3.0, 4.0 for head 0
        py_k_cache_pool[0, token_slot, 1, i] = float(i + 5)  # Values 5.0, 6.0, 7.0, 8.0 for head 1

    # 3. V-Cache Pool (not used by kernel yet for output, but signature requires it)
    py_v_cache_pool = mx.zeros_like(py_k_cache_pool)

    # 4. Page Table: [NumBatchSequences, MaxLogicalBlocksPerSequenceInTable]
    # Both threads will target sequence 0, which has logical block 0 mapped to physical page 0
    py_page_table = mx.array(
        [
            [0, 99],  # Seq 0: log_block 0 -> phys_page 0, log_block 1 -> phys_page 99 (dummy)
        ],
        dtype=mx.uint32,
    )
    assert py_page_table.shape == (1, cfg_max_logical_blocks_per_seq_in_pagetable)  # Only 1 sequence in this test

    # 5. sequence_lengths: [NumBatchSequences]
    py_sequence_lengths = mx.array([64], dtype=mx.int32)  # Seq 0 has 64 tokens

    # 6. query_to_seq_map: [num_tokens * num_q_heads]. Map all to sequence 0.
    # For flattened dispatch: (q_token_idx * num_q_heads + q_head_idx) -> seq_idx
    # Since we have 1 token with 2 heads, we need 2 entries, both mapping to seq 0
    py_query_to_seq_map = mx.array([0, 0], dtype=mx.int32)

    # 7. query_token_offset: [num_tokens * num_q_heads]. All heads for the same token target same position
    # Since we have 1 token with 2 heads, we need 2 entries, both with the same token_slot
    py_query_token_offset = mx.array([token_slot, token_slot], dtype=mx.int32)

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
    # Expected behavior: The kernel adds the query value to the sum of all elements in the K-vector
    # Each query head should target the corresponding KV head based on the new mapping logic
    # The output shape should be (num_tokens, num_q_heads, head_dim) = (1, 2, 4)
    #
    # For token 0, q_head 0: targets kv_head 0
    #   query=100.0 values from head, K-vector sum = 1.0+2.0+3.0+4.0 = 10.0,
    #   Output should have 100.0 + 10.0 = 110.0 in all elements
    #
    # For token 0, q_head 1: targets kv_head 1
    #   query=200.0 values from head, K-vector sum = 5.0+6.0+7.0+8.0 = 26.0,
    #   Output should have 200.0 + 26.0 = 226.0 in all elements

    # The C++ primitive now preserves the 3D shape of the query for the output
    # Expected shape: (num_tokens, num_q_heads, head_dim) = (1, 2, 4)
    # Create a 3D expected output array with appropriate values
    expected_output = mx.zeros((num_tokens, num_q_heads, cfg_head_dim), dtype=mx.float16)

    # For token 0, q_head 0: fill with 110.0 (query 100.0 + K-vector sum 10.0)
    expected_output[0, 0, :] = 110.0

    # For token 0, q_head 1: fill with 226.0 (query 200.0 + K-vector sum 26.0)
    expected_output[0, 1, :] = 226.0

    logger.info(f"Test: Expected output: {expected_output}")
    logger.info(f"Test: Actual output  : {output_arr}")
    assert output_arr.shape == expected_output.shape
    assert output_arr.dtype == expected_output.dtype
    assert mx.allclose(output_arr, expected_output, atol=1e-3)  # Looser tolerance for float math


def test_invalid_physical_page_id_in_page_table():
    """Test that when page_table contains a physical_page_id >= num_physical_pages, the kernel
    safely handles it by returning the original query value for the affected thread."""
    # --- Config ---
    cfg_tokens_per_page = 64
    cfg_num_kv_heads = 1
    cfg_head_dim = 4

    # --- Inputs ---
    py_queries = mx.array([100.0, 200.0], dtype=mx.float16)

    # Setup K-Cache Pool with 2 physical pages
    num_physical_pages = 2
    k_cache_shape = (num_physical_pages, cfg_tokens_per_page, cfg_num_kv_heads, cfg_head_dim)
    py_k_cache_pool = mx.zeros(k_cache_shape, dtype=mx.float16)

    # V-Cache Pool
    py_v_cache_pool = mx.zeros_like(py_k_cache_pool)

    # INVALID page_table: physical_page_id = 2 is out of bounds (>= num_physical_pages)
    py_page_table = mx.array(
        [
            [0, 99],  # Valid: phys_page 0 is within bounds
            [2, 88],  # Invalid: phys_page 2 is out of bounds (>= num_physical_pages=2)
        ],
        dtype=mx.uint32,
    )

    py_sequence_lengths = mx.array([10, 5], dtype=mx.int32)
    py_query_to_seq_map = mx.array([0, 1], dtype=mx.int32)
    py_query_token_offset = mx.array([0, 0], dtype=mx.int32)

    # Call paged_attention and expect safe handling of invalid input
    output_arr = paged_attention(
        py_queries,
        py_k_cache_pool,
        py_v_cache_pool,
        py_page_table,
        py_sequence_lengths,
        py_query_to_seq_map,
        py_query_token_offset,
    )
    mx.eval(output_arr)  # Force eager execution

    # --- Expected Output ---
    # Thread 0 (valid input): should process normally, query=100.0, K-vector sum depends on what's in py_k_cache_pool[0, 0, 0, :]
    # Thread 1 (invalid physical_page_id=2): should return zeros due to invalid input
    expected_thread0_output = 100.0 + 0.0  # 0.0 because py_k_cache_pool is filled with zeros
    expected_thread1_output = 0.0  # Now returning 0.0 (not the original query) on error
    expected_output = mx.array([expected_thread0_output, expected_thread1_output], dtype=mx.float16)

    logger.info(f"Test: Expected output: {expected_output}")
    logger.info(f"Test: Actual output  : {output_arr}")
    assert output_arr.shape == expected_output.shape
    assert output_arr.dtype == expected_output.dtype
    # Check that thread 1's output equals zero (indicating error handling)
    assert mx.allclose(output_arr[1], mx.array([expected_thread1_output], dtype=mx.float16), atol=1e-3)


def test_negative_query_token_offset():
    """Test that when query_token_offset contains negative values, the kernel
    safely handles it by returning the original query value for the affected thread."""
    # --- Config ---
    cfg_tokens_per_page = 64
    cfg_num_kv_heads = 1
    cfg_head_dim = 4

    # --- Inputs ---
    py_queries = mx.array([100.0, 200.0], dtype=mx.float16)

    # Setup K-Cache Pool
    num_physical_pages = 2
    k_cache_shape = (num_physical_pages, cfg_tokens_per_page, cfg_num_kv_heads, cfg_head_dim)
    py_k_cache_pool = mx.zeros(k_cache_shape, dtype=mx.float16)

    # V-Cache Pool
    py_v_cache_pool = mx.zeros_like(py_k_cache_pool)

    # Valid page_table
    py_page_table = mx.array(
        [
            [0, 99],
            [1, 88],
        ],
        dtype=mx.uint32,
    )

    py_sequence_lengths = mx.array([10, 5], dtype=mx.int32)
    py_query_to_seq_map = mx.array([0, 1], dtype=mx.int32)

    # INVALID query_token_offset: contains a negative value
    py_query_token_offset = mx.array([-1, 0], dtype=mx.int32)

    # Call paged_attention and expect safe handling of invalid input
    output_arr = paged_attention(
        py_queries,
        py_k_cache_pool,
        py_v_cache_pool,
        py_page_table,
        py_sequence_lengths,
        py_query_to_seq_map,
        py_query_token_offset,
    )
    mx.eval(output_arr)  # Force eager execution

    # --- Expected Output ---
    # Thread 0 (negative token offset): should return zeros due to invalid input
    # Thread 1 (valid input): should process normally, query=200.0, K-vector sum depends on what's in py_k_cache_pool[1, 0, 0, :]
    expected_thread0_output = 0.0  # Return 0.0 (not original query) on error
    expected_thread1_output = 200.0 + 0.0  # 0.0 because py_k_cache_pool is filled with zeros
    expected_output = mx.array([expected_thread0_output, expected_thread1_output], dtype=mx.float16)

    logger.info(f"Test: Expected output: {expected_output}")
    logger.info(f"Test: Actual output  : {output_arr}")
    assert output_arr.shape == expected_output.shape
    assert output_arr.dtype == expected_output.dtype
    # Check that thread 0's output equals zero (indicating error handling)
    assert mx.allclose(output_arr[0], mx.array([expected_thread0_output], dtype=mx.float16), atol=1e-3)


def test_invalid_seq_idx_in_query_map():
    """Test that when query_to_seq_map contains a seq_idx >= num_sequences, the kernel
    safely handles it by returning the original query value for the affected thread."""
    # --- Config ---
    cfg_tokens_per_page = 64
    cfg_num_kv_heads = 1
    cfg_head_dim = 4

    # --- Inputs ---
    py_queries = mx.array([100.0, 200.0], dtype=mx.float16)

    # Setup K-Cache Pool
    num_physical_pages = 2
    k_cache_shape = (num_physical_pages, cfg_tokens_per_page, cfg_num_kv_heads, cfg_head_dim)
    py_k_cache_pool = mx.zeros(k_cache_shape, dtype=mx.float16)

    # V-Cache Pool
    py_v_cache_pool = mx.zeros_like(py_k_cache_pool)

    # Valid page_table with 2 sequences
    py_page_table = mx.array(
        [
            [0, 99],  # Seq 0
            [1, 88],  # Seq 1
        ],
        dtype=mx.uint32,
    )

    py_sequence_lengths = mx.array([10, 5], dtype=mx.int32)

    # INVALID query_to_seq_map: contains seq_idx=2, which is >= num_sequences=2
    py_query_to_seq_map = mx.array([0, 2], dtype=mx.int32)

    py_query_token_offset = mx.array([0, 0], dtype=mx.int32)

    # Call paged_attention and expect safe handling of invalid input
    output_arr = paged_attention(
        py_queries,
        py_k_cache_pool,
        py_v_cache_pool,
        py_page_table,
        py_sequence_lengths,
        py_query_to_seq_map,
        py_query_token_offset,
    )
    mx.eval(output_arr)  # Force eager execution

    # --- Expected Output ---
    # Thread 0 (valid input): should process normally, query=100.0, K-vector sum depends on what's in py_k_cache_pool[0, 0, 0, :]
    # Thread 1 (invalid seq_idx=2): should return zeros due to invalid input
    expected_thread0_output = 100.0 + 0.0  # 0.0 because py_k_cache_pool is filled with zeros
    expected_thread1_output = 0.0  # Return 0.0 (not original query) on error
    expected_output = mx.array([expected_thread0_output, expected_thread1_output], dtype=mx.float16)

    logger.info(f"Test: Expected output: {expected_output}")
    logger.info(f"Test: Actual output  : {output_arr}")
    assert output_arr.shape == expected_output.shape
    assert output_arr.dtype == expected_output.dtype
    # Check that thread 1's output equals zero (indicating error handling)
    assert mx.allclose(output_arr[1], mx.array([expected_thread1_output], dtype=mx.float16), atol=1e-3)


def test_invalid_gqa_configuration():
    """Test that when num_q_heads is not a multiple of num_kv_heads, the C++ primitive throws an exception."""
    # --- Config ---
    num_tokens = 1  # Test with 1 token position
    num_q_heads = 3  # Not a multiple of num_kv_heads=2
    cfg_tokens_per_page = 64
    cfg_num_kv_heads = 2  # Important: num_q_heads (3) is not a multiple of this value (2)
    cfg_head_dim = 4  # Let K vectors be of dimension 4 for this test

    # --- Inputs ---
    # 1. Queries: 3D [num_tokens, num_q_heads, head_dim]
    py_queries = mx.zeros((num_tokens, num_q_heads, cfg_head_dim), dtype=mx.float16)
    # Fill each query head with a distinct value for clear testing
    py_queries[0, 0, :] = 100.0  # First query head all 100.0
    py_queries[0, 1, :] = 200.0  # Second query head all 200.0
    py_queries[0, 2, :] = 300.0  # Third query head all 300.0

    # 2. K-Cache Pool: [NumPhysPages, TokensPerPage, NumKVHeads, HeadDim]
    num_physical_pages = 1  # We only need one physical page for this test
    k_cache_shape = (num_physical_pages, cfg_tokens_per_page, cfg_num_kv_heads, cfg_head_dim)
    py_k_cache_pool = mx.zeros(k_cache_shape, dtype=mx.float16)

    # 3. V-Cache Pool (not used by kernel yet for output, but signature requires it)
    py_v_cache_pool = mx.zeros_like(py_k_cache_pool)

    # 4. Page Table: [NumBatchSequences, MaxLogicalBlocksPerSequenceInTable]
    py_page_table = mx.array(
        [
            [0, 99],  # Seq 0: log_block 0 -> phys_page 0, log_block 1 -> phys_page 99 (dummy)
        ],
        dtype=mx.uint32,
    )

    # 5. sequence_lengths: [NumBatchSequences]
    py_sequence_lengths = mx.array([64], dtype=mx.int32)  # Seq 0 has 64 tokens

    # 6. query_to_seq_map: [num_tokens * num_q_heads]. Map all to sequence 0.
    # For flattened dispatch: (q_token_idx * num_q_heads + q_head_idx) -> seq_idx
    # Since we have 1 token with 3 heads, we need 3 entries, all mapping to seq 0
    py_query_to_seq_map = mx.array([0, 0, 0], dtype=mx.int32)

    # 7. query_token_offset: [num_tokens * num_q_heads]. All heads for the same token target same position
    py_query_token_offset = mx.array([0, 0, 0], dtype=mx.int32)

    # --- Call Op and Expect Exception ---
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
        mx.eval(output_arr)  # Force eager execution
