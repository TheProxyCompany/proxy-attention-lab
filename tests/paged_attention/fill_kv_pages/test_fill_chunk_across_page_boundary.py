import logging

import mlx.core as mx
import pytest

from proxy_attention_lab import fill_kv_pages

logger = logging.getLogger(__name__)


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
def test_fill_chunk_across_page_boundary(dtype):
    """Test filling a chunk that spans across a page boundary for both float16 and bfloat16.

    This test verifies that the fill_kv_pages operation correctly writes
    new key and value data when a chunk of tokens starts in one logical block
    (physical page) and ends in the next.
    """
    # Fixed test parameters
    num_kv_heads = 1
    head_dim = 4
    primitive_tokens_per_page = 4

    # Scenario: Write tokens starting mid-page and crossing to next page
    start_logical_position = primitive_tokens_per_page // 2  # Start at position 2 (mid-page)
    num_new_tokens_in_chunk = primitive_tokens_per_page  # 4 tokens will cross boundary

    num_sequences_in_batch = 1
    num_physical_pages = 3  # Need at least 2, using 3 for safety

    logger.info(
        f"Test parameters: num_new_tokens_in_chunk={num_new_tokens_in_chunk}, "
        f"start_logical_position={start_logical_position}, "
        f"num_sequences_in_batch={num_sequences_in_batch}, "
        f"num_kv_heads={num_kv_heads}, head_dim={head_dim}, "
        f"tokens_per_page={primitive_tokens_per_page}, "
        f"num_physical_pages={num_physical_pages}, dtype={dtype}"
    )

    # Create distinct new keys and values
    # Shape: [num_new_tokens_in_chunk, num_kv_heads, head_dim]
    new_keys_data = []
    new_values_data = []
    for i in range(num_new_tokens_in_chunk):
        # Keys: [i*10+1, i*10+2, i*10+3, i*10+4]
        # Values: [i*10+5, i*10+6, i*10+7, i*10+8]
        key_row = [i * 10 + j + 1 for j in range(head_dim)]
        value_row = [i * 10 + j + 5 for j in range(head_dim)]
        new_keys_data.append([key_row])  # Extra dimension for kv_heads
        new_values_data.append([value_row])

    new_keys = mx.array(new_keys_data, dtype=dtype)
    new_values = mx.array(new_values_data, dtype=dtype)

    logger.debug(f"new_keys shape: {new_keys.shape}")
    logger.debug(f"new_values shape: {new_values.shape}")
    for i in range(num_new_tokens_in_chunk):
        logger.debug(f"Token {i} key: {new_keys[i].tolist()}, value: {new_values[i].tolist()}")

    # Initialize global pools with zeros
    global_key_pool = mx.zeros((num_physical_pages, primitive_tokens_per_page, num_kv_heads, head_dim), dtype=dtype)
    global_value_pool = mx.zeros((num_physical_pages, primitive_tokens_per_page, num_kv_heads, head_dim), dtype=dtype)

    logger.debug(f"global_key_pool shape: {global_key_pool.shape}")
    logger.debug(f"global_value_pool shape: {global_value_pool.shape}")

    # Page table: map logical block 0 to physical page 0, logical block 1 to physical page 1
    # Shape: [1, 2] (one sequence, two logical blocks)
    page_table = mx.array([[0, 1]], dtype=mx.uint32)
    logger.debug(f"page_table shape: {page_table.shape}, values: {page_table.tolist()}")

    # Write positions: consecutive positions starting from start_logical_position
    # With start_logical_position=2 and 4 tokens: positions will be [2, 3, 4, 5]
    # This means: slot 2,3 in page 0, then slot 0,1 in page 1
    current_token_write_positions = mx.array(
        [start_logical_position + i for i in range(num_new_tokens_in_chunk)], dtype=mx.int32
    )
    logger.debug(
        f"current_token_write_positions shape: {current_token_write_positions.shape}, "
        f"values: {current_token_write_positions.tolist()}"
    )

    # Log the expected mapping
    for i, pos in enumerate(current_token_write_positions.tolist()):
        logical_block = pos // primitive_tokens_per_page
        slot_in_block = pos % primitive_tokens_per_page
        physical_page = page_table[0, logical_block].item()
        logger.debug(
            f"Token {i} at logical pos {pos} -> logical block {logical_block}, "
            f"slot {slot_in_block}, physical page {physical_page}"
        )

    # Query to sequence mapping: all tokens belong to sequence 0
    query_to_seq_map = mx.array([0] * num_new_tokens_in_chunk, dtype=mx.uint32)
    logger.debug(f"query_to_seq_map shape: {query_to_seq_map.shape}, values: {query_to_seq_map.tolist()}")

    # Evaluate all inputs before the call
    mx.eval(
        new_keys,
        new_values,
        global_key_pool,
        global_value_pool,
        page_table,
        current_token_write_positions,
        query_to_seq_map,
    )

    logger.info("Calling fill_kv_pages...")

    # Call fill_kv_pages
    updated_k_pool, updated_v_pool = fill_kv_pages(
        new_keys=new_keys,
        new_values=new_values,
        global_key_pool=global_key_pool,
        global_value_pool=global_value_pool,
        page_table=page_table,
        current_token_write_positions=current_token_write_positions,
        query_to_seq_map=query_to_seq_map,
    )

    # Force kernel execution
    mx.eval(updated_k_pool, updated_v_pool)

    logger.info("fill_kv_pages execution completed")

    # Build expected pools
    expected_k_pool = mx.zeros((num_physical_pages, primitive_tokens_per_page, num_kv_heads, head_dim), dtype=dtype)
    expected_v_pool = mx.zeros((num_physical_pages, primitive_tokens_per_page, num_kv_heads, head_dim), dtype=dtype)

    # Fill expected data based on where each token should be written
    # With start_logical_position=2 and 4 tokens:
    # Token 0 -> logical pos 2 -> page 0, slot 2
    # Token 1 -> logical pos 3 -> page 0, slot 3
    # Token 2 -> logical pos 4 -> page 1, slot 0
    # Token 3 -> logical pos 5 -> page 1, slot 1
    for i in range(num_new_tokens_in_chunk):
        logical_pos = start_logical_position + i
        logical_block = logical_pos // primitive_tokens_per_page
        slot_in_block = logical_pos % primitive_tokens_per_page
        physical_page = [0, 1][logical_block]  # Direct mapping from page table

        # Expected keys and values
        expected_key = [i * 10 + j + 1 for j in range(head_dim)]
        expected_value = [i * 10 + j + 5 for j in range(head_dim)]

        expected_k_pool[physical_page, slot_in_block, 0, :] = mx.array(expected_key, dtype=dtype)
        expected_v_pool[physical_page, slot_in_block, 0, :] = mx.array(expected_value, dtype=dtype)

        logger.debug(
            f"Token {i}: logical_pos={logical_pos}, physical_page={physical_page}, "
            f"slot={slot_in_block}, key={expected_key}"
        )

    # Data assertions
    logger.debug("Comparing key pools...")
    if not mx.allclose(updated_k_pool, expected_k_pool, rtol=1e-3, atol=1e-3):
        # Log differences for debugging
        for page in range(num_physical_pages):
            logger.debug(f"Page {page} actual keys:")
            for slot in range(primitive_tokens_per_page):
                actual = updated_k_pool[page, slot, 0, :].tolist()
                expected = expected_k_pool[page, slot, 0, :].tolist()
                logger.debug(f"  Slot {slot}: actual={actual}, expected={expected}")
                if actual != expected and any(v != 0 for v in expected):
                    logger.error(f"  MISMATCH at page={page}, slot={slot}")

        pytest.fail(
            "Data in updated_k_pool does not match expected_k_pool. "
            "The chunk spanning page boundary was not written correctly."
        )

    logger.debug("Comparing value pools...")
    if not mx.allclose(updated_v_pool, expected_v_pool, rtol=1e-3, atol=1e-3):
        # Log differences for debugging
        for page in range(num_physical_pages):
            logger.debug(f"Page {page} actual values:")
            for slot in range(primitive_tokens_per_page):
                actual = updated_v_pool[page, slot, 0, :].tolist()
                expected = expected_v_pool[page, slot, 0, :].tolist()
                logger.debug(f"  Slot {slot}: actual={actual}, expected={expected}")
                if actual != expected and any(v != 0 for v in expected):
                    logger.error(f"  MISMATCH at page={page}, slot={slot}")

        pytest.fail(
            "Data in updated_v_pool does not match expected_v_pool. "
            "The chunk spanning page boundary was not written correctly."
        )

    logger.info(
        f"Test passed: Chunk of {num_new_tokens_in_chunk} tokens correctly written "
        f"across page boundary (starting at position {start_logical_position}), dtype={dtype}"
    )
