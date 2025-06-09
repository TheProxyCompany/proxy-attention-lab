import logging

import mlx.core as mx
import pytest

from proxy_attention_lab import fill_kv_pages

logger = logging.getLogger(__name__)


@pytest.mark.parametrize(
    "num_new_tokens_in_chunk,start_logical_position",
    [
        (3, 0),  # Small chunk starting at beginning
        (3, 4),  # Small chunk starting at second block (with tokens_per_page=4)
        (3, 0),  # Tokens less than a page
        (4, 0),  # Exactly one page
        (5, 0),  # More than one page
        (3, 4),  # Starting in second logical block
        (17, 0),  # Total new tokens = 17, max logical blocks = 5
    ],
)
@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
@pytest.mark.parametrize("num_kv_heads", [1, 8])
@pytest.mark.parametrize("head_dim", [4, 128])
def test_fill_multiple_tokens_single_sequence(
    num_kv_heads, head_dim, num_new_tokens_in_chunk, start_logical_position, dtype
):
    """Test filling multiple tokens for a single sequence (simulating prefill chunk).

    This test verifies that the fill_kv_pages operation correctly writes
    multiple new key and value tokens for the same sequence, simulating
    a prefill operation for a chunk of tokens.

    Args:
        num_new_tokens_in_chunk: Number of new tokens to write
        start_logical_position: Starting logical position for writing
        dtype: Data type to use for the test (mx.float16 or mx.bfloat16)
    """
    # Fixed test parameters
    num_sequences_in_batch = 1
    primitive_tokens_per_page = 4

    # Calculate required logical blocks and physical pages
    end_logical_position = start_logical_position + num_new_tokens_in_chunk - 1
    max_logical_block = end_logical_position // primitive_tokens_per_page
    num_logical_blocks_needed = max_logical_block + 1
    num_physical_pages = num_logical_blocks_needed + 1  # Extra page for safety

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
    # Use distinct values for easy verification
    new_keys_data = []
    new_values_data = []
    for i in range(num_new_tokens_in_chunk):
        # Create data for all KV heads
        token_keys = []
        token_values = []
        for _ in range(num_kv_heads):
            # Keys: [i*10+1, i*10+2, i*10+3, i*10+4] for each head
            # Values: [i*10+5, i*10+6, i*10+7, i*10+8] for each head
            key_head = [i * 10 + j + 1 for j in range(head_dim)]
            value_head = [i * 10 + j + 5 for j in range(head_dim)]
            token_keys.append(key_head)
            token_values.append(value_head)
        new_keys_data.append(token_keys)
        new_values_data.append(token_values)

    new_keys = mx.array(new_keys_data, dtype=dtype)
    new_values = mx.array(new_values_data, dtype=dtype)

    logger.debug(f"new_keys shape: {new_keys.shape}, first row: {new_keys[0].tolist()}")
    logger.debug(f"new_values shape: {new_values.shape}, first row: {new_values[0].tolist()}")

    # Initialize global pools with zeros
    # Shape: [num_physical_pages, tokens_per_page, num_kv_heads, head_dim]
    global_key_pool = mx.zeros((num_physical_pages, primitive_tokens_per_page, num_kv_heads, head_dim), dtype=dtype)
    global_value_pool = mx.zeros((num_physical_pages, primitive_tokens_per_page, num_kv_heads, head_dim), dtype=dtype)

    logger.debug(f"global_key_pool shape: {global_key_pool.shape}")
    logger.debug(f"global_value_pool shape: {global_value_pool.shape}")

    # Page table: map logical blocks to distinct physical pages
    # Shape: [1, num_logical_blocks_needed]
    page_table_data = [[i for i in range(num_logical_blocks_needed)]]
    page_table = mx.array(page_table_data, dtype=mx.uint32)
    logger.debug(f"page_table shape: {page_table.shape}, values: {page_table.tolist()}")

    # Write positions: consecutive positions starting from start_logical_position
    # Shape: [num_new_tokens_in_chunk]
    current_token_write_positions = mx.array(
        [start_logical_position + i for i in range(num_new_tokens_in_chunk)], dtype=mx.int32
    )
    logger.debug(
        f"current_token_write_positions shape: {current_token_write_positions.shape}, "
        f"values: {current_token_write_positions.tolist()}"
    )

    # Query to sequence mapping: all tokens belong to sequence 0
    # Shape: [num_new_tokens_in_chunk]
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
    for i in range(num_new_tokens_in_chunk):
        logical_pos = start_logical_position + i
        logical_block = logical_pos // primitive_tokens_per_page
        slot_in_block = logical_pos % primitive_tokens_per_page
        physical_page = page_table_data[0][logical_block]

        # Expected keys and values for all heads
        for h in range(num_kv_heads):
            expected_key = [i * 10 + j + 1 for j in range(head_dim)]
            expected_value = [i * 10 + j + 5 for j in range(head_dim)]

            expected_k_pool[physical_page, slot_in_block, h, :] = mx.array(expected_key, dtype=dtype)
            expected_v_pool[physical_page, slot_in_block, h, :] = mx.array(expected_value, dtype=dtype)

        logger.debug(
            f"Token {i}: logical_pos={logical_pos}, physical_page={physical_page}, "
            f"slot={slot_in_block}, key={expected_key}"
        )

    # Data assertions
    logger.debug("Comparing key pools...")
    if not mx.allclose(updated_k_pool, expected_k_pool, rtol=1e-3, atol=1e-3):
        # Log differences for debugging
        for page in range(num_physical_pages):
            for slot in range(primitive_tokens_per_page):
                for h in range(num_kv_heads):
                    actual = updated_k_pool[page, slot, h, :].tolist()
                    expected = expected_k_pool[page, slot, h, :].tolist()
                    if actual != expected:
                        logger.error(
                            f"Mismatch at page={page}, slot={slot}, head={h}: actual={actual}, expected={expected}"
                        )

        pytest.fail(
            f"Data in updated_k_pool does not match expected_k_pool for "
            f"num_new_tokens_in_chunk={num_new_tokens_in_chunk}, "
            f"start_logical_position={start_logical_position}, "
            f"dtype={dtype}"
        )

    logger.debug("Comparing value pools...")
    if not mx.allclose(updated_v_pool, expected_v_pool, rtol=1e-3, atol=1e-3):
        # Log differences for debugging
        for page in range(num_physical_pages):
            for slot in range(primitive_tokens_per_page):
                for h in range(num_kv_heads):
                    actual = updated_v_pool[page, slot, h, :].tolist()
                    expected = expected_v_pool[page, slot, h, :].tolist()
                    if actual != expected:
                        logger.error(
                            f"Mismatch at page={page}, slot={slot}, head={h}: actual={actual}, expected={expected}"
                        )

        pytest.fail(
            f"Data in updated_v_pool does not match expected_v_pool for "
            f"num_new_tokens_in_chunk={num_new_tokens_in_chunk}, "
            f"start_logical_position={start_logical_position}, "
            f"dtype={dtype}"
        )

    logger.info(
        f"Test passed: {num_new_tokens_in_chunk} tokens correctly written "
        f"starting at logical position {start_logical_position} (dtype={dtype})"
    )
