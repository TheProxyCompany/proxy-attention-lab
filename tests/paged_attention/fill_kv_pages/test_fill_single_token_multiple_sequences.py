import logging

import mlx.core as mx
import pytest

from proxy_attention_lab import fill_kv_pages

logger = logging.getLogger(__name__)


@pytest.mark.parametrize(
    "num_sequences_in_batch,write_to_same_page_slot_type",
    [
        (2, "distinct_pages_distinct_slots"),
        (4, "distinct_pages_distinct_slots"),
        (2, "same_page_distinct_slots"),
        (4, "same_page_distinct_slots"),
        (2, "distinct_pages_same_slot"),
        (4, "distinct_pages_same_slot"),
    ],
)
@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
def test_fill_single_token_multiple_sequences(num_sequences_in_batch, write_to_same_page_slot_type, dtype):
    """Test filling single token each for multiple sequences (simulating batched decode).

    This test verifies that the fill_kv_pages operation correctly writes
    new key and value data when multiple sequences each contribute one token,
    simulating a batched decode operation.

    Args:
        num_sequences_in_batch: Number of sequences in the batch
        write_to_same_page_slot_type: Scenario type for page/slot assignment:
            - "distinct_pages_distinct_slots": each sequence writes to different page & slot
            - "same_page_distinct_slots": all sequences write to same page, different slots
            - "distinct_pages_same_slot": sequences write to different pages, same slot
        dtype: Data type to use for the test (mx.float16 or mx.bfloat16)
    """
    # Fixed test parameters
    num_kv_heads = 1
    head_dim = 4
    primitive_tokens_per_page = 4

    # Determine physical pages needed based on scenario
    if write_to_same_page_slot_type == "distinct_pages_distinct_slots":
        num_physical_pages = num_sequences_in_batch + 1  # Each seq gets its own page
    elif write_to_same_page_slot_type == "same_page_distinct_slots":
        num_physical_pages = 2  # All sequences share one page
    else:  # distinct_pages_same_slot
        num_physical_pages = num_sequences_in_batch + 1  # Each seq gets its own page

    logger.info(
        f"Test parameters: num_sequences_in_batch={num_sequences_in_batch}, "
        f"write_type={write_to_same_page_slot_type}, "
        f"num_kv_heads={num_kv_heads}, head_dim={head_dim}, "
        f"tokens_per_page={primitive_tokens_per_page}, "
        f"num_physical_pages={num_physical_pages}, dtype={dtype}"
    )

    # Create distinct new keys and values
    # Shape: [num_sequences_in_batch, num_kv_heads, head_dim]
    new_keys_data = []
    new_values_data = []
    for seq_idx in range(num_sequences_in_batch):
        # Keys: [seq_idx*10+1, seq_idx*10+2, seq_idx*10+3, seq_idx*10+4]
        # Values: [seq_idx*10+5, seq_idx*10+6, seq_idx*10+7, seq_idx*10+8]
        key_row = [seq_idx * 10 + j + 1 for j in range(head_dim)]
        value_row = [seq_idx * 10 + j + 5 for j in range(head_dim)]
        new_keys_data.append([key_row])  # Extra dimension for kv_heads
        new_values_data.append([value_row])

    new_keys = mx.array(new_keys_data, dtype=dtype)
    new_values = mx.array(new_values_data, dtype=dtype)

    logger.debug(f"new_keys shape: {new_keys.shape}")
    logger.debug(f"new_values shape: {new_values.shape}")

    # Initialize global pools with zeros
    global_key_pool = mx.zeros((num_physical_pages, primitive_tokens_per_page, num_kv_heads, head_dim), dtype=dtype)
    global_value_pool = mx.zeros((num_physical_pages, primitive_tokens_per_page, num_kv_heads, head_dim), dtype=dtype)

    # Setup page table and write positions based on scenario
    page_table_data = []
    current_token_write_positions_data = []

    for seq_idx in range(num_sequences_in_batch):
        if write_to_same_page_slot_type == "distinct_pages_distinct_slots":
            # Each sequence maps to its own page
            page_table_row = [seq_idx, seq_idx]  # Two logical blocks, both map to same physical page
            # Each sequence writes to a different slot
            write_position = seq_idx % primitive_tokens_per_page
        elif write_to_same_page_slot_type == "same_page_distinct_slots":
            # All sequences map to the same physical page (page 0)
            page_table_row = [0, 0]
            # Each sequence writes to a different slot
            write_position = seq_idx % primitive_tokens_per_page
        else:  # distinct_pages_same_slot
            # Each sequence maps to its own page
            page_table_row = [seq_idx, seq_idx]
            # All sequences write to the same slot (slot 0)
            write_position = 0

        page_table_data.append(page_table_row)
        current_token_write_positions_data.append(write_position)

    page_table = mx.array(page_table_data, dtype=mx.uint32)
    current_token_write_positions = mx.array(current_token_write_positions_data, dtype=mx.int32)

    logger.debug(f"page_table shape: {page_table.shape}, values: {page_table.tolist()}")
    logger.debug(
        f"current_token_write_positions shape: {current_token_write_positions.shape}, "
        f"values: {current_token_write_positions.tolist()}"
    )

    # Query to sequence mapping: token i belongs to sequence i
    query_to_seq_map = mx.array(list(range(num_sequences_in_batch)), dtype=mx.uint32)
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

    # Fill expected data based on where each sequence's token should be written
    for seq_idx in range(num_sequences_in_batch):
        logical_pos = current_token_write_positions_data[seq_idx]
        logical_block = logical_pos // primitive_tokens_per_page
        slot_in_block = logical_pos % primitive_tokens_per_page
        physical_page = page_table_data[seq_idx][logical_block]

        # Expected keys and values for this sequence
        expected_key = [seq_idx * 10 + j + 1 for j in range(head_dim)]
        expected_value = [seq_idx * 10 + j + 5 for j in range(head_dim)]

        expected_k_pool[physical_page, slot_in_block, 0, :] = mx.array(expected_key, dtype=dtype)
        expected_v_pool[physical_page, slot_in_block, 0, :] = mx.array(expected_value, dtype=dtype)

        logger.debug(
            f"Sequence {seq_idx}: logical_pos={logical_pos}, physical_page={physical_page}, "
            f"slot={slot_in_block}, key={expected_key}"
        )

    # Data assertions
    logger.debug("Comparing key pools...")
    if not mx.allclose(updated_k_pool, expected_k_pool, rtol=1e-3, atol=1e-3):
        # Log differences for debugging
        for page in range(num_physical_pages):
            for slot in range(primitive_tokens_per_page):
                actual = updated_k_pool[page, slot, 0, :].tolist()
                expected = expected_k_pool[page, slot, 0, :].tolist()
                if actual != expected:
                    logger.error(f"Mismatch at page={page}, slot={slot}: actual={actual}, expected={expected}")

        pytest.fail(
            f"Data in updated_k_pool does not match expected_k_pool for "
            f"num_sequences_in_batch={num_sequences_in_batch}, "
            f"write_type={write_to_same_page_slot_type}, "
            f"dtype={dtype}"
        )

    logger.debug("Comparing value pools...")
    if not mx.allclose(updated_v_pool, expected_v_pool, rtol=1e-3, atol=1e-3):
        # Log differences for debugging
        for page in range(num_physical_pages):
            for slot in range(primitive_tokens_per_page):
                actual = updated_v_pool[page, slot, 0, :].tolist()
                expected = expected_v_pool[page, slot, 0, :].tolist()
                if actual != expected:
                    logger.error(f"Mismatch at page={page}, slot={slot}: actual={actual}, expected={expected}")

        pytest.fail(
            f"Data in updated_v_pool does not match expected_v_pool for "
            f"num_sequences_in_batch={num_sequences_in_batch}, "
            f"write_type={write_to_same_page_slot_type}, "
            f"dtype={dtype}"
        )

    logger.info(
        f"Test passed: {num_sequences_in_batch} sequences correctly written "
        f"with write_type={write_to_same_page_slot_type}, dtype={dtype}"
    )
