import logging

import mlx.core as mx
import pytest

from proxy_attention_lab import fill_kv_pages

logger = logging.getLogger(__name__)


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
def test_fill_single_token_single_sequence(dtype):
    """Test filling a single token for a single sequence, for both float16 and bfloat16.

    This test verifies that the fill_kv_pages operation correctly writes
    new key and value data to the appropriate location in the global pools.

    Expected to fail initially as the Metal kernel is not yet implemented.
    """
    # Test parameters (simple case)
    num_new_tokens_total = 1
    num_sequences_in_batch = 1
    num_kv_heads = 1
    head_dim = 4
    primitive_tokens_per_page = 2
    num_physical_pages = 1

    logger.info(
        f"Test parameters: num_new_tokens_total={num_new_tokens_total}, "
        f"num_sequences_in_batch={num_sequences_in_batch}, "
        f"num_kv_heads={num_kv_heads}, head_dim={head_dim}, "
        f"tokens_per_page={primitive_tokens_per_page}, dtype={dtype}"
    )

    # Create distinct new keys and values
    new_keys = mx.array([[[1.0, 2.0, 3.0, 4.0]]], dtype=dtype)
    new_values = mx.array([[[5.0, 6.0, 7.0, 8.0]]], dtype=dtype)

    logger.debug(f"new_keys shape: {new_keys.shape}, values: {new_keys.tolist()}")
    logger.debug(f"new_values shape: {new_values.shape}, values: {new_values.tolist()}")

    # Initialize global pools with zeros
    # Shape: [num_physical_pages, tokens_per_page, num_kv_heads, head_dim]
    global_key_pool = mx.zeros((num_physical_pages, primitive_tokens_per_page, num_kv_heads, head_dim), dtype=dtype)
    global_value_pool = mx.zeros((num_physical_pages, primitive_tokens_per_page, num_kv_heads, head_dim), dtype=dtype)

    logger.debug(f"global_key_pool shape: {global_key_pool.shape}")
    logger.debug(f"global_value_pool shape: {global_value_pool.shape}")

    # Page table: sequence 0's logical block 0 maps to physical page 0
    page_table = mx.array([[0]], dtype=mx.uint32)
    logger.debug(f"page_table shape: {page_table.shape}, values: {page_table.tolist()}")

    # Write position: write the new token to logical position 0
    current_token_write_positions = mx.array([0], dtype=mx.int32)
    logger.debug(
        f"current_token_write_positions shape: {current_token_write_positions.shape}, "
        f"values: {current_token_write_positions.tolist()}"
    )

    # Query to sequence mapping: the new token belongs to sequence 0
    query_to_seq_map = mx.array([0], dtype=mx.uint32)
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

    # Define expected data after the fill operation
    # The new key [1.0, 2.0, 3.0, 4.0] should be written to page 0, slot 0
    expected_k_pool = mx.zeros((num_physical_pages, primitive_tokens_per_page, num_kv_heads, head_dim), dtype=dtype)
    expected_k_pool[0, 0, 0, :] = mx.array([1.0, 2.0, 3.0, 4.0], dtype=dtype)

    # The new value [5.0, 6.0, 7.0, 8.0] should be written to page 0, slot 0
    expected_v_pool = mx.zeros((num_physical_pages, primitive_tokens_per_page, num_kv_heads, head_dim), dtype=dtype)
    expected_v_pool[0, 0, 0, :] = mx.array([5.0, 6.0, 7.0, 8.0], dtype=dtype)

    # Data assertions - these are expected to fail initially
    logger.debug(f"updated_k_pool: {updated_k_pool.tolist()}")
    logger.debug(f"expected_k_pool: {expected_k_pool.tolist()}")
    logger.debug(f"updated_v_pool: {updated_v_pool.tolist()}")
    logger.debug(f"expected_v_pool: {expected_v_pool.tolist()}")

    if not mx.allclose(updated_k_pool, expected_k_pool, rtol=1e-3, atol=1e-3):
        pytest.fail(
            f"Data in updated_k_pool does not match expected_k_pool for dtype={dtype}. "
            "The Metal kernel needs to be implemented to write the key data."
        )

    if not mx.allclose(updated_v_pool, expected_v_pool, rtol=1e-3, atol=1e-3):
        pytest.fail(
            f"Data in updated_v_pool does not match expected_v_pool for dtype={dtype}. "
            "The Metal kernel needs to be implemented to write the value data."
        )

    logger.info(f"Test passed: Key and value data were correctly written to the pools for dtype={dtype}")
