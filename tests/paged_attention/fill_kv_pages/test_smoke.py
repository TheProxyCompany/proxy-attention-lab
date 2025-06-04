# Copyright 2025 The Proxy Company. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Smoke test for fill_kv_pages functionality."""
# tests/fill_kv_pages/test_smoke.py

import logging

import mlx.core as mx
import numpy as np  # For creating initial data if needed

from proxy_attention_lab import fill_kv_pages  # Import the Python op

logger = logging.getLogger(__name__)


def test_fill_kv_pages_smoke() -> None:
    """
    Smoke test: verifies fill_kv_pages runs without error on minimal, valid inputs.
    Ensures the Python op, C++ primitive, and Metal kernel interface are functional.
    """
    logger.info(f"Test: {test_fill_kv_pages_smoke.__name__}")

    # Define parameters for a small test case
    num_new_tokens = 4  # e.g., batch_size=2, 2 new tokens each (or 4 sequences, 1 new token each)
    num_sequences_in_batch = 2  # Must be <= num_new_tokens if each seq gets >= 1 token

    # KV cache geometry (matches FillKVPagesPrimitive constructor and FillKVPagesParams)
    num_kv_heads = 8
    head_dim = 64
    tokens_per_page = 16  # Small for testing

    # Global pool dimensions
    num_physical_pages = 10  # Enough pages for this small test

    # Max logical blocks per sequence in page_table
    # Needs to be large enough to hold all tokens for any sequence based on its `current_token_positions`
    # For this test, assume a max sequence length that fits a few pages
    max_logical_blocks_per_seq = 4

    dtype = mx.float16  # Common dtype for K/V cache

    # 1. Create new_keys and new_values
    # Shape: [num_new_tokens, num_kv_heads, head_dim]
    new_keys = mx.random.normal([num_new_tokens, num_kv_heads, head_dim], dtype=dtype)
    new_values = mx.random.normal([num_new_tokens, num_kv_heads, head_dim], dtype=dtype)

    # 2. Create global_key_pool and global_value_pool (these will be "updated")
    # Shape: [num_physical_pages, tokens_per_page, num_kv_heads, head_dim]
    global_key_pool = mx.zeros([num_physical_pages, tokens_per_page, num_kv_heads, head_dim], dtype=dtype)
    global_value_pool = mx.zeros([num_physical_pages, tokens_per_page, num_kv_heads, head_dim], dtype=dtype)

    # 3. Create page_table
    # Shape: [num_sequences_in_batch, max_logical_blocks_per_seq]
    # For smoke test, assign unique physical pages to logical blocks
    pt_data = mx.arange(num_sequences_in_batch * max_logical_blocks_per_seq, dtype=mx.uint32) % num_physical_pages
    page_table = mx.array(pt_data.reshape(num_sequences_in_batch, max_logical_blocks_per_seq))

    # 4. Create current_token_positions
    # Shape: [num_new_tokens]. Each element is the logical slot index to write to.
    # Example: two sequences, two tokens each.
    # Seq 0 gets tokens at pos 0, 1. Seq 1 gets tokens at pos 0, 1.
    # (This assumes num_new_tokens is total tokens across all sequences in this update step)
    # If num_new_tokens = 4, num_sequences_in_batch = 2, then assume first 2 tokens for seq 0, next 2 for seq 1
    # This needs to align with query_to_seq_map.
    # Example: tokens write to logical positions [0, 1, 0, 1] for their respective sequences
    # A simpler setup for smoke: assume each new token writes to a unique logical position
    # For this test, let's make it simple: each new token is for a *different* sequence,
    # and writes to the start of that sequence. This makes query_to_seq_map and positions simpler.
    # To do this, num_new_tokens must equal num_sequences_in_batch for this simple setup.
    if num_new_tokens != num_sequences_in_batch:
        # Adjust for a simpler smoke test mapping: one new token per sequence
        logger.warning(f"Adjusting num_new_tokens to {num_sequences_in_batch} for simpler smoke test mapping.")
        num_new_tokens = num_sequences_in_batch
        new_keys = mx.random.normal([num_new_tokens, num_kv_heads, head_dim], dtype=dtype)
        new_values = mx.random.normal([num_new_tokens, num_kv_heads, head_dim], dtype=dtype)

    # Each new token writes to logical position 0 of its sequence
    current_token_positions = mx.array(np.zeros(num_new_tokens, dtype=np.int32))

    # 5. Create query_to_seq_map
    # Shape: [num_new_tokens]. Maps each new token to its sequence index in the batch.
    # If one new token per sequence: [0, 1, 2, ..., num_new_tokens-1]
    query_to_seq_map = mx.array(np.arange(num_new_tokens, dtype=np.uint32))

    logger.debug("Smoke test inputs prepared:")
    logger.debug(f"  new_keys.shape: {new_keys.shape}, dtype: {new_keys.dtype}")
    logger.debug(f"  new_values.shape: {new_values.shape}, dtype: {new_values.dtype}")
    logger.debug(f"  global_key_pool.shape: {global_key_pool.shape}, dtype: {global_key_pool.dtype}")
    logger.debug(f"  global_value_pool.shape: {global_value_pool.shape}, dtype: {global_value_pool.dtype}")
    logger.debug(f"  page_table.shape: {page_table.shape}, dtype: {page_table.dtype}")
    logger.debug(
        f"  current_token_positions.shape: {current_token_positions.shape}, dtype: {current_token_positions.dtype}"
    )
    logger.debug(f"  query_to_seq_map.shape: {query_to_seq_map.shape}, dtype: {query_to_seq_map.dtype}")

    try:
        # Call the operation
        mx.eval(
            new_keys,
            new_values,
            global_key_pool,
            global_value_pool,
            page_table,
            current_token_positions,
            query_to_seq_map,
        )  # Ensure inputs are on device

        updated_k_pool, updated_v_pool = fill_kv_pages(
            new_keys,
            new_values,
            global_key_pool,
            global_value_pool,
            page_table,
            current_token_positions,
            query_to_seq_map,
        )

        # Evaluate the outputs to ensure the kernel runs
        mx.eval(updated_k_pool, updated_v_pool)

        logger.info("fill_kv_pages smoke test executed successfully.")

        # Basic shape and dtype checks on outputs
        assert updated_k_pool.shape == global_key_pool.shape, "Updated K pool shape mismatch"
        assert updated_k_pool.dtype == global_key_pool.dtype, "Updated K pool dtype mismatch"
        assert updated_v_pool.shape == global_value_pool.shape, "Updated V pool shape mismatch"
        assert updated_v_pool.dtype == global_value_pool.dtype, "Updated V pool dtype mismatch"

    except Exception as e:
        logger.error(f"fill_kv_pages smoke test failed: {e}", exc_info=True)
        raise

    # Check if the output arrays are the same objects as the inputs
    logger.debug(f"Input K pool ID: {id(global_key_pool)}, Output K pool ID: {id(updated_k_pool)}")
    logger.debug(f"Input V pool ID: {id(global_value_pool)}, Output V pool ID: {id(updated_v_pool)}")
