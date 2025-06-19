# tests/paged_attention/fill_kv_pages/test_fill_multiple_tokens_single_sequence.py
# Unit tests for filling chunks of tokens with the fill_kv_pages operation.
#
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
# ============================================================================

import logging

import mlx.core as mx
import pytest

from proxy_attention_lab import fill_kv_pages

# A memory alignment of 16 bytes is standard for Metal.
MEMORY_ALIGNMENT_BYTES = 16

logger = logging.getLogger(__name__)


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
def test_fill_multiple_tokens_within_page(dtype: mx.Dtype):
    """
    Tests filling multiple tokens that all fit within a single page.
    """
    # 1. Arrange: Define parameters and prepare all input arrays.
    num_new_tokens = 3
    num_kv_heads = 2
    head_dim = 32
    tokens_per_page = 4
    num_physical_pages = 2
    elements_per_thread = MEMORY_ALIGNMENT_BYTES // dtype.size

    logger.info(f"Running test_fill_multiple_tokens_within_page with dtype={dtype}")

    # Source data: make values unique for each token and head
    new_keys_list = [
        [[(t * 1000) + (h * 100) + d for d in range(head_dim)] for h in range(num_kv_heads)]
        for t in range(num_new_tokens)
    ]
    new_values_list = [
        [[(t * 1000) + (h * 100) + d + 5000 for d in range(head_dim)] for h in range(num_kv_heads)]
        for t in range(num_new_tokens)
    ]
    new_keys = mx.array(new_keys_list, dtype=dtype)
    new_values = mx.array(new_values_list, dtype=dtype)

    # Destination caches, initially empty
    global_key_pool = mx.zeros(
        (
            num_physical_pages,
            num_kv_heads,
            head_dim // elements_per_thread,
            tokens_per_page,
            elements_per_thread,
        ),
        dtype=dtype,
    )
    global_value_pool = mx.zeros(
        (
            num_physical_pages,
            num_kv_heads,
            head_dim,
            tokens_per_page,
        ),
        dtype=dtype,
    )

    # Paging metadata: all tokens write to page 1
    page_table = mx.array([[1]], dtype=mx.uint32)
    current_token_write_positions = mx.arange(num_new_tokens, dtype=mx.int32)
    query_to_seq_map = mx.zeros(num_new_tokens, dtype=mx.uint32)

    # 2. Act: Execute the kernel.
    updated_k_pool, updated_v_pool = fill_kv_pages(
        new_keys=new_keys,
        new_values=new_values,
        global_key_pool=global_key_pool,
        global_value_pool=global_value_pool,
        page_table=page_table,
        current_token_write_positions=current_token_write_positions,
        query_to_seq_map=query_to_seq_map,
    )
    mx.eval(updated_k_pool, updated_v_pool)

    # 3. Assert: Define expected results and verify the outputs.
    expected_k_pool = mx.zeros_like(global_key_pool)
    expected_v_pool = mx.zeros_like(global_value_pool)
    for t in range(num_new_tokens):
        physical_page = 1
        slot_in_page = t
        for h in range(num_kv_heads):
            for d in range(head_dim):
                # V-cache is simple strided write
                expected_v_pool[physical_page, h, d, slot_in_page] = new_values[t, h, d]
                # K-cache is interleaved
                vec_chunk_idx = d // elements_per_thread
                elem_in_vec_idx = d % elements_per_thread
                expected_k_pool[physical_page, h, vec_chunk_idx, slot_in_page, elem_in_vec_idx] = new_keys[t, h, d]

    # Verification
    assert mx.allclose(updated_k_pool, expected_k_pool, atol=1e-3, rtol=1e-3)
    assert mx.allclose(updated_v_pool, expected_v_pool, atol=1e-3, rtol=1e-3)
    logger.info(f"Test passed for dtype={dtype}.")


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
def test_fill_multiple_tokens_across_pages(dtype: mx.Dtype):
    """
    Tests filling multiple tokens that cross a page boundary.
    """
    # 1. Arrange: Define parameters and prepare all input arrays.
    num_new_tokens = 3
    start_logical_pos = 1
    num_kv_heads = 2
    head_dim = 32
    tokens_per_page = 2
    num_physical_pages = 12  # Use distinct pages
    elements_per_thread = MEMORY_ALIGNMENT_BYTES // dtype.size

    logger.info(f"Running test_fill_multiple_tokens_across_pages with dtype={dtype}")

    # Source data
    new_keys_list = [
        [[(t * 1000) + (h * 100) + d for d in range(head_dim)] for h in range(num_kv_heads)]
        for t in range(num_new_tokens)
    ]
    new_values_list = [
        [[(t * 1000) + (h * 100) + d + 5000 for d in range(head_dim)] for h in range(num_kv_heads)]
        for t in range(num_new_tokens)
    ]
    new_keys = mx.array(new_keys_list, dtype=dtype)
    new_values = mx.array(new_values_list, dtype=dtype)

    # Destination caches, initially empty
    global_key_pool = mx.zeros(
        (
            num_physical_pages,
            num_kv_heads,
            head_dim // elements_per_thread,
            tokens_per_page,
            elements_per_thread,
        ),
        dtype=dtype,
    )
    global_value_pool = mx.zeros(
        (
            num_physical_pages,
            num_kv_heads,
            head_dim,
            tokens_per_page,
        ),
        dtype=dtype,
    )

    # Paging metadata: map logical blocks to distinct physical pages
    # Logical block 0 -> physical page 5
    # Logical block 1 -> physical page 10
    page_table = mx.array([[5, 10]], dtype=mx.uint32)
    write_positions = mx.arange(start_logical_pos, start_logical_pos + num_new_tokens, dtype=mx.int32)
    query_to_seq_map = mx.zeros(num_new_tokens, dtype=mx.uint32)

    # 2. Act: Execute the kernel.
    updated_k_pool, updated_v_pool = fill_kv_pages(
        new_keys=new_keys,
        new_values=new_values,
        global_key_pool=global_key_pool,
        global_value_pool=global_value_pool,
        page_table=page_table,
        current_token_write_positions=write_positions,
        query_to_seq_map=query_to_seq_map,
    )
    mx.eval(updated_k_pool, updated_v_pool)

    # 3. Assert: Define expected results and verify the outputs.
    expected_k_pool = mx.zeros_like(global_key_pool)
    expected_v_pool = mx.zeros_like(global_value_pool)
    for t in range(num_new_tokens):
        logical_pos = start_logical_pos + t
        logical_block = logical_pos // tokens_per_page
        slot_in_page = logical_pos % tokens_per_page
        physical_page = page_table[0, logical_block].item()

        for h in range(num_kv_heads):
            for d in range(head_dim):
                expected_v_pool[physical_page, h, d, slot_in_page] = new_values[t, h, d]
                vec_chunk_idx = d // elements_per_thread
                elem_in_vec_idx = d % elements_per_thread
                expected_k_pool[physical_page, h, vec_chunk_idx, slot_in_page, elem_in_vec_idx] = new_keys[t, h, d]

    # Verification
    assert mx.allclose(updated_k_pool, expected_k_pool, atol=1e-3, rtol=1e-3)
    assert mx.allclose(updated_v_pool, expected_v_pool, atol=1e-3, rtol=1e-3)
    logger.info(f"Test passed for dtype={dtype}.")


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
def test_fill_full_page_plus_one_token(dtype: mx.Dtype):
    """
    Tests filling exactly one page (16 tokens) plus one additional token.
    This tests the specific scenario from production logs where 17 tokens are written.
    """
    # 1. Arrange: Define parameters matching the production scenario
    num_new_tokens = 17
    num_kv_heads = 8
    head_dim = 128
    tokens_per_page = 16
    num_physical_pages = 3  # We only need 2 pages for this test
    elements_per_thread = 8  # 128 / 16 = 8, matching the k_cache shape

    logger.info(f"Running test_fill_full_page_plus_one_token with dtype={dtype}")
    logger.info(f"Testing scenario: {num_new_tokens} tokens with page size {tokens_per_page}")

    # Source data: Create unique values for each token, head, and dimension
    new_keys_list = [
        [[(t * 10000) + (h * 1000) + d for d in range(head_dim)] for h in range(num_kv_heads)]
        for t in range(num_new_tokens)
    ]
    new_values_list = [
        [[(t * 10000) + (h * 1000) + d + 50000 for d in range(head_dim)] for h in range(num_kv_heads)]
        for t in range(num_new_tokens)
    ]
    new_keys = mx.array(new_keys_list, dtype=dtype)
    new_values = mx.array(new_values_list, dtype=dtype)

    # Verify input shapes match production scenario
    assert new_keys.shape == (num_new_tokens, num_kv_heads, head_dim)
    assert new_values.shape == (num_new_tokens, num_kv_heads, head_dim)

    # Destination caches matching production shapes
    global_key_pool = mx.zeros(
        (
            num_physical_pages,
            num_kv_heads,
            head_dim // elements_per_thread,  # 128 / 8 = 16
            tokens_per_page,
            elements_per_thread,
        ),
        dtype=dtype,
    )
    global_value_pool = mx.zeros(
        (
            num_physical_pages,
            num_kv_heads,
            head_dim,
            tokens_per_page,
        ),
        dtype=dtype,
    )

    # Verify cache shapes
    assert global_key_pool.shape == (num_physical_pages, 8, 16, 16, 8)
    assert global_value_pool.shape == (num_physical_pages, 8, 128, 16)

    # Paging metadata: Use pages 1 and 2 (avoiding page 0 for clarity)
    page_table = mx.array([[1, 2]], dtype=mx.uint32)  # Shape [1, 2] as in production
    current_token_write_positions = mx.arange(num_new_tokens, dtype=mx.int32)  # [0, 1, ..., 16]
    query_to_seq_map = mx.zeros(num_new_tokens, dtype=mx.uint32)  # All tokens belong to sequence 0

    # 2. Act: Execute the kernel
    updated_k_pool, updated_v_pool = fill_kv_pages(
        new_keys=new_keys,
        new_values=new_values,
        global_key_pool=global_key_pool,
        global_value_pool=global_value_pool,
        page_table=page_table,
        current_token_write_positions=current_token_write_positions,
        query_to_seq_map=query_to_seq_map,
    )
    mx.eval(updated_k_pool, updated_v_pool)

    # 3. Assert: Verify the outputs
    expected_k_pool = mx.zeros_like(global_key_pool)
    expected_v_pool = mx.zeros_like(global_value_pool)

    for t in range(num_new_tokens):
        logical_pos = t
        logical_block = logical_pos // tokens_per_page  # 0 for first 16 tokens, 1 for token 16
        slot_in_page = logical_pos % tokens_per_page
        physical_page = page_table[0, logical_block].item()

        logger.debug(f"Token {t}: logical_block={logical_block}, slot={slot_in_page}, physical_page={physical_page}")

        for h in range(num_kv_heads):
            for d in range(head_dim):
                # V-cache: straightforward write
                expected_v_pool[physical_page, h, d, slot_in_page] = new_values[t, h, d]

                # K-cache: interleaved layout
                vec_chunk_idx = d // elements_per_thread
                elem_in_vec_idx = d % elements_per_thread
                expected_k_pool[physical_page, h, vec_chunk_idx, slot_in_page, elem_in_vec_idx] = new_keys[t, h, d]

    # Verification with appropriate tolerances
    assert mx.allclose(updated_k_pool, expected_k_pool, atol=1e-3, rtol=1e-3)
    assert mx.allclose(updated_v_pool, expected_v_pool, atol=1e-3, rtol=1e-3)

    # Additional verification: Check that exactly two pages were used
    # Page 1 should have all 16 slots filled
    for slot in range(tokens_per_page):
        assert mx.any(updated_k_pool[1, :, :, slot, :] != 0)
        assert mx.any(updated_v_pool[1, :, :, slot] != 0)

    # Page 2 should have only the first slot filled
    assert mx.any(updated_k_pool[2, :, :, 0, :] != 0)
    assert mx.any(updated_v_pool[2, :, :, 0] != 0)

    # Page 2 slots 1-15 should be empty
    for slot in range(1, tokens_per_page):
        assert mx.all(updated_k_pool[2, :, :, slot, :] == 0)
        assert mx.all(updated_v_pool[2, :, :, slot] == 0)

    logger.info(f"Test passed for dtype={dtype}. Successfully filled {num_new_tokens} tokens across 2 pages.")
