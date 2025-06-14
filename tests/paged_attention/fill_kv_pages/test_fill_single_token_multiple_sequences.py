# tests/paged_attention/fill_kv_pages/test_fill_single_token_multiple_sequences.py
# Unit tests for filling single tokens from multiple sequences (batched decode).
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
def test_fill_batched_decode_distinct_pages_and_slots(dtype: mx.Dtype):
    """
    Tests filling one token each from multiple sequences into different pages
    and different slots, a common batched decode scenario.
    """
    # 1. Arrange: Define parameters and prepare all input arrays.
    num_kv_heads = 2
    head_dim = 32
    tokens_per_page = 4
    num_physical_pages = 4
    elements_per_thread = MEMORY_ALIGNMENT_BYTES // dtype.size

    logger.info(f"Running test_fill_batched_decode_distinct_pages_and_slots with dtype={dtype}")

    # Source data: one token per sequence
    new_keys = mx.array(
        [
            [  # Token for sequence 0
                [(0 * 100) + d for d in range(head_dim)],
                [(0 * 100) + d + 1000 for d in range(head_dim)],
            ],
            [  # Token for sequence 1
                [(1 * 100) + d for d in range(head_dim)],
                [(1 * 100) + d + 1000 for d in range(head_dim)],
            ],
        ],
        dtype=dtype,
    )
    new_values = new_keys + 5000

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

    # Paging metadata:
    # - Seq 0 writes to physical page 1, slot 0
    # - Seq 1 writes to physical page 3, slot 1
    page_table = mx.array([[1], [3]], dtype=mx.uint32)
    current_token_write_positions = mx.array([0, 1], dtype=mx.int32)
    query_to_seq_map = mx.array([0, 1], dtype=mx.uint32)

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

    # Expected data for sequence 0
    for h in range(num_kv_heads):
        for d in range(head_dim):
            expected_v_pool[1, h, d, 0] = new_values[0, h, d]
            vec_chunk = d // elements_per_thread
            elem_in_vec = d % elements_per_thread
            expected_k_pool[1, h, vec_chunk, 0, elem_in_vec] = new_keys[0, h, d]

    # Expected data for sequence 1
    for h in range(num_kv_heads):
        for d in range(head_dim):
            expected_v_pool[3, h, d, 1] = new_values[1, h, d]
            vec_chunk = d // elements_per_thread
            elem_in_vec = d % elements_per_thread
            expected_k_pool[3, h, vec_chunk, 1, elem_in_vec] = new_keys[1, h, d]

    assert mx.allclose(updated_k_pool, expected_k_pool, atol=1e-3, rtol=1e-3)
    assert mx.allclose(updated_v_pool, expected_v_pool, atol=1e-3, rtol=1e-3)
    logger.info(f"Test passed for dtype={dtype}.")


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
def test_fill_batched_decode_same_page(dtype: mx.Dtype):
    """
    Tests filling tokens from multiple sequences into the *same* physical page
    but *different* slots.
    """
    # 1. Arrange
    num_sequences = 2
    num_kv_heads = 2
    head_dim = 32
    tokens_per_page = 4
    num_physical_pages = 4
    elements_per_thread = MEMORY_ALIGNMENT_BYTES // dtype.size

    logger.info(f"Running test_fill_batched_decode_same_page with dtype={dtype}")

    new_keys = mx.random.normal((num_sequences, num_kv_heads, head_dim)).astype(dtype)
    new_values = mx.random.normal((num_sequences, num_kv_heads, head_dim)).astype(dtype)

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

    # Paging metadata: Both sequences write to physical page 2
    page_table = mx.array([[2], [2]], dtype=mx.uint32)
    # Seq 0 writes to slot 1, Seq 1 writes to slot 3
    current_token_write_positions = mx.array([1, 3], dtype=mx.int32)
    query_to_seq_map = mx.array([0, 1], dtype=mx.uint32)

    # 2. Act
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

    # 3. Assert
    expected_k_pool = mx.zeros_like(global_key_pool)
    expected_v_pool = mx.zeros_like(global_value_pool)

    # Seq 0 writes to page 2, slot 1
    for h in range(num_kv_heads):
        for d in range(head_dim):
            expected_v_pool[2, h, d, 1] = new_values[0, h, d]
            vec_chunk = d // elements_per_thread
            elem_in_vec = d % elements_per_thread
            expected_k_pool[2, h, vec_chunk, 1, elem_in_vec] = new_keys[0, h, d]

    # Seq 1 writes to page 2, slot 3
    for h in range(num_kv_heads):
        for d in range(head_dim):
            expected_v_pool[2, h, d, 3] = new_values[1, h, d]
            vec_chunk = d // elements_per_thread
            elem_in_vec = d % elements_per_thread
            expected_k_pool[2, h, vec_chunk, 3, elem_in_vec] = new_keys[1, h, d]

    assert mx.allclose(updated_k_pool, expected_k_pool, atol=1e-3, rtol=1e-3)
    assert mx.allclose(updated_v_pool, expected_v_pool, atol=1e-3, rtol=1e-3)
    logger.info(f"Test passed for dtype={dtype}.")
