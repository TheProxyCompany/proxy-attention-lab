# tests/paged_attention/fill_kv_pages/test_fill_kv_pages_data.py
# Unit tests for the fill_kv_pages operation.
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
def test_fill_single_token(dtype: mx.Dtype):
    """
    Tests filling a single token for a single sequence.

    This test verifies that the kernel correctly writes new key and value
    data into the K and V caches according to their specified memory layouts.
    """
    # 1. Arrange: Define parameters and prepare all input arrays.
    num_new_tokens = 1
    num_kv_heads = 1
    head_dim = 64
    tokens_per_page = 2
    num_physical_pages = 1
    elements_per_thread = MEMORY_ALIGNMENT_BYTES // dtype.size

    logger.info(
        f"Running test_fill_single_token with dtype={dtype}, head_dim={head_dim}, tokens_per_page={tokens_per_page}"
    )

    # Source data to be written
    new_keys = mx.arange(1, head_dim + 1, dtype=dtype).reshape(num_new_tokens, num_kv_heads, head_dim)
    new_values = mx.arange(head_dim + 1, 2 * head_dim + 1, dtype=dtype).reshape(num_new_tokens, num_kv_heads, head_dim)

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

    # Paging metadata for a single token in a single sequence
    page_table = mx.array([[0]], dtype=mx.uint32)
    current_token_write_positions = mx.array([0], dtype=mx.int32)
    query_to_seq_map = mx.array([0], dtype=mx.uint32)

    # 2. Act: Execute the kernel.
    logger.info("Calling pal_core.ops.fill_kv_pages...")
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
    logger.info("Kernel execution completed.")

    # 3. Assert: Define expected results and verify the outputs.
    # Expected K-cache with interleaved layout
    expected_k_pool = mx.zeros_like(global_key_pool)
    for i in range(head_dim):
        vec_chunk_idx = i // elements_per_thread
        elem_in_vec_idx = i % elements_per_thread
        expected_k_pool[0, 0, vec_chunk_idx, 0, elem_in_vec_idx] = new_keys[0, 0, i]

    # Expected V-cache with standard layout
    expected_v_pool = mx.zeros_like(global_value_pool)
    for i in range(head_dim):
        expected_v_pool[0, 0, i, 0] = new_values[0, 0, i]

    # Verification
    if not mx.allclose(updated_k_pool, expected_k_pool, atol=1e-3, rtol=1e-3):
        logger.error("K-cache verification failed.")
        pytest.fail(f"Data in updated_k_pool does not match expected_k_pool for dtype={dtype}.")

    if not mx.allclose(updated_v_pool, expected_v_pool, atol=1e-3, rtol=1e-3):
        logger.error("V-cache verification failed.")
        pytest.fail(f"Data in updated_v_pool does not match expected_v_pool for dtype={dtype}.")

    logger.info(f"Test passed for dtype={dtype}.")
