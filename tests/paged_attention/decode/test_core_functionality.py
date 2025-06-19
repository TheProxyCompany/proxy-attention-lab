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
"""Core functionality tests for paged attention operations.

This module contains tests that verify the core functionality of the paged attention
mechanism, including token fetching, vector operations, and the attention computation.
"""

import logging

import mlx.core as mx
import pytest

from proxy_attention_lab.pal_core import (
    get_k_cache_shape,
    get_k_cache_stripe_size,
    get_v_cache_shape,
    paged_attention_decode,
)

logger = logging.getLogger(__name__)


@pytest.mark.parametrize("head_dim", [32, 64, 128, 256])
@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
def test_fetch_k_vector_element_for_first_token_of_sequence(head_dim, dtype) -> None:
    """Test K vector fetch for first token with dot product of Q and K.

    This test verifies that the paged attention operation correctly fetches
    K vector elements for the first token of each sequence and applies the
    appropriate scaling factor to the dot product.
    """
    # Configuration
    num_sequences = 2
    num_q_heads = 1  # Single query head per sequence
    cfg_tokens_per_page = 16
    cfg_num_kv_heads = 1
    cfg_head_dim = head_dim
    cfg_max_logical_pages_per_seq_in_pagetable = 2

    # Create 3D queries with shape [num_sequences, num_q_heads, cfg_head_dim]
    py_queries = mx.zeros((num_sequences, num_q_heads, cfg_head_dim), dtype=dtype)
    # Set Q-vectors with specific values - only first element non-zero
    py_queries[0, 0, 0] = 100.0  # Sequence 0, head 0: [100.0, 0.0, 0.0, ..., 0.0]
    py_queries[1, 0, 0] = 200.0  # Sequence 1, head 0: [200.0, 0.0, 0.0, ..., 0.0]

    # Create K-cache with specific values
    num_physical_pages = 2
    # Use get_k_cache_shape() for striped format
    k_cache_shape = get_k_cache_shape(num_physical_pages, cfg_num_kv_heads, cfg_head_dim, cfg_tokens_per_page, dtype)
    py_k_cache_pool = mx.zeros(k_cache_shape, dtype=dtype)

    # Set K-vector for page 0, kv_head 0, token 0 to [11.0, 0.0, 0.0, ..., 0.0]
    k_vec = mx.zeros(cfg_head_dim, dtype=dtype)
    k_vec[0] = 11.0
    for d in range(cfg_head_dim):
        stripe_idx = d // get_k_cache_stripe_size(dtype)
        offset_in_stripe = d % get_k_cache_stripe_size(dtype)
        py_k_cache_pool[0, 0, stripe_idx, 0, offset_in_stripe] = k_vec[d]

    # Set K-vector for page 1, kv_head 0, token 0 to [22.0, 0.0, 0.0, ..., 0.0]
    k_vec = mx.zeros(cfg_head_dim, dtype=dtype)
    k_vec[0] = 22.0
    for d in range(cfg_head_dim):
        stripe_idx = d // get_k_cache_stripe_size(dtype)
        offset_in_stripe = d % get_k_cache_stripe_size(dtype)
        py_k_cache_pool[1, 0, stripe_idx, 0, offset_in_stripe] = k_vec[d]

    # Set up V-cache with distinct values
    v_cache_shape = get_v_cache_shape(num_physical_pages, cfg_num_kv_heads, cfg_head_dim, cfg_tokens_per_page, dtype)
    py_v_cache_pool = mx.zeros(v_cache_shape, dtype=dtype)
    # Create a pattern that works for any head_dim
    v_pattern_0 = mx.arange(cfg_head_dim, dtype=dtype) * 10.0 + 10.0  # [10, 20, 30, ...]
    v_pattern_1 = mx.arange(cfg_head_dim, dtype=dtype) * 100.0 + 100.0  # [100, 200, 300, ...]
    py_v_cache_pool[0, 0, :, 0] = v_pattern_0
    py_v_cache_pool[1, 0, :, 0] = v_pattern_1

    # Page table maps: seq 0, block 0 -> page 0; seq 1, block 0 -> page 1
    py_page_table = mx.array(
        [
            [0, 99],  # Sequence 0: logical block 0 -> physical page 0, block 1 -> page 99 (unused)
            [1, 88],  # Sequence 1: logical block 0 -> physical page 1, block 1 -> page 88 (unused)
        ],
        dtype=mx.uint32,
    )
    assert py_page_table.shape == (num_sequences, cfg_max_logical_pages_per_seq_in_pagetable)

    # Set up sequence metadata
    py_sequence_lengths = mx.array([1, 1], dtype=mx.int32)  # Each sequence has 1 token

    py_queries = mx.contiguous(py_queries)
    py_k_cache_pool = mx.contiguous(py_k_cache_pool)
    py_v_cache_pool = mx.contiguous(py_v_cache_pool)
    py_page_table = mx.contiguous(py_page_table)
    py_sequence_lengths = mx.contiguous(py_sequence_lengths)

    # Print the shapes of the inputs
    mx.eval(py_queries)
    mx.eval(py_k_cache_pool)
    mx.eval(py_v_cache_pool)
    mx.eval(py_page_table)
    mx.eval(py_sequence_lengths)

    # Run paged attention
    output_arr = paged_attention_decode(
        py_queries,
        py_k_cache_pool,
        py_v_cache_pool,
        py_page_table,
        py_sequence_lengths,
    )
    mx.eval(output_arr)

    # --- Calculate expected output (Python reference) ---

    # Scale factor for attention scores
    scale = 1.0 / mx.sqrt(mx.array(float(cfg_head_dim))).item()

    # Expected scores (single token history means 100% attention weight):
    # Item 0: Q[0] dot K[0,0,0] * scale = 100.0 * 11.0 * scale
    # Item 1: Q[1] dot K[1,0,0] * scale = 200.0 * 22.0 * scale
    score0 = 100.0 * 11.0 * scale
    score1 = 200.0 * 22.0 * scale

    # For single history token, softmax prob is always 1.0

    # Create expected V outputs
    # V-aggregation for item 0: V[0,0,0] * prob[0] = V[0,0,0] * 1.0
    expected_V_item0 = py_v_cache_pool[0, 0, :, 0].astype(mx.float32)
    # V-aggregation for item 1: V[1,0,0] * prob[0] = V[1,0,0] * 1.0
    expected_V_item1 = py_v_cache_pool[1, 0, :, 0].astype(mx.float32)

    # Combine and reshape to match kernel output
    expected_V_output = mx.array([[expected_V_item0.astype(dtype)], [expected_V_item1.astype(dtype)]])

    # For 3D queries [num_sequences, num_q_heads, head_dim],
    # output shape is [num_sequences, num_q_heads, head_dim]
    expected_output_shape = (num_sequences, num_q_heads, cfg_head_dim)

    # Log test details
    logger.info(
        f"Test: {test_fetch_k_vector_element_for_first_token_of_sequence.__name__} head_dim={head_dim} dtype={dtype}"
    )
    logger.info(f"  Scale factor: {scale}")
    logger.info(f"  Score 0: {score0}, Score 1: {score1}")
    logger.info(f"  Expected output shape: {expected_output_shape}")
    logger.info(f"  Expected V output: {expected_V_output}")
    logger.info(f"  Actual V output: {output_arr}")

    # Verify results
    assert output_arr.shape == expected_output_shape, (
        f"Output shape {output_arr.shape} does not match expected {expected_output_shape}"
    )
    assert output_arr.dtype == dtype, f"Output dtype {output_arr.dtype} does not match {dtype}"

    if not mx.allclose(output_arr, expected_V_output, atol=1e-2, rtol=1e-2):
        logger.info(f"Output values(last 10): {output_arr[-10:]}")
        logger.info(f"Expected values(last 10): {expected_V_output[-10:]}")

        for i in range(output_arr.shape[0]):
            for j in range(output_arr.shape[1]):
                if output_arr[i, j] == 0:
                    logger.info(f"zero found at {i}, {j}")
                    if j > 10:
                        break
                else:
                    logger.info(f"index {i}, {j}: {output_arr[i, j]}")

    assert mx.allclose(output_arr, expected_V_output, atol=1e-2, rtol=1e-2), (
        "Output values do not match expected values"
    )


@pytest.mark.parametrize("head_dim", [32, 64, 128, 256])
@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
def test_fetch_entire_k_vector_for_specific_token_slot(head_dim, dtype) -> None:
    """Test dot product calculation between complete Q and K vectors.

    This test verifies that the paged attention operation correctly computes
    the dot product between query and key vectors and applies appropriate scaling.
    It uses uniform values in each query vector to simplify verification.
    """
    # Configuration
    num_sequences = 2
    num_q_heads = 1  # Single query head per sequence
    cfg_tokens_per_page = 16
    cfg_num_kv_heads = 1
    cfg_head_dim = head_dim
    cfg_max_logical_pages_per_seq_in_pagetable = 2

    # Create 3D queries with shape [num_sequences, num_q_heads, cfg_head_dim]
    py_queries = mx.zeros((num_sequences, num_q_heads, cfg_head_dim), dtype=dtype)
    # Fill each query vector with uniform values
    py_queries[0, 0, :] = 100.0  # Sequence 0, head 0: all elements = 100.0
    py_queries[1, 0, :] = 200.0  # Sequence 1, head 0: all elements = 200.0

    # Create K-cache with specific values
    num_physical_pages = 2
    # Use get_k_cache_shape() for striped format
    k_cache_shape = get_k_cache_shape(num_physical_pages, cfg_num_kv_heads, cfg_head_dim, cfg_tokens_per_page, dtype)
    py_k_cache_pool = mx.zeros(k_cache_shape, dtype=dtype)

    # Set K-vectors with sequential values (only first few elements for simplicity)
    # K-vector for page 0, kv_head 0, token 0
    k_vec = mx.zeros(cfg_head_dim, dtype=dtype)
    for i in range(min(4, cfg_head_dim)):
        k_vec[i] = float(i + 1)  # [1, 2, 3, 4, 0, 0, ...]
    for d in range(cfg_head_dim):
        stripe_idx = d // get_k_cache_stripe_size(dtype)
        offset_in_stripe = d % get_k_cache_stripe_size(dtype)
        py_k_cache_pool[0, 0, stripe_idx, 0, offset_in_stripe] = k_vec[d]

    # K-vector for page 1, kv_head 0, token 0
    k_vec = mx.zeros(cfg_head_dim, dtype=dtype)
    for i in range(min(4, cfg_head_dim)):
        k_vec[i] = float(i + 5)  # [5, 6, 7, 8, 0, 0, ...]
    for d in range(cfg_head_dim):
        stripe_idx = d // get_k_cache_stripe_size(dtype)
        offset_in_stripe = d % get_k_cache_stripe_size(dtype)
        py_k_cache_pool[1, 0, stripe_idx, 0, offset_in_stripe] = k_vec[d]

    # Set up V-cache with distinct values
    v_cache_shape = get_v_cache_shape(num_physical_pages, cfg_num_kv_heads, cfg_head_dim, cfg_tokens_per_page, dtype)
    py_v_cache_pool = mx.zeros(v_cache_shape, dtype=dtype)
    # Create patterns that work for any head_dim
    v_pattern_0 = mx.arange(cfg_head_dim, dtype=dtype) * 10.0 + 10.0  # [10, 20, 30, ...]
    v_pattern_1 = mx.arange(cfg_head_dim, dtype=dtype) * 10.0 + 50.0  # [50, 60, 70, ...]
    py_v_cache_pool[0, 0, :, 0] = v_pattern_0
    py_v_cache_pool[1, 0, :, 0] = v_pattern_1

    # Page table maps: seq 0, block 0 -> page 0; seq 1, block 0 -> page 1
    py_page_table = mx.array(
        [
            [0, 99],  # Sequence 0: logical block 0 -> physical page 0, block 1 -> page 99 (unused)
            [1, 88],  # Sequence 1: logical block 0 -> physical page 1, block 1 -> page 88 (unused)
        ],
        dtype=mx.uint32,
    )
    assert py_page_table.shape == (num_sequences, cfg_max_logical_pages_per_seq_in_pagetable)

    # Set up sequence metadata
    py_sequence_lengths = mx.array([1, 1], dtype=mx.int32)  # Each sequence has 1 token

    # Run paged attention
    output_arr = paged_attention_decode(
        py_queries,
        py_k_cache_pool,
        py_v_cache_pool,
        py_page_table,
        py_sequence_lengths,
    )
    mx.eval(output_arr)

    # --- Calculate expected output (Python reference) ---

    # Scale factor for attention scores
    scale = 1.0 / mx.sqrt(mx.array(float(cfg_head_dim))).item()

    # Calculate dot products
    # For K[0,0,0]: first 4 elements are [1,2,3,4], rest are 0
    # Q[0] dot K[0,0,0] = 100.0 * (1+2+3+4) = 100.0 * 10 = 1000.0
    dot_product_0 = 100.0 * sum(range(1, min(5, cfg_head_dim + 1)))

    # For K[1,0,0]: first 4 elements are [5,6,7,8], rest are 0
    # Q[1] dot K[1,0,0] = 200.0 * (5+6+7+8) = 200.0 * 26 = 5200.0
    dot_product_1 = 200.0 * sum(range(5, min(9, cfg_head_dim + 5)))

    # Expected scores (single token history means 100% attention weight):
    score0 = dot_product_0 * scale
    score1 = dot_product_1 * scale

    # For single history token, softmax prob is always 1.0

    # Create expected V outputs
    # V-aggregation for item 0: V[0,0,0] * prob[0] = V[0,0,0] * 1.0
    expected_V_item0 = py_v_cache_pool[0, 0, :, 0].astype(mx.float32)
    # V-aggregation for item 1: V[1,0,0] * prob[0] = V[1,0,0] * 1.0
    expected_V_item1 = py_v_cache_pool[1, 0, :, 0].astype(mx.float32)

    # Combine and reshape to match kernel output
    expected_V_output = mx.array([[expected_V_item0.astype(dtype)], [expected_V_item1.astype(dtype)]])

    # For 3D queries [num_sequences, num_q_heads, head_dim],
    # output shape is [num_sequences, num_q_heads, head_dim]
    expected_output_shape = (num_sequences, num_q_heads, cfg_head_dim)

    # Log test details
    logger.info(
        f"Test: {test_fetch_entire_k_vector_for_specific_token_slot.__name__} head_dim={head_dim} dtype={dtype}"
    )
    logger.info(f"  Scale factor: {scale}")
    logger.info(f"  Score 0: {score0}, Score 1: {score1}")
    logger.info(f"  Expected output shape: {expected_output_shape}")
    logger.info(f"  Expected V output: {expected_V_output}")
    logger.info(f"  Actual V output: {output_arr}")

    # Verify results
    assert output_arr.shape == expected_output_shape, (
        f"Output shape {output_arr.shape} does not match expected {expected_output_shape}"
    )
    assert output_arr.dtype == dtype, f"Output dtype {output_arr.dtype} does not match {dtype}"
    assert mx.allclose(output_arr, expected_V_output, atol=1e-2, rtol=1e-2), (
        "Output values do not match expected values"
    )
