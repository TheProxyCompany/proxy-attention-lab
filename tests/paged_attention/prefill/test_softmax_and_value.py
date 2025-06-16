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
"""Tests for softmax and value aggregation in paged attention prefill.

This module contains tests focused on the softmax computation and value vector
aggregation aspects of the paged attention prefill operation.
"""

import logging

import mlx.core as mx
import pytest

from proxy_attention_lab.pal_core import (
    get_k_cache_shape,
    get_k_cache_stripe_size,
    get_v_cache_shape,
    paged_attention_prefill,
)

logger = logging.getLogger(__name__)


@pytest.mark.parametrize("head_dim", [32, 64, 128])
@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
def test_v_aggregation_prefill_single_query_no_history(head_dim, dtype) -> None:
    """Test V-aggregation for prefill with a single query and no history.

    Tests the prefill operation for one query attending only to itself (causal mask),
    verifying proper softmax probability calculation and value aggregation.

    This test demonstrates the simplest prefill case:
    1. Single query token attending only to itself
    2. Softmax probability should be 1.0
    3. Output should equal the value vector
    """
    logger.info(f"Test: {test_v_aggregation_prefill_single_query_no_history.__name__} (dtype={dtype})")

    # Configuration
    num_seqs = 1
    num_q_heads = 1
    num_kv_heads = 1
    tokens_per_page = 16
    prompt_len = 1
    history_len = 0

    # --- Setup test inputs ---
    # Queries for prefill: [prompt_len * num_seqs, num_q_heads, head_dim]
    py_queries = mx.ones((prompt_len, num_q_heads, head_dim), dtype=dtype)

    # Keys for prefill: [prompt_len * num_seqs, num_kv_heads, head_dim]
    py_keys = mx.ones((prompt_len, num_kv_heads, head_dim), dtype=dtype)
    py_keys[0, 0, :4] = mx.array([1.0, 1.0, 0.0, 0.0], dtype=dtype)

    # Values for prefill: [prompt_len * num_seqs, num_kv_heads, head_dim]
    py_values = mx.zeros((prompt_len, num_kv_heads, head_dim), dtype=dtype)
    py_values[0, 0, :4] = mx.array([10.0, 20.0, 30.0, 40.0], dtype=dtype)

    # Empty KV cache (no history)
    k_cache_shape = get_k_cache_shape(0, num_kv_heads, head_dim, tokens_per_page, dtype)
    v_cache_shape = get_v_cache_shape(0, num_kv_heads, head_dim, tokens_per_page, dtype)
    py_k_cache_pool = mx.zeros(k_cache_shape, dtype=dtype)
    py_v_cache_pool = mx.zeros(v_cache_shape, dtype=dtype)

    # Empty page table
    py_page_table = mx.zeros((num_seqs, 0), dtype=mx.uint32)

    # Context lengths: [num_seqs] - zero for no history
    py_context_lens = mx.array([history_len], dtype=mx.int32)

    logger.info("  Test Configuration:")
    logger.info(f"    num_seqs: {num_seqs}, num_q_heads: {num_q_heads}, num_kv_heads: {num_kv_heads}")
    logger.info(f"    head_dim: {head_dim}, prompt_len: {prompt_len}, history_len: {history_len}")

    # Expected output: Since query attends only to itself, output = V
    expected_output = mx.zeros((prompt_len * num_q_heads, head_dim), dtype=dtype)
    expected_output[0, :4] = mx.array([10.0, 20.0, 30.0, 40.0], dtype=dtype)

    mx.eval(py_queries, py_keys, py_values, py_k_cache_pool, py_v_cache_pool, py_page_table, py_context_lens)

    # --- Run paged attention prefill ---
    output_arr = paged_attention_prefill(
        py_queries, py_keys, py_values, py_k_cache_pool, py_v_cache_pool, py_page_table, py_context_lens
    )
    mx.eval(output_arr)

    logger.info("  Attention Output:")
    logger.info(f"    Output shape: {output_arr.shape}")
    logger.info(f"    Expected shape: {expected_output.shape}")
    logger.info(f"    Output values (first 8): {output_arr[0, :8]}")
    logger.info(f"    Expected values (first 8): {expected_output[0, :8]}")

    # Verify output shape
    assert output_arr.shape == expected_output.shape, (
        f"Output shape {output_arr.shape} does not match expected {expected_output.shape}"
    )
    assert output_arr.dtype == dtype, f"Output dtype {output_arr.dtype} does not match {dtype}"

    # Verify output values
    assert mx.allclose(output_arr, expected_output, atol=1e-2), (
        f"Value mismatch. Expected: {expected_output[0, :8]}, Got: {output_arr[0, :8]}"
    )


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
def test_prefill_causal_mask_two_tokens(dtype) -> None:
    """Test prefill with two tokens and causal masking.

    This test verifies that the causal mask is correctly applied during prefill:
    - Token 0 can only attend to itself
    - Token 1 can attend to both token 0 and itself
    """
    logger.info(f"Test: {test_prefill_causal_mask_two_tokens.__name__} (dtype={dtype})")

    # Configuration
    num_seqs = 1
    num_q_heads = 1
    num_kv_heads = 1
    head_dim = 32
    tokens_per_page = 16
    prompt_len = 2
    history_len = 0

    # Setup inputs - 2 query tokens
    py_queries = mx.ones((prompt_len, num_q_heads, head_dim), dtype=dtype)

    # Keys with different patterns
    py_keys = mx.zeros((prompt_len, num_kv_heads, head_dim), dtype=dtype)
    py_keys[0, 0, :2] = mx.array([1.0, 1.0], dtype=dtype)  # Token 0: [1,1,0,0,...]
    py_keys[1, 0, 1:3] = mx.array([1.0, 1.0], dtype=dtype)  # Token 1: [0,1,1,0,...]

    # Values with distinct patterns
    py_values = mx.zeros((prompt_len, num_kv_heads, head_dim), dtype=dtype)
    py_values[0, 0, :4] = mx.array([1.0, 0.0, 0.0, 0.0], dtype=dtype)
    py_values[1, 0, :4] = mx.array([0.0, 1.0, 0.0, 0.0], dtype=dtype)

    # Empty KV cache (no history)
    py_k_cache_pool = mx.zeros((0, num_kv_heads, head_dim // get_k_cache_stripe_size(dtype),
                                tokens_per_page, get_k_cache_stripe_size(dtype)), dtype=dtype)
    py_v_cache_pool = mx.zeros((0, num_kv_heads, head_dim, tokens_per_page), dtype=dtype)
    py_page_table = mx.zeros((num_seqs, 0), dtype=mx.uint32)
    py_context_lens = mx.array([history_len], dtype=mx.int32)

    # For token 0: Can only see itself
    # Score = Q[0] · K[0] * scale = 2.0 * scale
    # Softmax of single value = 1.0
    # Output[0] = V[0] * 1.0 = [1,0,0,0,...]

    # For token 1: Can see token 0 and itself
    # Score[0] = Q[1] · K[0] * scale = 2.0 * scale
    # Score[1] = Q[1] · K[1] * scale = 2.0 * scale
    # Both scores are equal, so softmax = [0.5, 0.5]
    # Output[1] = V[0] * 0.5 + V[1] * 0.5 = [0.5, 0.5, 0, 0, ...]

    expected_output = mx.zeros((prompt_len * num_q_heads, head_dim), dtype=dtype)
    expected_output[0, :4] = mx.array([1.0, 0.0, 0.0, 0.0], dtype=dtype)
    expected_output[1, :4] = mx.array([0.5, 0.5, 0.0, 0.0], dtype=dtype)

    mx.eval(py_queries, py_keys, py_values, py_k_cache_pool, py_v_cache_pool, py_page_table, py_context_lens)

    # Run paged attention prefill
    output_arr = paged_attention_prefill(
        py_queries, py_keys, py_values, py_k_cache_pool, py_v_cache_pool, py_page_table, py_context_lens
    )
    mx.eval(output_arr)

    logger.info(f"    Output token 0 (first 4): {output_arr[0, :4]}")
    logger.info(f"    Expected token 0 (first 4): {expected_output[0, :4]}")
    logger.info(f"    Output token 1 (first 4): {output_arr[1, :4]}")
    logger.info(f"    Expected token 1 (first 4): {expected_output[1, :4]}")

    assert mx.allclose(output_arr, expected_output, atol=1e-2), (
        f"Causal mask test failed. Token 0: Expected {expected_output[0, :4]}, Got {output_arr[0, :4]}. "
        f"Token 1: Expected {expected_output[1, :4]}, Got {output_arr[1, :4]}"
    )


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
def test_prefill_with_history(dtype) -> None:
    """Test prefill with existing history in KV cache.

    This test verifies that prefill correctly attends to both:
    1. Tokens in the KV cache (history)
    2. New prompt tokens (with causal mask)
    """
    logger.info(f"Test: {test_prefill_with_history.__name__} (dtype={dtype})")

    # Configuration
    num_q_heads = 1
    num_kv_heads = 1
    head_dim = 32
    tokens_per_page = 16
    prompt_len = 2
    history_len = 2

    # Setup prompt queries/keys/values (2 new tokens)
    py_queries = mx.ones((prompt_len, num_q_heads, head_dim), dtype=dtype)

    py_keys = mx.zeros((prompt_len, num_kv_heads, head_dim), dtype=dtype)
    py_keys[0, 0, 2:4] = mx.array([1.0, 1.0], dtype=dtype)  # New token 0: [0,0,1,1,...]
    py_keys[1, 0, 3:5] = mx.array([1.0, 1.0], dtype=dtype)  # New token 1: [0,0,0,1,1,0,...]

    py_values = mx.zeros((prompt_len, num_kv_heads, head_dim), dtype=dtype)
    py_values[0, 0, :4] = mx.array([0.0, 0.0, 1.0, 0.0], dtype=dtype)
    py_values[1, 0, :4] = mx.array([0.0, 0.0, 0.0, 1.0], dtype=dtype)

    # Setup KV cache with history (2 cached tokens)
    num_pages = 1
    k_cache_shape = get_k_cache_shape(num_pages, num_kv_heads, head_dim, tokens_per_page, dtype)
    py_k_cache_pool = mx.zeros(k_cache_shape, dtype=dtype)

    # History token 0: [1,1,0,0,...]
    k_hist_0 = mx.zeros(head_dim, dtype=dtype)
    k_hist_0[:2] = 1.0
    for d in range(head_dim):
        stripe_idx = d // get_k_cache_stripe_size(dtype)
        offset_in_stripe = d % get_k_cache_stripe_size(dtype)
        py_k_cache_pool[0, 0, stripe_idx, 0, offset_in_stripe] = k_hist_0[d]

    # History token 1: [0,1,1,0,...]
    k_hist_1 = mx.zeros(head_dim, dtype=dtype)
    k_hist_1[1:3] = 1.0
    for d in range(head_dim):
        stripe_idx = d // get_k_cache_stripe_size(dtype)
        offset_in_stripe = d % get_k_cache_stripe_size(dtype)
        py_k_cache_pool[0, 0, stripe_idx, 1, offset_in_stripe] = k_hist_1[d]

    # V cache with history
    v_cache_shape = get_v_cache_shape(num_pages, num_kv_heads, head_dim, tokens_per_page, dtype)
    py_v_cache_pool = mx.zeros(v_cache_shape, dtype=dtype)
    py_v_cache_pool[0, 0, :4, 0] = mx.array([1.0, 0.0, 0.0, 0.0], dtype=dtype)  # History V[0]
    py_v_cache_pool[0, 0, :4, 1] = mx.array([0.0, 1.0, 0.0, 0.0], dtype=dtype)  # History V[1]

    py_page_table = mx.array([[0]], dtype=mx.uint32)
    py_context_lens = mx.array([history_len], dtype=mx.int32)

    # Calculate expected output
    # Prompt token 0 can see: history[0,1] and prompt[0]
    # Prompt token 1 can see: history[0,1] and prompt[0,1]

    # For prompt token 0:
    # Scores: [Q·K_hist0, Q·K_hist1, Q·K_prompt0] = [2, 2, 2] * scale
    # Softmax: [1/3, 1/3, 1/3]
    # Output = V_hist0/3 + V_hist1/3 + V_prompt0/3 = [1/3, 1/3, 1/3, 0, ...]

    # For prompt token 1:
    # Scores: [Q·K_hist0, Q·K_hist1, Q·K_prompt0, Q·K_prompt1] = [2, 2, 2, 2] * scale
    # Softmax: [1/4, 1/4, 1/4, 1/4]
    # Output = (V_hist0 + V_hist1 + V_prompt0 + V_prompt1)/4 = [1/4, 1/4, 1/4, 1/4, ...]

    expected_output = mx.zeros((prompt_len * num_q_heads, head_dim), dtype=dtype)
    expected_output[0, :4] = mx.array([1/3, 1/3, 1/3, 0.0], dtype=dtype)
    expected_output[1, :4] = mx.array([0.25, 0.25, 0.25, 0.25], dtype=dtype)

    mx.eval(py_queries, py_keys, py_values, py_k_cache_pool, py_v_cache_pool, py_page_table, py_context_lens)

    # Run paged attention prefill
    output_arr = paged_attention_prefill(
        py_queries, py_keys, py_values, py_k_cache_pool, py_v_cache_pool, py_page_table, py_context_lens
    )
    mx.eval(output_arr)

    logger.info("  Results:")
    logger.info(f"    Output token 0 (first 4): {output_arr[0, :4]}")
    logger.info(f"    Expected token 0 (first 4): {expected_output[0, :4]}")
    logger.info(f"    Output token 1 (first 4): {output_arr[1, :4]}")
    logger.info(f"    Expected token 1 (first 4): {expected_output[1, :4]}")

    assert mx.allclose(output_arr, expected_output, atol=1e-2), (
        f"Prefill with history test failed. Token 0: Expected {expected_output[0, :4]}, Got {output_arr[0, :4]}. "
        f"Token 1: Expected {expected_output[1, :4]}, Got {output_arr[1, :4]}"
    )
