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
"""Tests for softmax and value aggregation in paged attention decode.

This module contains tests focused on the softmax computation and value vector
aggregation aspects of the paged attention decode operation.
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
def test_v_aggregation_decode_single_sequence(head_dim, dtype) -> None:
    """Test V-aggregation for decode with a single sequence.

    Tests the decode operation for one sequence with one query head attending to
    cached KV tokens, verifying proper softmax probability calculation and value
    aggregation.

    This test demonstrates the full decode attention computation flow:
    1. Query-Key dot product to get raw attention scores
    2. Scale the scores by 1/sqrt(head_dim)
    3. Calculate softmax probabilities
    4. Weight the value vectors with the softmax probabilities
    5. Return the aggregated value vectors
    """
    logger.info(f"Test: {test_v_aggregation_decode_single_sequence.__name__} (dtype={dtype})")

    # Configuration
    num_seqs = 1
    num_q_heads = 1
    num_kv_heads = 1
    tokens_per_page = 16
    context_len = 1  # Number of tokens in KV cache for this sequence

    # --- Setup test inputs ---
    # Query: [num_seqs, num_q_heads, head_dim] - shape for decode
    # Using simple values for easy manual calculation
    py_queries = mx.ones((num_seqs, num_q_heads, head_dim), dtype=dtype)
    logger.info(f"  Query shape: {py_queries.shape}")

    # K-cache: [NumPhysPages, NumKVHeads, HeadDim // QK_VECTOR_WIDTH, TokensPerPage, QK_VECTOR_WIDTH]
    # Initialize with zeros, then set specific values using striped format
    num_pages = 1
    k_cache_shape = get_k_cache_shape(num_pages, num_kv_heads, head_dim, tokens_per_page, dtype)
    py_k_cache_pool = mx.zeros(k_cache_shape, dtype=dtype)

    # Set K for token 0: first 4 dims = 1.0, rest = 0.0
    # Create a K vector and write it in striped fashion
    k_vec = mx.zeros(head_dim, dtype=dtype)
    k_vec[:4] = 1.0
    for d in range(head_dim):
        stripe_idx = d // get_k_cache_stripe_size(dtype)
        offset_in_stripe = d % get_k_cache_stripe_size(dtype)
        py_k_cache_pool[0, 0, stripe_idx, 0, offset_in_stripe] = k_vec[d]

    # V-cache: [num_pages, num_kv_heads, head_dim, tokens_per_page]
    v_cache_shape = get_v_cache_shape(num_pages, num_kv_heads, head_dim, tokens_per_page, dtype)
    py_v_cache_pool = mx.zeros(v_cache_shape, dtype=dtype)
    # Set V for token 0: distinctive pattern for verification
    py_v_cache_pool[0, 0, :4, 0] = mx.array([10.0, 20.0, 30.0, 40.0], dtype=dtype)

    # Page table: [num_seqs, max_logical_pages_per_seq]
    # Sequence 0 uses physical page 0
    py_page_table = mx.array([[0]], dtype=mx.uint32)

    # Context lengths: [num_seqs] - how many tokens are in KV cache
    py_context_lens = mx.array([context_len], dtype=mx.int32)

    logger.info("  Test Configuration:")
    logger.info(f"    num_seqs: {num_seqs}, num_q_heads: {num_q_heads}, num_kv_heads: {num_kv_heads}")
    logger.info(f"    head_dim: {head_dim}, tokens_per_page: {tokens_per_page}")
    logger.info(f"    context_length: {context_len}")

    # Expected output: V * softmax_prob
    # V[0,0,0] = [10, 20, 30, 40, 0, 0, ..., 0]
    # Since softmax_prob = 1.0, output = V[0,0,0]
    expected_output = mx.zeros((num_seqs, num_q_heads, head_dim), dtype=dtype)
    expected_output[0, 0, :4] = mx.array([10.0, 20.0, 30.0, 40.0], dtype=dtype)

    mx.eval(py_queries)
    mx.eval(py_k_cache_pool)
    mx.eval(py_v_cache_pool)
    mx.eval(py_page_table)
    mx.eval(py_context_lens)

    # --- Run paged attention decode ---
    output_arr = paged_attention_decode(py_queries, py_k_cache_pool, py_v_cache_pool, py_page_table, py_context_lens)
    mx.eval(output_arr)

    logger.info("  Attention Output:")
    logger.info(f"    Output shape: {output_arr.shape}")
    logger.info(f"    Expected shape: {expected_output.shape}")
    logger.info(f"    Output values (first 8): {output_arr[0, 0, :8]}")
    logger.info(f"    Expected values (first 8): {expected_output[0, 0, :8]}")

    # Verify output shape
    assert output_arr.shape == expected_output.shape, (
        f"Output shape {output_arr.shape} does not match expected {expected_output.shape}"
    )
    assert output_arr.dtype == dtype, f"Output dtype {output_arr.dtype} does not match {dtype}"

    # Verify output values
    assert mx.allclose(output_arr, expected_output, atol=1e-2), (
        f"Value mismatch. Expected: {expected_output[0, 0, :8]}, Got: {output_arr[0, 0, :8]}"
    )


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
def test_decode_multi_token_context(dtype) -> None:
    """Test decode with multiple tokens in context.

    This test verifies that the softmax computation works correctly when
    attending to multiple cached tokens.
    """
    logger.info(f"Test: {test_decode_multi_token_context.__name__} (dtype={dtype})")

    # Configuration
    num_seqs = 1
    num_q_heads = 1
    num_kv_heads = 1
    head_dim = 32
    tokens_per_page = 16
    context_len = 3  # Three tokens in KV cache

    # Setup inputs
    py_queries = mx.ones((num_seqs, num_q_heads, head_dim), dtype=dtype)

    # K-cache with 3 different keys using striped format
    num_pages = 1
    k_cache_shape = get_k_cache_shape(num_pages, num_kv_heads, head_dim, tokens_per_page, dtype)
    py_k_cache_pool = mx.zeros(k_cache_shape, dtype=dtype)

    # Token 0: [1,1,0,0,...]
    k_vec_0 = mx.zeros(head_dim, dtype=dtype)
    k_vec_0[:2] = 1.0
    for d in range(head_dim):
        stripe_idx = d // get_k_cache_stripe_size(dtype)
        offset_in_stripe = d % get_k_cache_stripe_size(dtype)
        py_k_cache_pool[0, 0, stripe_idx, 0, offset_in_stripe] = k_vec_0[d]

    # Token 1: [0,1,1,0,...]
    k_vec_1 = mx.zeros(head_dim, dtype=dtype)
    k_vec_1[1:3] = 1.0
    for d in range(head_dim):
        stripe_idx = d // get_k_cache_stripe_size(dtype)
        offset_in_stripe = d % get_k_cache_stripe_size(dtype)
        py_k_cache_pool[0, 0, stripe_idx, 1, offset_in_stripe] = k_vec_1[d]

    # Token 2: [0,0,1,1,...]
    k_vec_2 = mx.zeros(head_dim, dtype=dtype)
    k_vec_2[2:4] = 1.0
    for d in range(head_dim):
        stripe_idx = d // get_k_cache_stripe_size(dtype)
        offset_in_stripe = d % get_k_cache_stripe_size(dtype)
        py_k_cache_pool[0, 0, stripe_idx, 2, offset_in_stripe] = k_vec_2[d]

    # V-cache with distinct values - shape is [num_pages, num_kv_heads, head_dim, tokens_per_page]
    v_cache_shape = get_v_cache_shape(num_pages, num_kv_heads, head_dim, tokens_per_page, dtype)
    py_v_cache_pool = mx.zeros(v_cache_shape, dtype=dtype)
    py_v_cache_pool[0, 0, :4, 0] = mx.array([1.0, 0.0, 0.0, 0.0], dtype=dtype)
    py_v_cache_pool[0, 0, :4, 1] = mx.array([0.0, 1.0, 0.0, 0.0], dtype=dtype)
    py_v_cache_pool[0, 0, :4, 2] = mx.array([0.0, 0.0, 1.0, 0.0], dtype=dtype)

    py_page_table = mx.array([[0]], dtype=mx.uint32)
    py_context_lens = mx.array([context_len], dtype=mx.int32)

    # Calculate expected output
    scale = 1.0 / mx.sqrt(mx.array(float(head_dim))).item()

    # Helper to reconstruct K-vector from striped cache
    def get_k_vector_from_striped_cache(cache, token_idx):
        vec = mx.zeros(head_dim, dtype=dtype)
        for d in range(head_dim):
            stripe_idx = d // get_k_cache_stripe_size(dtype)
            offset_in_stripe = d % get_k_cache_stripe_size(dtype)
            vec[d] = cache[0, 0, stripe_idx, token_idx, offset_in_stripe]
        return vec

    # Compute scores: Q·K for each cached token
    scores = []
    for i in range(context_len):
        k_vec = get_k_vector_from_striped_cache(py_k_cache_pool, i)
        score = mx.sum(py_queries[0, 0] * k_vec) * scale
        scores.append(score)

    scores = mx.array(scores, dtype=mx.float32)

    # Softmax: all scores are equal, so each gets 1/3 probability
    probs = mx.softmax(scores)

    # Expected output: weighted sum of V vectors
    expected_output = mx.zeros((num_seqs, num_q_heads, head_dim), dtype=dtype)
    for i in range(3):
        v_vec = mx.zeros(head_dim, dtype=dtype)
        v_vec[:4] = py_v_cache_pool[0, 0, :4, i]
        expected_output[0, 0] += v_vec * probs[i].item()

    mx.eval(py_queries, py_k_cache_pool, py_v_cache_pool, py_page_table, py_context_lens)
    # Run paged attention
    output_arr = paged_attention_decode(
        py_queries,
        py_k_cache_pool,
        py_v_cache_pool,
        py_page_table,
        py_context_lens,
    )
    mx.eval(output_arr)

    logger.info(f"    Softmax probs: {probs}")
    logger.info(f"    Output values (first 4): {output_arr[0, 0, :4]}")
    logger.info(f"    Expected values (first 4): {expected_output[0, 0, :4]}")

    assert mx.allclose(output_arr, expected_output, atol=1e-2), (
        f"Multi-token context test failed. Expected: {expected_output[0, 0, :4]}, Got: {output_arr[0, 0, :4]}"
    )
