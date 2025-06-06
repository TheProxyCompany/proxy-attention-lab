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
"""Tests for softmax and value aggregation in paged attention.

This module contains tests focused on the softmax computation and value vector
aggregation aspects of the paged attention operation.
"""

import logging

import mlx.core as mx
import pytest

from proxy_attention_lab import paged_attention

logger = logging.getLogger(__name__)


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
def test_v_aggregation_local_accumulation(dtype) -> None:
    """Test V-aggregation with local accumulation for both float16 and bfloat16.

    Tests the V-aggregation for a single item with a single history token,
    verifying proper softmax probability calculation using global max and sum_exp.
    The kernel should compute and output correctly weighted V vectors.

    This test demonstrates the full attention computation flow:
    1. Query-Key dot product to get raw attention scores
    2. Scale the scores by 1/sqrt(head_dim)
    3. Calculate softmax probabilities
    4. Weight the value vectors with the softmax probabilities
    5. Return the aggregated value vectors
    """
    logger.info(f"Test: {test_v_aggregation_local_accumulation.__name__} (dtype={dtype})")

    num_items = 1
    cfg_head_dim = 4
    cfg_num_kv_heads = 1
    cfg_tokens_per_page = 64

    # --- Setup test inputs ---
    # Q: [1, head_dim]
    py_queries = mx.array([[1.0, 1.0, 1.0, 1.0]], dtype=dtype)
    logger.info(f"  Query shape: {py_queries.shape}, values: {py_queries}")

    # K-cache: K for hist_idx = 0
    py_k_cache_pool = mx.zeros((1, cfg_tokens_per_page, cfg_num_kv_heads, cfg_head_dim), dtype=dtype)
    py_k_cache_pool[0, 0, 0, :] = mx.array([1.0, 1.0, 0.0, 0.0], dtype=dtype)  # Raw dot = 2.0

    # V-cache: V for hist_idx = 0
    py_v_cache_pool = mx.zeros((1, cfg_tokens_per_page, cfg_num_kv_heads, cfg_head_dim), dtype=dtype)
    py_v_cache_pool[0, 0, 0, :] = mx.array([10.0, 20.0, 30.0, 40.0], dtype=dtype)

    # Page table, sequence lengths, etc. for 1 item, 1 history token at pos 0
    py_page_table = mx.array([[0]], dtype=mx.uint32)
    py_sequence_lengths = mx.array([1], dtype=mx.int32)  # Item has 1 token (the current Q)
    py_query_to_seq_map = mx.array([0], dtype=mx.int32)
    py_query_token_offset = mx.array([1], dtype=mx.int32)  # Effective history length = 1 (looks at hist_idx 0)

    logger.info("  Test Configuration:")
    logger.info(f"    head_dim: {cfg_head_dim}, tokens_per_page: {cfg_tokens_per_page}")
    logger.info(
        f"    sequence_length: {py_sequence_lengths.item()}, query_token_offset: {py_query_token_offset.item()}"
    )

    # --- Calculate expected output (Python reference) ---
    py_scale = 1.0 / mx.sqrt(mx.array(float(cfg_head_dim))).item()  # 0.5

    # Note: Q is now pre-scaled in the kernel, so the dot product directly gives the scaled score
    # K-vector at position 0 is [1.0, 1.0, 0.0, 0.0]
    # Q-vector is [1.0, 1.0, 1.0, 1.0]
    # Dot product is 1.0*1.0 + 1.0*1.0 + 1.0*0.0 + 1.0*0.0 = 2.0
    # When pre-scaled: 2.0 * 0.5 = 1.0
    scaled_score_hist0 = 2.0 * py_scale  # 2.0 * 0.5 = 1.0

    # Global max score is the max score found = 1.0
    test_final_max_score = scaled_score_hist0

    # Global sum_exp_score is exp(score - max_score) = exp(0) = 1.0
    test_final_sum_exp_score = mx.exp(scaled_score_hist0 - test_final_max_score).item()  # exp(0) = 1.0

    # Softmax probability calculation
    softmax_prob_hist0 = (
        mx.exp(scaled_score_hist0 - test_final_max_score).item() / test_final_sum_exp_score
    )  # 1.0 / 1.0 = 1.0

    # Now the kernel performs a full threadgroup reduction, so we expect the complete V vector
    # Since we have only 1 item and 1 history token, thread 0 processes that token
    # The weight is softmax_prob_hist0 = 1.0 (since there's only one token in history)
    # So the final aggregated V will be V * 1.0 = [10.0, 20.0, 30.0, 40.0]
    expected_v_output = py_v_cache_pool[0, 0, 0, :] * softmax_prob_hist0  # [10.0, 20.0, 30.0, 40.0] * 1.0
    expected_v_output_reshaped = expected_v_output.reshape(num_items, cfg_head_dim)

    mx.eval(py_queries)
    mx.eval(py_k_cache_pool)
    mx.eval(py_v_cache_pool)
    mx.eval(py_page_table)
    mx.eval(py_sequence_lengths)
    mx.eval(py_query_to_seq_map)
    mx.eval(py_query_token_offset)

    # --- Run paged attention ---
    output_arr = paged_attention(
        py_queries,
        py_k_cache_pool,
        py_v_cache_pool,  # V-cache is now an input
        py_page_table,
        py_sequence_lengths,
        py_query_to_seq_map,
        py_query_token_offset,
        use_fused_kernel=True,
    )
    mx.eval(output_arr)

    logger.info("  Attention Output:")
    logger.info(f"    Output shape: {output_arr.shape}")
    logger.info(f"    Expected output shape: {(num_items, cfg_head_dim)}")
    logger.info(f"    Expected V output: {expected_v_output_reshaped}")
    logger.info(f"    Computed softmax probability: {softmax_prob_hist0}")

    # Assert the shape declared by C++ output_shapes
    expected_shape = (num_items, cfg_head_dim)
    assert output_arr.shape == expected_shape, (
        f"Output shape {output_arr.shape} does not match expected {expected_shape}"
    )
    assert output_arr.dtype == dtype, f"Output dtype {output_arr.dtype} does not match {dtype}"
    # Value assertion - fully reduced V vector
    assert mx.allclose(output_arr, expected_v_output_reshaped, atol=1e-2), (
        f"Value mismatch. Expected (full V): {expected_v_output_reshaped}, Got: {output_arr}"
    )
