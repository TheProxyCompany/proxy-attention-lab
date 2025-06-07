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
"""Tests for full attention computation.

This module contains tests that verify the complete paged attention operation,
including matrix multiplication, softmax, and value aggregation.
"""

import logging

import mlx.core as mx
import pytest

from proxy_attention_lab import paged_attention

logger = logging.getLogger(__name__)


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
def test_full_attention_in_one_block(dtype) -> None:
    """Test full attention computation in a single block for both float16 and bfloat16.

    Tests the full attention computation (max score, softmax, and V-aggregation)
    for a single item with multiple history tokens.

    This test verifies that the kernel:
    1. Identifies the current Q token's logical position
    2. Loops through historical token positions
    3. Computes dot product scores with each historical K-vector
    4. Finds the maximum score and calculates softmax probabilities
    5. Fetches V-vectors and applies softmax weights
    6. Performs threadgroup reduction to produce the final weighted V-vector

    All token positions in this test are within the same logical block 0.
    """
    logger.info(f"Test: {test_full_attention_in_one_block.__name__} (dtype={dtype})")

    # --- Configuration ---
    num_q_threads = 1  # Just one query thread for this test
    cfg_tokens_per_page = 64
    cfg_num_kv_heads = 1
    cfg_head_dim = 4

    # Current token position is 3, so we'll attend to history positions 0, 1, and 2
    current_position = 3

    logger.info("  Test Configuration:")
    logger.info(f"    head_dim: {cfg_head_dim}, tokens_per_page: {cfg_tokens_per_page}")
    logger.info(f"    current_position: {current_position}")

    # --- Setup test inputs ---
    # 1. Q-vector: Shape [num_q_threads, cfg_head_dim]
    py_queries = mx.array([[1.0, 2.0, 3.0, 4.0]], dtype=dtype)

    # 2. K-Cache Pool: [NumPhysPages, TokensPerPage, NumKVHeads, HeadDim]
    num_physical_pages = 1
    k_cache_shape = (num_physical_pages, cfg_tokens_per_page, cfg_num_kv_heads, cfg_head_dim)
    py_k_cache_pool = mx.zeros(k_cache_shape, dtype=dtype)

    # Historical K-vectors with different values to produce different scores
    # K-vector at position 0
    py_k_cache_pool[0, 0, 0, :] = mx.array([1.0, 1.0, 1.0, 1.0], dtype=dtype)
    # K-vector at position 1
    py_k_cache_pool[0, 1, 0, :] = mx.array([2.0, 2.0, 2.0, 2.0], dtype=dtype)
    # K-vector at position 2
    py_k_cache_pool[0, 2, 0, :] = mx.array([0.5, 0.5, 0.5, 0.5], dtype=dtype)

    logger.info("  KV Cache Setup:")
    logger.info(f"    K at position 0: {py_k_cache_pool[0, 0, 0, :]}")
    logger.info(f"    K at position 1: {py_k_cache_pool[0, 1, 0, :]}")
    logger.info(f"    K at position 2: {py_k_cache_pool[0, 2, 0, :]}")

    # 3. V-Cache Pool with distinct values for each position
    py_v_cache_pool = mx.zeros(k_cache_shape, dtype=dtype)
    py_v_cache_pool[0, 0, 0, :] = mx.array([10.0, 11.0, 12.0, 13.0], dtype=dtype)  # Position 0
    py_v_cache_pool[0, 1, 0, :] = mx.array([20.0, 21.0, 22.0, 23.0], dtype=dtype)  # Position 1
    py_v_cache_pool[0, 2, 0, :] = mx.array([30.0, 31.0, 32.0, 33.0], dtype=dtype)  # Position 2

    # 4. Page Table: Maps logical block 0 to physical page 0
    py_page_table = mx.array([[0]], dtype=mx.uint32)  # Shape (1, 1)

    # 5. Sequence Lengths: One sequence with enough tokens
    py_sequence_lengths = mx.array([cfg_tokens_per_page], dtype=mx.int32)

    # 6. Query to Sequence Map: Maps our single query to sequence 0
    py_query_to_seq_map = mx.zeros(num_q_threads, dtype=mx.int32)

    # 7. Query Token Offset: Current position
    py_query_token_offset = mx.array([current_position], dtype=mx.int32)

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
        py_v_cache_pool,
        py_page_table,
        py_sequence_lengths,
        py_query_to_seq_map,
        py_query_token_offset,
        use_fused_kernel=True,
    )
    mx.eval(output_arr)

    # --- Calculate expected output (Python reference) ---
    # Scale factor for dot product
    py_scale = 1.0 / mx.sqrt(mx.array(float(cfg_head_dim))).item()

    # Calculate scores for each history position
    # Use float32 for reference math to avoid dtype shenanigans
    q_vec = py_queries[0].astype(mx.float32)
    k0 = py_k_cache_pool[0, 0, 0, :].astype(mx.float32)
    k1 = py_k_cache_pool[0, 1, 0, :].astype(mx.float32)
    k2 = py_k_cache_pool[0, 2, 0, :].astype(mx.float32)
    score0 = (mx.sum(q_vec * k0) * py_scale).item()
    score1 = (mx.sum(q_vec * k1) * py_scale).item()
    score2 = (mx.sum(q_vec * k2) * py_scale).item()
    scores = [score0, score1, score2]

    logger.info("  Attention Scores:")
    logger.info(f"    Position 0 score: {score0:.4f}")
    logger.info(f"    Position 1 score: {score1:.4f}")
    logger.info(f"    Position 2 score: {score2:.4f}")

    # Find maximum score
    max_score = max(scores)  # Should be 10.0 from position 1
    logger.info(f"    Maximum score: {max_score:.4f} (at position 1)")

    # Calculate softmax probabilities
    def softmax(scores, max_score):
        exp_scores = [mx.exp(s - max_score).item() for s in scores]
        sum_exp = sum(exp_scores)
        return [es / sum_exp for es in exp_scores]

    probs = softmax(scores, max_score)
    logger.info("  Softmax Probabilities:")
    for i, prob in enumerate(probs):
        logger.info(f"    Position {i}: {prob:.6f}")

    # Expected weighted sum of V-vectors
    expected_v = mx.zeros(cfg_head_dim, dtype=mx.float32)
    for i, prob in enumerate(probs):
        v_vec = py_v_cache_pool[0, i, 0, :].astype(mx.float32)
        expected_v += v_vec * prob

    expected_v_reshaped = expected_v.astype(dtype).reshape(1, cfg_head_dim)

    logger.info("  Attention Output:")
    logger.info(f"    Expected output shape: {expected_v_reshaped.shape}")
    logger.info(f"    Actual output shape: {output_arr.shape}")
    logger.info(f"    Expected V-aggregation: {expected_v_reshaped}")
    logger.info(f"    Actual output: {output_arr}")

    # Output should be [num_q_threads, head_dim] with weighted V-vectors
    expected_shape = (num_q_threads, cfg_head_dim)

    # Verify results
    assert output_arr.shape == expected_shape, (
        f"Output shape {output_arr.shape} does not match expected {expected_shape}"
    )
    assert output_arr.dtype == dtype, f"Output dtype {output_arr.dtype} does not match {dtype}"
    assert mx.allclose(output_arr, expected_v_reshaped, atol=1e-2), (
        f"Value mismatch. Expected: {expected_v_reshaped}, Got: {output_arr}"
    )
