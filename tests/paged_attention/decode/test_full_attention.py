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


@pytest.mark.parametrize("head_dim", [32, 64, 128, 256])
@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
def test_full_attention_in_one_block(head_dim, dtype) -> None:
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
    mx.clear_cache()
    logger.info(f"Test: {test_full_attention_in_one_block.__name__} (dtype={dtype})")

    # --- Configuration ---
    cfg_tokens_per_page = 16
    cfg_num_kv_heads = 1
    cfg_head_dim = head_dim

    # Current token position is 3, so we'll attend to history positions 0, 1, and 2
    current_position = 3

    logger.info("  Test Configuration:")
    logger.info(f"    head_dim: {cfg_head_dim}, tokens_per_page: {cfg_tokens_per_page}")
    logger.info(f"    current_position: {current_position}")

    # --- Setup test inputs ---
    # 1. Q-vector: Shape [num_q_threads, cfg_head_dim]
    # Create a query vector with cfg_head_dim elements
    q_values = mx.arange(1, cfg_head_dim + 1, dtype=mx.float32) / 10.0
    py_queries = mx.array([q_values.tolist()], dtype=dtype)

    # 2. K-Cache Pool: [NumPhysPages, NumKVHeads, TokensPerPage, HeadDim]
    num_physical_pages = 1
    k_cache_shape = (num_physical_pages, cfg_num_kv_heads, cfg_tokens_per_page, cfg_head_dim)
    py_k_cache_pool = mx.zeros(k_cache_shape, dtype=dtype)

    # Historical K-vectors with different values to produce different scores
    # Create distinct patterns for each position in the KV cache
    k_multipliers = [1.0, 2.0, 0.5]

    # Create a more efficient setup using broadcasting
    for pos, multiplier in enumerate(k_multipliers):
        py_k_cache_pool[0, 0, pos, :] = mx.ones(cfg_head_dim, dtype=dtype) * multiplier

    logger.info("  KV Cache Setup:")
    for pos in range(len(k_multipliers)):
        logger.info(f"    K at position {pos}: {py_k_cache_pool[0, 0, pos, :5]}...")  # Show first 5 elements

    # 3. V-Cache Pool with distinct values for each position
    py_v_cache_pool = mx.zeros(k_cache_shape, dtype=dtype)

    # More efficient V-cache setup using vectorized operations
    # Create base values for each position (10.0, 20.0, 30.0)
    v_base_values = [10.0, 20.0, 30.0]

    # For each position, set ascending values starting from the base
    for pos, base in enumerate(v_base_values):
        py_v_cache_pool[0, 0, pos, :] = mx.arange(base, base + cfg_head_dim, dtype=dtype)

    mx.eval(py_queries)

    # Debug: Check what values are actually stored
    logger.info("  V-Cache Values:")
    for pos in range(len(v_base_values)):
        logger.info(f"    V[{pos}] first few: {py_v_cache_pool[0, 0, pos, :5]}")
        logger.info(f"    V[{pos}] last few: {py_v_cache_pool[0, 0, pos, -5:]}")

    # 4. Page Table: Maps logical block 0 to physical page 0
    py_page_table = mx.array([[0]], dtype=mx.uint32)  # Shape (1, 1)

    # 5. Sequence Lengths: One sequence with current_position tokens
    # The kernel should only attend to positions 0, 1, 2 (before current_position=3)
    py_sequence_lengths = mx.array([current_position], dtype=mx.int32)

    # --- Calculate expected output (Python reference) ---
    # Scale factor for dot product
    py_scale = 1.0 / mx.sqrt(mx.array(float(cfg_head_dim))).item()

    # Calculate scores for each history position
    # Use float32 for reference math to avoid dtype shenanigans
    scores = [mx.sum(py_queries[0] * py_k_cache_pool[0, 0, i, :]) * py_scale for i in range(3)]

    logger.info("  Attention Scores:")
    logger.info(f"    Position 0 score: {scores[0]:.4f}")
    logger.info(f"    Position 1 score: {scores[1]:.4f}")
    logger.info(f"    Position 2 score: {scores[2]:.4f}")

    probs = mx.softmax(mx.array(scores, dtype=mx.float32))
    logger.info("  Softmax Probabilities:")
    for i, prob in enumerate(probs):
        logger.info(f"    Position {i}: {prob:.6f}")

    # Expected weighted sum of V-vectors
    expected_v = mx.zeros(cfg_head_dim, dtype=mx.float32)
    for i, prob in enumerate(probs):
        v_vec = py_v_cache_pool[0, 0, i, :].astype(mx.float32)
        expected_v += v_vec * prob

    expected_v_reshaped = expected_v.astype(dtype).reshape(1, cfg_head_dim)
    mx.eval(expected_v_reshaped)

    old_v_cache_point = [i for i in py_v_cache_pool[0, 0, 1, :].tolist()]
    logger.info(f"  Old v cache point: {old_v_cache_point[:5]}, {old_v_cache_point[-5:]}")

    old_k_cache_point = [i for i in py_k_cache_pool[0, 0, 1, :].tolist()]
    logger.info(f"  Old k cache point: {old_k_cache_point[:5]}, {old_k_cache_point[-5:]}")

    # --- Run paged attention ---
    output_arr = paged_attention(py_queries, py_k_cache_pool, py_v_cache_pool, py_page_table, py_sequence_lengths)
    mx.eval(output_arr)

    logger.info("  Attention Output:")
    logger.info(f"    Expected output shape: {expected_v_reshaped.shape}")
    logger.info(f"    Actual output shape: {output_arr.shape}")
    logger.info(f"    Expected V-aggregation: {expected_v_reshaped}")
    logger.info(f"    Actual output: {output_arr}")

    new_v_cache_point = [i for i in py_v_cache_pool[0, 0, 1, :].tolist()]
    logger.info(f"  New v cache point: {new_v_cache_point[:5]}, {new_v_cache_point[-5:]}")

    new_k_cache_point = [i for i in py_k_cache_pool[0, 0, 1, :].tolist()]
    logger.info(f"  New k cache point: {new_k_cache_point[:5]}, {new_k_cache_point[-5:]}")

    if old_v_cache_point != new_v_cache_point:
        logger.info(f"  Old v cache point: {old_v_cache_point[:5]}, {old_v_cache_point[-5:]}")
        logger.info(f"  New v cache point: {new_v_cache_point[:5]}, {new_v_cache_point[-5:]}")

    if old_k_cache_point != new_k_cache_point:
        logger.info(f"  Old k cache point: {old_k_cache_point[:5]}, {old_k_cache_point[-5:]}")
        logger.info(f"  New k cache point: {new_k_cache_point[:5]}, {new_k_cache_point[-5:]}")

    assert old_v_cache_point == new_v_cache_point and old_k_cache_point == new_k_cache_point, (
        "Cache point should not change"
    )

    # Output should be [num_q_threads, head_dim] with weighted V-vectors
    expected_shape = (1, cfg_head_dim)

    # Verify results
    assert output_arr.shape == expected_shape, (
        f"Output shape {output_arr.shape} does not match expected {expected_shape}"
    )
    assert output_arr.dtype == dtype, f"Output dtype {output_arr.dtype} does not match {dtype}"
    assert mx.allclose(output_arr, expected_v_reshaped, atol=1e-2), (
        f"Value mismatch. Expected: {expected_v_reshaped}, Got: {output_arr}"
    )
