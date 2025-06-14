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
"""Equivalency tests for paged attention against MLX scaled_dot_product_attention.

This module verifies that the paged_attention kernel produces numerically
equivalent results to mlx.fast.scaled_dot_product_attention for scenarios
that can be mapped to standard attention (e.g., single sequence, full history,
non-paged).
"""

import logging

import mlx.core as mx
import pytest

from proxy_attention_lab import paged_attention
from proxy_attention_lab.pal_core import (
    get_k_cache_shape,
    get_k_cache_stripe_size,
    get_optimal_page_size,
    get_v_cache_shape,
)

logger = logging.getLogger(__name__)


# TODO: FIX LONG CONTEXT CASES
@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
@pytest.mark.parametrize(
    "batch_size, history_len, num_heads, head_dim",
    [
        (1, 16, (1, 1), 32),  # Basic case
        (1, 32, (1, 1), 32),  # Longer history
        (1, 16, (4, 1), 32),  # MQA (num_q_heads > num_kv_heads)
        (1, 16, (4, 2), 32),  # GQA (num_q_heads > num_kv_heads, num_kv_heads > 1)
        (1, 16, (1, 1), 64),  # Different head dimension
        (1, 64, (2, 2), 32),  # num q = num kv heads
        (1, 64, (4, 2), 128),  # 128 head dim
        (1, 64, (32, 16), 128),  # Gemma 3 27b
        (2, 32, (4, 4), 32),  # Batched Example
        (2, 64, (32, 16), 128),  # Gemma 3 27b
        (3, 128, (32, 16), 128),  # Gemma 3 27b
        (16, 64, (4, 4), 32),  # Batched Example
        # (1, 1024, (32, 16), 128),  # Gemma 3 27b
        # (1, 4096, (32, 16), 128),  # Long history, Gemma 3 27b // should use pass 2
        # (1, 8192, (32, 16), 128),  # Long history, Gemma 3 27b // should use pass 2
    ],
)
def test_pal_decode_vs_sdpa_equivalency(batch_size, history_len, num_heads, head_dim, dtype):
    """Compare PAL paged_attention in decode mode with MLX SDPA kernel.

    This test verifies that our paged_attention implementation in decode mode
    (processing a single new token per sequence) produces numerically equivalent
    results to MLX's scaled_dot_product_attention (SDPA) function across different
    configurations. We test various combinations of:

    - Batch sizes (single item and batched)
    - History lengths (shorter and longer)
    - Head configurations (including MQA and GQA variants)
    - Head dimensions

    This ensures that our implementation matches the standard attention mechanism
    when the inputs are directly comparable.
    """
    mx.clear_cache()
    mx.random.seed(11)
    logger.info(f"Test: {test_pal_decode_vs_sdpa_equivalency.__name__}")

    num_q_heads, num_kv_heads = num_heads
    tokens_per_page = get_optimal_page_size()  # Standard page size

    logger.info("  Test Configuration:")
    logger.info(f"    Batch size: {batch_size}, History length: {history_len}")
    logger.info(f"    Query heads: {num_q_heads}, KV heads: {num_kv_heads}, Head dim: {head_dim}")
    logger.info(f"    Data type: {dtype}")

    # --- 1. Setup Inputs & Run MLX SDPA (Reference) ---
    # In decode mode, we have a single new token (query) attending to the history
    sdpa_q_shape = (batch_size, num_q_heads, 1, head_dim)  # Single token per sequence
    sdpa_kv_shape = (batch_size, num_kv_heads, history_len, head_dim)  # History tokens

    # Create random inputs
    sdpa_queries = mx.random.normal(sdpa_q_shape, dtype=dtype)
    sdpa_keys = mx.random.normal(sdpa_kv_shape, dtype=dtype)
    sdpa_values = mx.random.normal(sdpa_kv_shape, dtype=dtype)

    # For decode, we use a mask of zeros that allows the query to attend to all history tokens
    decode_mask = mx.zeros((1, history_len), dtype=dtype)
    scale = 1.0 / mx.sqrt(float(head_dim))

    logger.info("  Running MLX SDPA (Reference) - Decode Mode:")
    logger.info(f"    Query shape: {sdpa_q_shape}, K/V shape: {sdpa_kv_shape}")
    logger.info(f"    Mask shape: {decode_mask.shape}, allows full attention to history")

    # Run the reference SDPA implementation
    sdpa_output = mx.fast.scaled_dot_product_attention(
        sdpa_queries,
        sdpa_keys,
        sdpa_values,
        scale=scale,
        mask=decode_mask,
    )
    mx.eval(sdpa_output)
    logger.info(f"    SDPA output shape: {sdpa_output.shape}")

    # --- 2. Prepare Inputs for PAL's paged_attention in decode mode ---
    # Calculate KV cache parameters
    num_logical_pages_per_seq = (history_len + tokens_per_page - 1) // tokens_per_page
    num_total_physical_pages = batch_size * num_logical_pages_per_seq

    # Reshape the query for PAL format: [batch_size, num_q_heads, head_dim]
    # Each sequence has a single query token
    pal_queries = sdpa_queries.reshape(batch_size, num_q_heads, head_dim)

    # Create empty KV cache pools with new layout:
    # K-cache: [pages, kv_heads, head_dim // stripe_size, tokens, stripe_size] (striped format)
    # V-cache: [pages, kv_heads, head_dim, tokens] (dimension order changed)
    k_cache_shape = get_k_cache_shape(num_total_physical_pages, num_kv_heads, head_dim, tokens_per_page, dtype)
    v_cache_shape = get_v_cache_shape(num_total_physical_pages, num_kv_heads, head_dim, tokens_per_page, dtype)
    pal_k_cache_pool = mx.zeros(k_cache_shape, dtype=dtype)
    pal_v_cache_pool = mx.zeros(v_cache_shape, dtype=dtype)

    logger.info("  Preparing PAL paged_attention inputs for decode mode:")
    logger.info(f"    PAL queries shape: {pal_queries.shape}")
    logger.info(f"    K cache shape: {pal_k_cache_pool.shape}")
    logger.info(f"    V cache shape: {pal_v_cache_pool.shape}")
    logger.info(f"    Logical pages per sequence: {num_logical_pages_per_seq}")
    logger.info(f"    Total physical pages: {num_total_physical_pages}")

    # Populate KV cache from SDPA inputs
    for b_idx in range(batch_size):
        # sdpa_keys[b_idx] shape is [num_kv_heads, history_len, head_dim]
        keys_to_cache_b = sdpa_keys[b_idx]
        values_to_cache_b = sdpa_values[b_idx]

        for l_idx in range(num_logical_pages_per_seq):
            physical_page_idx = b_idx * num_logical_pages_per_seq + l_idx
            token_start_in_seq = l_idx * tokens_per_page
            token_end_in_seq = min((l_idx + 1) * tokens_per_page, history_len)
            tokens_to_copy_count = token_end_in_seq - token_start_in_seq

            if tokens_to_copy_count > 0:
                # For each KV head, copy its contiguous block of tokens for this page
                for kv_h_idx in range(num_kv_heads):
                    # Slice the tokens for the current head and page
                    keys_for_head_slice = keys_to_cache_b[kv_h_idx, token_start_in_seq:token_end_in_seq, :]
                    values_for_head_slice = values_to_cache_b[kv_h_idx, token_start_in_seq:token_end_in_seq, :]

                    # Write K-cache using striped format with vectorized operations
                    stripe_size = get_k_cache_stripe_size(dtype)
                    num_stripes = head_dim // stripe_size

                    # Reshape keys to match striped format: [tokens, num_stripes, stripe_size]
                    keys_reshaped = keys_for_head_slice.reshape(tokens_to_copy_count, num_stripes, stripe_size)

                    # Transpose to [num_stripes, tokens, stripe_size] to match cache layout
                    keys_striped = mx.transpose(keys_reshaped, (1, 0, 2))

                    # Write all stripes at once
                    pal_k_cache_pool[physical_page_idx, kv_h_idx, :, :tokens_to_copy_count, :] = keys_striped

                    # Write V-cache with new dimension order: [page, kv_head, head_dim, token]
                    pal_v_cache_pool[physical_page_idx, kv_h_idx, :, :tokens_to_copy_count] = values_for_head_slice.T

    # Create page table mapping
    pal_page_table_list = []
    for b_idx in range(batch_size):
        sequence_physical_page_indices = [
            b_idx * num_logical_pages_per_seq + l_idx for l_idx in range(num_logical_pages_per_seq)
        ]
        pal_page_table_list.append(sequence_physical_page_indices)
    pal_page_table = mx.array(pal_page_table_list, dtype=mx.uint32)

    # Set sequence length for each batch item to the history length
    pal_sequence_lengths = mx.array([history_len] * batch_size, dtype=mx.int32)

    logger.info("  PAL metadata arrays for decode:")
    logger.info(f"    Page table shape: {pal_page_table.shape}")
    logger.info(f"    Sequence lengths: {pal_sequence_lengths}")

    # --- 3. Apply MLX Best Practices for Array Preparation ---
    logger.info("  Applying MLX best practices for array preparation:")

    # Ensure pal_queries is fully evaluated and contiguous
    logger.info("    Preparing pal_queries...")
    pal_queries = mx.contiguous(pal_queries)
    mx.eval(pal_queries)

    # Ensure pal_k_cache_pool is fully evaluated and contiguous
    logger.info("    Preparing pal_k_cache_pool...")
    pal_k_cache_pool = mx.contiguous(pal_k_cache_pool)
    mx.eval(pal_k_cache_pool)

    # Ensure pal_v_cache_pool is fully evaluated and contiguous
    logger.info("    Preparing pal_v_cache_pool...")
    pal_v_cache_pool = mx.contiguous(pal_v_cache_pool)
    mx.eval(pal_v_cache_pool)

    # Validate all inputs using the validation function
    logger.info("  Validating all inputs before passing to Metal kernel:")
    # --- 4. Run PAL paged_attention in decode mode ---
    logger.info("  Running PAL paged_attention (decode mode):")

    pal_output = paged_attention(pal_queries, pal_k_cache_pool, pal_v_cache_pool, pal_page_table, pal_sequence_lengths)
    mx.eval(pal_output)
    logger.info(f"    PAL output shape: {pal_output.shape}")

    # --- 4. Compare PAL output with SDPA output ---
    sdpa_output_reshaped = sdpa_output.reshape(batch_size, num_q_heads, head_dim)
    pal_output_reshaped = pal_output.reshape(batch_size, num_q_heads, head_dim)

    logger.info("  Comparing outputs:")
    logger.info(f"    PAL decode output shape: {pal_output.shape}")
    logger.info(f"    Reshaped SDPA output shape: {sdpa_output_reshaped.shape}")

    assert pal_output_reshaped.shape == sdpa_output_reshaped.shape, (
        f"Shape mismatch: PAL output {pal_output_reshaped.shape}, SDPA for comparison {sdpa_output_reshaped.shape}"
    )

    # Calculate differences between outputs
    diff = mx.abs(pal_output_reshaped - sdpa_output_reshaped)
    max_diff = mx.max(diff).item()
    mean_diff = mx.mean(diff).item()
    logger.info(f"    Difference metrics - Max: {max_diff:.6f}, Mean: {mean_diff:.6f}")

    # For FP16/BF16, we allow slightly larger differences due to numerical precision & different implementation
    current_atol = 1e-2
    current_rtol = 1e-4
    logger.info(f"    Tolerance values - Absolute: {current_atol}, Relative: {current_rtol}")

    assert mx.allclose(pal_output_reshaped, sdpa_output_reshaped, atol=current_atol, rtol=current_rtol), (
        f"Numerical mismatch between PAL paged_attention decode and MLX SDPA: "
        f"Max diff: {max_diff}, Mean diff: {mean_diff}"
    )
