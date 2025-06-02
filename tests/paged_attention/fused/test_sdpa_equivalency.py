# Copyright 2024 The Proxy Company. All Rights Reserved.
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

logger = logging.getLogger(__name__)

tokens_per_page = 64


@pytest.mark.parametrize(
    "batch_size, history_len, num_heads, head_dim, dtype",
    [
        (1, 16, (1, 1), 32, mx.float16),  # Basic case
        (1, 32, (1, 1), 32, mx.float16),  # Longer history
        (1, 16, (4, 1), 32, mx.float16),  # MQA (num_q_heads > num_kv_heads)
        (1, 16, (4, 2), 32, mx.float16),  # GQA (num_q_heads > num_kv_heads, num_kv_heads > 1)
        (1, 16, (1, 1), 64, mx.float16),  # Different head dimension
        (1, 64, (2, 2), 32, mx.float16),  # num q = num kv heads
        (1, 64, (4, 2), 128, mx.float16),  # 128 head dim
        (1, 1024, (32, 16), 128, mx.float16),  # Gemma 3 27b
        (2, 32, (4, 4), 32, mx.float16),  # Batched Example
        (2, 2048, (32, 16), 128, mx.float16),  # Batch of 2, Gemma 3 27b
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

    The test:
    1. Runs MLX SDPA with random inputs as the reference implementation
    2. Converts those same inputs to the format expected by paged_attention in decode mode
    3. Runs paged_attention with is_prefill=False and reshapes the output to match SDPA's output format
    4. Compares the outputs to ensure they're numerically equivalent within tolerance

    This ensures that our implementation matches the standard attention mechanism
    when the inputs are directly comparable.
    """
    mx.random.seed(11)  # signed - jckwind :)
    logger.info(f"Test: {test_pal_decode_vs_sdpa_equivalency.__name__}")

    num_q_heads, num_kv_heads = num_heads
    tokens_per_page = 64  # Standard page size

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

    # Create empty KV cache pools
    pal_k_cache_pool = mx.zeros((num_total_physical_pages, tokens_per_page, num_kv_heads, head_dim), dtype=dtype)
    pal_v_cache_pool = mx.zeros((num_total_physical_pages, tokens_per_page, num_kv_heads, head_dim), dtype=dtype)

    logger.info("  Preparing PAL paged_attention inputs for decode mode:")
    logger.info(f"    PAL queries shape: {pal_queries.shape}")
    logger.info(f"    KV cache shape: {pal_k_cache_pool.shape}")
    logger.info(f"    Logical pages per sequence: {num_logical_pages_per_seq}")
    logger.info(f"    Total physical pages: {num_total_physical_pages}")

    # Populate KV cache from SDPA inputs
    # For each sequence in the batch, we need to populate its history in the KV cache
    for b_idx in range(batch_size):
        # Get keys/values for this sequence and transpose for easier slicing
        keys_to_cache_b = sdpa_keys[b_idx].transpose(1, 0, 2)  # [history_len, num_kv_heads, head_dim]
        values_to_cache_b = sdpa_values[b_idx].transpose(1, 0, 2)  # [history_len, num_kv_heads, head_dim]

        # For each logical page in the sequence
        for l_idx in range(num_logical_pages_per_seq):
            physical_page_idx = b_idx * num_logical_pages_per_seq + l_idx

            # Calculate token range for this page
            token_start_in_seq = l_idx * tokens_per_page
            token_end_in_seq = min((l_idx + 1) * tokens_per_page, history_len)
            tokens_to_copy_count = token_end_in_seq - token_start_in_seq

            # Copy tokens to the KV cache if there are any in this page
            if tokens_to_copy_count > 0:
                pal_k_cache_pool[physical_page_idx, :tokens_to_copy_count, :, :] = keys_to_cache_b[
                    token_start_in_seq:token_end_in_seq, :, :
                ]
                pal_v_cache_pool[physical_page_idx, :tokens_to_copy_count, :, :] = values_to_cache_b[
                    token_start_in_seq:token_end_in_seq, :, :
                ]

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

    # For decode, query_to_seq_map maps each single token to its sequence index
    pal_query_to_seq_map = mx.arange(batch_size, dtype=mx.int32)

    # Query token offset is positioned after the history (history_len + 1)
    pal_query_token_offset = mx.array([history_len + 1] * batch_size, dtype=mx.int32)

    logger.info("  PAL metadata arrays for decode:")
    logger.info(f"    Page table shape: {pal_page_table.shape}")
    logger.info(f"    Sequence lengths: {pal_sequence_lengths}")
    logger.info(f"    Query to sequence map: {pal_query_to_seq_map}")
    logger.info(f"    Query token offset (positioned after history): {pal_query_token_offset}")

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

    pal_output = paged_attention(
        pal_queries,
        pal_k_cache_pool,
        pal_v_cache_pool,
        pal_page_table,
        pal_sequence_lengths,
        pal_query_to_seq_map,
        pal_query_token_offset,
        use_fused_kernel=True,
        # explicitly use decode mode
    )
    mx.eval(pal_output)
    logger.info(f"    PAL output shape: {pal_output.shape}")

    # --- 4. Compare PAL output with SDPA output ---
    # SDPA output is (batch_size, num_q_heads, 1, head_dim)
    # PAL output is (batch_size*num_q_heads, head_dim) given pal_queries shape (batch_size, num_q_heads, head_dim)
    # We need to reshape the SDPA output to match
    sdpa_output_reshaped = sdpa_output.reshape(batch_size * num_q_heads, head_dim)

    logger.info("  Comparing outputs:")
    logger.info(f"    PAL decode output shape: {pal_output.shape}")
    logger.info(f"    Reshaped SDPA output shape: {sdpa_output_reshaped.shape}")

    assert pal_output.shape == sdpa_output_reshaped.shape, (
        f"Shape mismatch: PAL output {pal_output.shape}, SDPA for comparison {sdpa_output_reshaped.shape}"
    )

    # Calculate differences between outputs
    diff = mx.abs(pal_output - sdpa_output_reshaped)
    max_diff = mx.max(diff).item()
    mean_diff = mx.mean(diff).item()
    logger.info(f"    Difference metrics - Max: {max_diff:.6f}, Mean: {mean_diff:.6f}")

    # For FP16, we allow slightly larger differences due to numerical precision & different implementation
    current_atol = 1e-2
    current_rtol = 1e-5
    logger.info(f"    Tolerance values - Absolute: {current_atol}, Relative: {current_rtol}")

    # Assert outputs match within tolerance
    assert mx.allclose(pal_output, sdpa_output_reshaped, atol=current_atol, rtol=current_rtol), (
        f"Numerical mismatch between PAL paged_attention decode and MLX SDPA for params: "
        f"bs={batch_size}, hl={history_len}, nqh={num_q_heads}, nkvh={num_kv_heads}, hd={head_dim}, dt={dtype}. "
        f"Max diff: {max_diff}, Mean diff: {mean_diff}"
    )
