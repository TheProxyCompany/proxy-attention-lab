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
import mlx.nn as nn
import pytest

from proxy_attention_lab import paged_attention

logger = logging.getLogger(__name__)

tokens_per_page = 64


@pytest.mark.parametrize(
    "batch_size, seq_len, num_heads, head_dim, dtype",
    [
        (1, 16, (1, 1), 32, mx.float16),  # Original case
        (1, 32, (1, 1), 32, mx.float16),  # Longer sequence
        (1, 16, (4, 1), 32, mx.float16),  # MQA (num_q_heads > num_kv_heads)
        (1, 16, (4, 2), 32, mx.float16),  # GQA (num_q_heads > num_kv_heads, num_kv_heads > 1)
        (1, 16, (1, 1), 64, mx.float16),  # Different head dimension
        (1, 32, (2, 2), 32, mx.float16),  # Different number of heads
        (2, 32, (4, 4), 32, mx.float16),  # Batched Example
    ],
)
def test_pal_vs_sdpa_equivalency_mha(batch_size, seq_len, num_heads, head_dim, dtype):
    """
    Compare PAL paged_attention with MLX SDPA for an MHA/MQA/GQA case.

    This test sets up a standard MHA scenario, computes the attention result
    using MLX's SDPA, then transforms the inputs to fit PAL's paged_attention
    kernel, runs PAL's kernel, and asserts numerical equivalency.
    """
    num_q_heads, num_kv_heads = num_heads

    # --- 1. Setup Inputs & Run MLX SDPA (Reference) ---
    sdpa_q_shape = (batch_size, num_q_heads, seq_len, head_dim)
    sdpa_kv_shape = (batch_size, num_kv_heads, seq_len, head_dim)

    sdpa_queries = mx.random.normal(sdpa_q_shape, dtype=dtype)
    sdpa_keys = mx.random.normal(sdpa_kv_shape, dtype=dtype)
    sdpa_values = mx.random.normal(sdpa_kv_shape, dtype=dtype)

    # Create a causal mask for each item in the batch if batch_size > 1
    # However, MLX SDPA expects a single mask that applies to all batch items if it's 2D (L, S) or 3D (H, L, S)
    # or a 4D mask (B, H, L, S). For typical causal masking, a 2D mask is fine and broadcasted.
    causal_mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len).astype(dtype)
    scale = 1.0 / mx.sqrt(float(head_dim))

    logger.info(
        f"Equivalency Test (params: bs={batch_size}, sl={seq_len}, nqh={num_q_heads}, nkvh={num_kv_heads}, hd={head_dim}, dt={dtype}): "
        f"Calling mx.fast.scaled_dot_product_attention."
    )
    sdpa_output = mx.fast.scaled_dot_product_attention(
        sdpa_queries,
        sdpa_keys,
        sdpa_values,
        scale=scale,
        mask=causal_mask,
    )
    mx.eval(sdpa_output)
    logger.info(f"Equivalency Test: SDPA output shape={sdpa_output.shape}")

    # --- 2. Prepare Inputs for PAL's paged_attention ---
    # PAL expects queries as (TotalTokens, NumQHeads, HeadDim)
    # SDPA queries are (batch_size, num_q_heads, seq_len, head_dim)
    # Reshape to (batch_size * seq_len, num_q_heads, head_dim)
    # Order: [s0_t0, s0_t1, ..., s0_t_sl-1, s1_t0, ..., s_b-1_t_sl-1]
    pal_queries = sdpa_queries.transpose(0, 2, 1, 3).reshape(batch_size * seq_len, num_q_heads, head_dim)

    num_logical_pages_per_seq = (seq_len + tokens_per_page - 1) // tokens_per_page
    num_total_physical_pages = batch_size * num_logical_pages_per_seq

    pal_k_cache_pool = mx.zeros((num_total_physical_pages, tokens_per_page, num_kv_heads, head_dim), dtype=dtype)
    pal_v_cache_pool = mx.zeros((num_total_physical_pages, tokens_per_page, num_kv_heads, head_dim), dtype=dtype)

    for b_idx in range(batch_size):
        # Keys/Values for current sequence in batch: (num_kv_heads, seq_len, head_dim)
        # Transpose to (seq_len, num_kv_heads, head_dim) for easier slicing by token
        keys_to_cache_b = sdpa_keys[b_idx].transpose(1, 0, 2)
        values_to_cache_b = sdpa_values[b_idx].transpose(1, 0, 2)

        for l_idx in range(num_logical_pages_per_seq):
            physical_page_idx = b_idx * num_logical_pages_per_seq + l_idx

            token_start_in_seq = l_idx * tokens_per_page
            token_end_in_seq = min((l_idx + 1) * tokens_per_page, seq_len)
            tokens_to_copy_count = token_end_in_seq - token_start_in_seq

            if tokens_to_copy_count > 0:
                pal_k_cache_pool[physical_page_idx, :tokens_to_copy_count, :, :] = keys_to_cache_b[
                    token_start_in_seq:token_end_in_seq, :, :
                ]
                pal_v_cache_pool[physical_page_idx, :tokens_to_copy_count, :, :] = values_to_cache_b[
                    token_start_in_seq:token_end_in_seq, :, :
                ]

    pal_page_table_list = []
    for b_idx in range(batch_size):
        sequence_physical_page_indices = [
            b_idx * num_logical_pages_per_seq + l_idx for l_idx in range(num_logical_pages_per_seq)
        ]
        pal_page_table_list.append(sequence_physical_page_indices)
    pal_page_table = mx.array(pal_page_table_list, dtype=mx.uint32)

    pal_sequence_lengths = mx.array([seq_len] * batch_size, dtype=mx.int32)

    # query_to_seq_map: maps each token in pal_queries to its sequence index
    # pal_queries has tokens ordered as [seq0_tokens, seq1_tokens, ...]
    # Corrected: Should be [0 (SL times), 1 (SL times), ...]
    pal_query_to_seq_map = mx.repeat(mx.arange(batch_size, dtype=mx.int32), repeats=seq_len)

    # query_token_offset: for causal attention, 1-indexed position within the sequence
    # For [s0_t0, s0_t1, ..., s1_t0, s1_t1, ...], offsets are [1, 2, ..., SL, 1, 2, ..., SL, ...]
    pal_query_token_offset = mx.tile(mx.arange(1, seq_len + 1, dtype=mx.int32), batch_size)

    logger.info(f"Equivalency Test: PAL Query shape={pal_queries.shape}")
    logger.info(f"Equivalency Test: PAL K-Cache shape={pal_k_cache_pool.shape}")
    logger.info(f"Equivalency Test: PAL Page Table shape={pal_page_table.shape}, content: {pal_page_table}")
    logger.info(
        f"Equivalency Test: PAL Seq Lengths shape={pal_sequence_lengths.shape}, content: {pal_sequence_lengths}"
    )
    logger.info(
        f"Equivalency Test: PAL Q_to_Seq Map shape={pal_query_to_seq_map.shape}"
    )  # , content: {pal_query_to_seq_map}")
    logger.info(
        f"Equivalency Test: PAL Q_Token Offset shape={pal_query_token_offset.shape}"
    )  # , content: {pal_query_token_offset}")

    # --- 3. Run PAL paged_attention ---
    logger.info("Equivalency Test: Calling PAL paged_attention.")
    pal_output = paged_attention(
        pal_queries,
        pal_k_cache_pool,
        pal_v_cache_pool,
        pal_page_table,
        pal_sequence_lengths,
        pal_query_to_seq_map,
        pal_query_token_offset,
    )
    mx.eval(pal_output)
    logger.info(f"Equivalency Test: PAL output shape={pal_output.shape}")

    # --- 4. Compare PAL output with SDPA output ---
    # SDPA output is (batch_size, num_q_heads, seq_len, head_dim)
    # PAL output (from C++ op, given pal_queries shape (B*SL, NQ, HD)) is (B*SL*NQ, HD)
    # We need to reshape sdpa_output to match this.
    # (B, NQ, SL, HD) -> transpose(0, 2, 1, 3) -> (B, SL, NQ, HD)
    # -> reshape(-1, HD) -> (B*SL*NQ, HD)
    sdpa_output_reshaped = sdpa_output.transpose(0, 2, 1, 3).reshape(-1, head_dim)

    logger.info(f"Equivalency Test: PAL output shape={pal_output.shape}")  # Added this log to confirm PAL output shape
    logger.info(f"Equivalency Test: SDPA output for comparison shape={sdpa_output_reshaped.shape}")

    assert pal_output.shape == sdpa_output_reshaped.shape, (
        f"Shape mismatch: PAL output {pal_output.shape}, SDPA for comparison {sdpa_output_reshaped.shape}"
    )

    diff = mx.abs(pal_output - sdpa_output_reshaped)
    max_diff = mx.max(diff).item()
    mean_diff = mx.mean(diff).item()
    logger.info(f"PAL vs SDPA Differences - Max: {max_diff}, Mean: {mean_diff}")

    # For FP16, we allow slightly larger differences due to numerical precision & different implementation
    current_atol = 1e-2
    current_rtol = 1e-5

    if not mx.allclose(pal_output, sdpa_output_reshaped, atol=current_atol, rtol=current_rtol):
        logger.error(
            f"PAL vs SDPA (params: {batch_size, seq_len, num_heads, head_dim, dtype}): Outputs do not match closely enough."
        )
        logger.error(f"Overall Max absolute difference: {max_diff}")
        logger.error(f"Overall Mean absolute difference: {mean_diff}")

    assert mx.allclose(pal_output, sdpa_output_reshaped, atol=current_atol, rtol=current_rtol), (
        f"Numerical mismatch between PAL paged_attention and MLX SDPA for params: "
        f"bs={batch_size}, sl={seq_len}, nqh={num_q_heads}, nkvh={num_kv_heads}, hd={head_dim}, dt={dtype}. "
        f"Max diff: {max_diff}, Mean diff: {mean_diff}"
    )
    logger.info(
        f"test_pal_vs_sdpa_equivalency PASSED for params: bs={batch_size}, sl={seq_len}, nqh={num_q_heads}, nkvh={num_kv_heads}, hd={head_dim}, dt={dtype}."
    )
