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
import math

import mlx.core as mx
import mlx.nn as nn

from proxy_attention_lab import paged_attention

logger = logging.getLogger(__name__)

# Note on MLX SDPA vs PAL numerical differences:
# Despite kernel hardening with improved online softmax and proper clamp values,
# we still see significant numerical differences between the two implementations.
# This is expected due to:
# 1. Different accumulation strategies (PAL does online o = o*alpha + p*v, MLX likely uses batch)
# 2. FP16 precision limitations and order of operations differences
# 3. Different softmax implementations (PAL uses online max-tracking, MLX likely different algorithm)
# These differences aren't problematic for inference tasks.


def test_pal_vs_sdpa_equivalency_mha():
    """
    Compare PAL paged_attention with MLX SDPA for an MHA case.

    This test sets up a standard MHA scenario, computes the attention result
    using MLX's SDPA, then transforms the inputs to fit PAL's paged_attention
    kernel, runs PAL's kernel, and asserts numerical equivalency.

    Note: Due to implementation differences (notably in softmax for float16),
    tolerances are relaxed. Results should be functionally equivalent for inference.
    """
    # --- Configuration ---
    batch_size = 1
    seq_len = 16
    num_q_heads = 2
    num_kv_heads = 2
    head_dim = 32
    dtype = mx.float16

    tokens_per_page = seq_len

    mx.random.seed(42)

    # --- 1. Setup Inputs & Run MLX SDPA (Reference) ---
    sdpa_q_shape = (batch_size, num_q_heads, seq_len, head_dim)
    sdpa_kv_shape = (batch_size, num_kv_heads, seq_len, head_dim)

    sdpa_queries = mx.random.normal(sdpa_q_shape, dtype=dtype)
    sdpa_keys = mx.random.normal(sdpa_kv_shape, dtype=dtype)
    sdpa_values = mx.random.normal(sdpa_kv_shape, dtype=dtype)

    causal_mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len).astype(dtype)
    scale = 1.0 / math.sqrt(head_dim)

    logger.info("Equivalency Test: Calling mx.fast.scaled_dot_product_attention.")
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
    pal_queries = sdpa_queries.squeeze(axis=0).transpose(1, 0, 2)

    num_physical_pages = 1
    pal_k_cache_pool = mx.zeros((num_physical_pages, tokens_per_page, num_kv_heads, head_dim), dtype=dtype)
    pal_v_cache_pool = mx.zeros((num_physical_pages, tokens_per_page, num_kv_heads, head_dim), dtype=dtype)

    keys_to_cache = sdpa_keys.squeeze(axis=0).transpose(1, 0, 2)
    values_to_cache = sdpa_values.squeeze(axis=0).transpose(1, 0, 2)

    pal_k_cache_pool[0, :seq_len, :, :] = keys_to_cache
    pal_v_cache_pool[0, :seq_len, :, :] = values_to_cache

    pal_page_table = mx.array([[0]], dtype=mx.uint32)
    pal_sequence_lengths = mx.array([seq_len], dtype=mx.int32)
    pal_query_to_seq_map = mx.zeros((seq_len,), dtype=mx.int32)
    pal_query_token_offset = mx.arange(1, seq_len + 1, dtype=mx.int32)

    logger.info(f"Equivalency Test: PAL Query shape={pal_queries.shape}")
    logger.info(f"Equivalency Test: PAL K-Cache shape={pal_k_cache_pool.shape}")

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
    sdpa_output_reshaped = sdpa_output.squeeze(axis=0).transpose(1, 0, 2).reshape(-1, head_dim)
    logger.info(f"Equivalency Test: SDPA output for comparison shape={sdpa_output_reshaped.shape}")

    assert pal_output.shape == sdpa_output_reshaped.shape, (
        f"Shape mismatch: PAL output {pal_output.shape}, SDPA for comparison {sdpa_output_reshaped.shape}"
    )

    atol = 1e-2
    rtol = 1e-2

    diff = mx.abs(pal_output - sdpa_output_reshaped)
    max_diff = mx.max(diff).item()
    mean_diff = mx.mean(diff).item()
    logger.info(f"PAL vs SDPA Differences - Max: {max_diff}, Mean: {mean_diff}")

    if not mx.allclose(pal_output, sdpa_output_reshaped, atol=atol, rtol=rtol):
        logger.error("PAL vs SDPA: Outputs do not match closely enough.")
        logger.error(f"Overall Max absolute difference: {max_diff}")
        logger.error(f"Overall Mean absolute difference: {mean_diff}")

    # For FP16, we allow larger differences due to numerical precision issues
    assert mx.allclose(pal_output, sdpa_output_reshaped, atol=atol, rtol=rtol), (
        "Numerical mismatch between PAL paged_attention and MLX SDPA for MHA case."
    )
    logger.info("test_pal_vs_sdpa_equivalency_mha PASSED.")
