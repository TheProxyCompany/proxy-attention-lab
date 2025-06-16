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
"""Basic smoke test for paged attention prefill functionality."""

import logging

import mlx.core as mx
import pytest

from proxy_attention_lab.pal_core import (
    get_k_cache_shape,
    get_v_cache_shape,
    paged_attention_prefill,
)

logger = logging.getLogger(__name__)


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
def test_paged_attention_prefill_smoke(dtype) -> None:
    """Verify that the paged_attention_prefill function runs without errors on simple inputs.

    This test creates minimal inputs to check that the function executes successfully
    and produces an output with the expected shape and type. The test serves as a basic
    sanity check for the prefill attention mechanism, verifying:

    1. The function can be called with correctly shaped random inputs
    2. The function produces an output without crashing
    3. The output has the expected dimensions based on input parameters
    4. The output contains valid numerical values (no NaN or Inf)
    """
    logger.info(f"Test: {test_paged_attention_prefill_smoke.__name__} (dtype={dtype})")

    # Query tensor parameters
    num_prompt_tokens = 4
    num_q_heads = 8
    head_dim = 32

    # KV cache parameters
    num_total_pages = 2
    tokens_per_page = 16
    num_kv_heads = 2

    # Batch parameters
    num_sequences_in_batch = 1
    max_logical_pages_per_seq_val = 1

    logger.info("  Test Configuration:")
    logger.info(f"    Prompt: {num_prompt_tokens} tokens, {num_q_heads} heads, {head_dim} dimensions")
    logger.info(f"    KV cache: {num_total_pages} pages, {tokens_per_page} tokens per page, {num_kv_heads} KV heads")
    logger.info(f"    Batch: {num_sequences_in_batch} sequences, {max_logical_pages_per_seq_val} blocks per sequence")
    logger.info(f"    Dtype: {dtype}")

    # Create test inputs with random values
    q_prompt = mx.random.normal((num_prompt_tokens, num_q_heads, head_dim)).astype(dtype)
    k_prompt = mx.random.normal((num_prompt_tokens, num_kv_heads, head_dim)).astype(dtype)
    v_prompt = mx.random.normal((num_prompt_tokens, num_kv_heads, head_dim)).astype(dtype)

    v_cache_paged = mx.random.normal(
        get_v_cache_shape(num_total_pages, num_kv_heads, head_dim, tokens_per_page, dtype)
    ).astype(dtype)
    k_cache_paged = mx.random.normal(
        get_k_cache_shape(num_total_pages, num_kv_heads, head_dim, tokens_per_page, dtype)
    ).astype(dtype)

    # Create page table: maps logical blocks to physical pages
    page_table_content = [[0] * max_logical_pages_per_seq_val for _ in range(num_sequences_in_batch)]
    page_table = mx.array(page_table_content, dtype=mx.uint32)

    # Metadata arrays
    context_len_arr = mx.array([tokens_per_page // 2] * num_sequences_in_batch, dtype=mx.int32)

    logger.info("  Input Shapes:")
    logger.info(f"    Q prompt: {q_prompt.shape}")
    logger.info(f"    K prompt: {k_prompt.shape}")
    logger.info(f"    V prompt: {v_prompt.shape}")
    logger.info(f"    K cache: {k_cache_paged.shape}")
    logger.info(f"    V cache: {v_cache_paged.shape}")
    logger.info(f"    Page table: {page_table.shape}")
    logger.info(f"    Context len arr: {context_len_arr.shape}")

    mx.eval(q_prompt)
    mx.eval(k_prompt)
    mx.eval(v_prompt)
    mx.eval(k_cache_paged)
    mx.eval(v_cache_paged)
    mx.eval(page_table)
    mx.eval(context_len_arr)

    try:
        # Run the paged attention prefill operation
        out = paged_attention_prefill(
            q_prompt,
            k_prompt,
            v_prompt,
            k_cache_paged,
            v_cache_paged,
            page_table,
            context_len_arr,
        )
        mx.eval(out)

        # For 3D queries [NumTokens, NumQHeads, HeadDim],
        # output is [NumTokens * NumQHeads, HeadDim] (2D layout)
        total_items = num_prompt_tokens * num_q_heads
        expected_output_shape = (total_items, head_dim)

        logger.info("  Output Verification:")
        logger.info(f"    Expected shape: {expected_output_shape}")
        logger.info(f"    Actual shape: {out.shape}")
        logger.info(f"    Dtype: {out.dtype}")
        logger.info(f"    Contains finite values: {mx.isfinite(out).all()}")

        # Verify output properties
        assert out.shape == expected_output_shape, (
            f"Output shape {out.shape} does not match expected shape {expected_output_shape}"
        )
        assert out.dtype == q_prompt.dtype, (
            f"Output dtype {out.dtype} does not match q_prompt dtype {q_prompt.dtype}"
        )
        assert mx.isfinite(out).all(), "Output attention vectors contain NaN or Inf values"

    except Exception as e:
        logger.error(f"Paged attention prefill smoke test failed: {e}", exc_info=True)
        raise
