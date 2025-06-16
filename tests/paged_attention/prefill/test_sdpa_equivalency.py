import logging

import mlx.core as mx
import numpy as np
import pytest

from proxy_attention_lab import paged_attention_prefill
from proxy_attention_lab.pal_core import (
    get_k_cache_shape,
    get_k_cache_stripe_size,
    get_optimal_page_size,
    get_v_cache_shape,
)

logger = logging.getLogger(__name__)


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
@pytest.mark.parametrize(
    "batch_size, history_len, prompt_len, num_heads, head_dim",
    [
        (1, 0, 16, (4, 4), 64),  # Initial prefill, no history
        (1, 128, 16, (4, 4), 64),  # Incremental prefill with history
        (2, 64, 32, (8, 2), 128),  # Batched incremental prefill with GQA
        (1, 256, 1, (32, 8), 128),  # Single-token incremental prefill (decode-like)
        (1, 0, 128, (32, 8), 128),  # Longer initial prefill
    ],
)
def test_pal_prefill_vs_sdpa_equivalency(batch_size, history_len, prompt_len, num_heads, head_dim, dtype):
    """
    Compare PAL paged_attention_prefill with MLX SDPA kernel.

    This test verifies that our tiled prefill kernel produces numerically equivalent
    results to MLX's SDPA function. We model a prefill scenario by concatenating
    a "history" context with a new "prompt" and running SDPA over the whole sequence.
    The test ensures our kernel, which handles history and prompt separately,
    arrives at the same result.
    """
    logger.info(f"Test: {test_pal_prefill_vs_sdpa_equivalency.__name__} for dtype={dtype}")

    num_q_heads, num_kv_heads = num_heads
    tokens_per_page = get_optimal_page_size()
    total_seq_len = history_len + prompt_len

    logger.info("  Test Configuration:")
    logger.info(f"    Batch: {batch_size}, History: {history_len}, Prompt: {prompt_len}")
    logger.info(f"    Heads (Q/KV): {num_q_heads}/{num_kv_heads}, Dim: {head_dim}")

    # --- 1. Setup Inputs & Run MLX SDPA (Reference) ---
    # Create one large sequence for SDPA that includes both history and prompt.
    sdpa_q_shape = (batch_size, num_q_heads, total_seq_len, head_dim)
    sdpa_k_shape = (batch_size, num_kv_heads, total_seq_len, head_dim)
    sdpa_v_shape = (batch_size, num_kv_heads, total_seq_len, head_dim)

    queries = mx.random.normal(sdpa_q_shape, dtype=dtype)
    keys = mx.random.normal(sdpa_k_shape, dtype=dtype)
    values = mx.random.normal(sdpa_v_shape, dtype=dtype)

    # Create a causal mask for the entire sequence length.
    mask = mx.triu(mx.full((total_seq_len, total_seq_len), -np.inf, dtype=dtype), k=1)
    scale = 1.0 / mx.sqrt(float(head_dim))

    logger.info("  Running MLX SDPA (Reference) - Prefill Mode:")
    logger.info(f"    Q/K/V shape: {queries.shape}")

    sdpa_output = mx.fast.scaled_dot_product_attention(queries, keys, values, scale=scale, mask=mask)
    mx.eval(sdpa_output)

    # We only care about the output corresponding to the prompt tokens.
    sdpa_output_for_prompt = sdpa_output[:, :, -prompt_len:, :]
    logger.info(f"    SDPA output shape (for prompt): {sdpa_output_for_prompt.shape}")

    # --- 2. Prepare Inputs for PAL's paged_attention_prefill ---
    # Separate the Q/K/V tensors into history and prompt parts.
    q_for_prefill = queries[:, :, -prompt_len:, :].transpose(0, 2, 1, 3)
    k_for_prefill = keys[:, :, -prompt_len:, :].transpose(0, 2, 1, 3)
    v_for_prefill = values[:, :, -prompt_len:, :].transpose(0, 2, 1, 3)

    q_prompt = q_for_prefill.reshape(batch_size * prompt_len, num_q_heads, head_dim)
    k_prompt = k_for_prefill.reshape(batch_size * prompt_len, num_kv_heads, head_dim)
    v_prompt = v_for_prefill.reshape(batch_size * prompt_len, num_kv_heads, head_dim)

    k_history = keys[:, :, :history_len, :]
    v_history = values[:, :, :history_len, :]

    # Populate the paged KV cache with the history part.
    num_logical_pages = (history_len + tokens_per_page - 1) // tokens_per_page
    num_total_pages = batch_size * num_logical_pages

    k_cache_shape = get_k_cache_shape(num_total_pages, num_kv_heads, head_dim, tokens_per_page, dtype)
    v_cache_shape = get_v_cache_shape(num_total_pages, num_kv_heads, head_dim, tokens_per_page, dtype)
    k_cache_paged = mx.zeros(k_cache_shape, dtype=dtype)
    v_cache_paged = mx.zeros(v_cache_shape, dtype=dtype)

    # This cache filling logic is identical to the decode test.
    for b in range(batch_size):
        for page_idx in range(num_logical_pages):
            p_page_idx = b * num_logical_pages + page_idx
            start = page_idx * tokens_per_page
            end = min(start + tokens_per_page, history_len)
            count = end - start
            if count <= 0:
                continue

            # Populate K cache (swizzled)
            k_slice = (
                k_history[b, :, start:end, :]
                .transpose(0, 2, 1)
                .reshape(num_kv_heads, count, -1, get_k_cache_stripe_size(dtype))
                .transpose(0, 2, 1, 3)
            )
            k_cache_paged[p_page_idx, :, :, :count, :] = k_slice

            # Populate V cache (strided)
            v_slice = v_history[b, :, start:end, :].transpose(0, 2, 1)
            v_cache_paged[p_page_idx, :, :, :count] = v_slice

    page_table = mx.arange(num_total_pages).reshape(batch_size, num_logical_pages)
    context_lens_arr = mx.array([history_len] * batch_size, dtype=mx.int32)

    logger.info("  Running PAL paged_attention_prefill:")
    logger.info(f"    Q prompt shape: {q_prompt.shape}")

    q_prompt = mx.contiguous(q_prompt)
    k_prompt = mx.contiguous(k_prompt)
    v_prompt = mx.contiguous(v_prompt)
    k_cache_paged = mx.contiguous(k_cache_paged)
    v_cache_paged = mx.contiguous(v_cache_paged)
    page_table = mx.contiguous(page_table)
    context_lens_arr = mx.contiguous(context_lens_arr)

    mx.eval(q_prompt)
    mx.eval(k_prompt)
    mx.eval(v_prompt)
    mx.eval(k_cache_paged)
    mx.eval(v_cache_paged)
    mx.eval(page_table)
    mx.eval(context_lens_arr)

    # --- 3. Run PAL paged_attention_prefill ---
    pal_output = paged_attention_prefill(
        q_prompt, k_prompt, v_prompt, k_cache_paged, v_cache_paged, page_table, context_lens_arr
    )
    mx.eval(pal_output)

    # --- 4. Compare Outputs ---
    # Reshape both outputs to be [batch, q_heads, prompt_len, head_dim] for comparison
    pal_output_reshaped = pal_output.reshape(batch_size, prompt_len, num_q_heads, head_dim).transpose(0, 2, 1, 3)

    logger.info("  Comparing outputs:")
    logger.info(f"    PAL output shape (reshaped): {pal_output_reshaped.shape}")
    logger.info(f"    SDPA output shape (for prompt): {sdpa_output_for_prompt.shape}")

    assert pal_output_reshaped.shape == sdpa_output_for_prompt.shape

    mean_diff = mx.mean(mx.abs(pal_output_reshaped - sdpa_output_for_prompt)).item()
    logger.info(f"    Mean absolute difference: {mean_diff:.6f}")
    max_diff = mx.max(mx.abs(pal_output_reshaped - sdpa_output_for_prompt)).item()
    logger.info(f"    Max absolute difference: {max_diff:.6f}")

    if not mx.allclose(pal_output_reshaped, sdpa_output_for_prompt, atol=1e-2, rtol=1e-2):
        for idx in range(pal_output_reshaped.shape[-1]):
            pal_vec = pal_output_reshaped[0, 0, idx]
            sdpa_vec = sdpa_output_for_prompt[0, 0, idx]
            if not mx.allclose(pal_vec, sdpa_vec, atol=1e-2, rtol=1e-2):
                logger.error(f"    PAL vec (idx={idx}): {pal_vec.tolist()[:10]}")
                logger.error(f"    SDPA vec (idx={idx}): {sdpa_vec.tolist()[:10]}")
                logger.error(
                    f"Vector mismatch at last axis index {idx}: max abs diff " +
                    f"{mx.max(mx.abs(pal_vec - sdpa_vec)).item():.6f}"
                )
                break
            else:
                logger.info(f"    PAL vec (idx={idx}): {pal_vec.tolist()[:10]}")
                logger.info(f"    SDPA vec (idx={idx}): {sdpa_vec.tolist()[:10]}")
                logger.info(f"Vector {idx} matches within tolerance: {mx.allclose(pal_vec, sdpa_vec, atol=1e-2, rtol=1e-2)}")


    assert mx.allclose(pal_output_reshaped, sdpa_output_for_prompt, atol=1e-2, rtol=1e-2)
