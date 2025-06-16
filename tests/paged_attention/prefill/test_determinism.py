import logging

import mlx.core as mx
import numpy as np
import pytest

from proxy_attention_lab.pal_core import (
    get_k_cache_shape,
    get_optimal_page_size,
    get_v_cache_shape,
    paged_attention_prefill,
)

logger = logging.getLogger(__name__)

@pytest.mark.parametrize("prompt_len", [64, 128, 256, 512, 1024])
@pytest.mark.parametrize("history_len", [64, 128, 256, 512, 1024])
@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
def test_paged_attention_determinism_prefill(prompt_len, history_len, dtype) -> None:
    """Test that paged_attention prefill output is deterministic for identical inputs.

    This test configures a moderately complex prefill scenario with multiple sequences
    and varying lengths, calls paged_attention twice with the same inputs,
    and asserts that the resulting output arrays are identical.
    """
    logger.info(f"Test: {test_paged_attention_determinism_prefill.__name__} (dtype={dtype})")
    num_q_heads = 32
    num_kv_heads = 16
    head_dim = 128

    tokens_per_page = get_optimal_page_size()
    logger.info(f"  Calculated tokens_per_page (D_s): {tokens_per_page}")

    # 1. Queries: Shape [TotalNumQueryTokensInBatch, NumQHeads, HeadDim]
    queries_shape = (prompt_len, num_q_heads, head_dim)
    py_queries = mx.random.normal(queries_shape, dtype=dtype)

    k_prompt_shape = (prompt_len, num_kv_heads, head_dim)
    v_prompt_shape = (prompt_len, num_kv_heads, head_dim)
    py_k_prompt = mx.random.normal(k_prompt_shape, dtype=dtype)
    py_v_prompt = mx.random.normal(v_prompt_shape, dtype=dtype)

    num_total_physical_pages = (history_len + tokens_per_page - 1) // tokens_per_page

    k_cache_shape = get_k_cache_shape(
        num_total_physical_pages,
        num_kv_heads,
        head_dim,
        tokens_per_page,
        dtype
    )
    v_cache_shape = get_v_cache_shape(
        num_total_physical_pages,
        num_kv_heads,
        head_dim,
        tokens_per_page,
        dtype
    )
    py_k_cache_pool = mx.random.normal(k_cache_shape, dtype=dtype)
    py_v_cache_pool = mx.random.normal(v_cache_shape, dtype=dtype)

    page_table = mx.arange(num_total_physical_pages).reshape(1, num_total_physical_pages)
    context_lens_arr = mx.array([history_len], dtype=mx.int32)

    # Ensure all inputs are evaluated before calling the kernel
    mx.eval(
        py_queries,
        py_k_prompt,
        py_v_prompt,
        py_k_cache_pool,
        py_v_cache_pool,
        page_table,
        context_lens_arr
    )

    # --- Call paged_attention the first time ---
    logger.info("  First call to paged_attention (prefill mode)...")
    output1 = paged_attention_prefill(
        py_queries,
        py_k_prompt,
        py_v_prompt,
        py_k_cache_pool,
        py_v_cache_pool,
        page_table,
        context_lens_arr
    )
    mx.eval(output1)

    logger.info("  Second call to paged_attention (prefill mode) with identical inputs...")
    output2 = paged_attention_prefill(
        py_queries,
        py_k_prompt,
        py_v_prompt,
        py_k_cache_pool,
        py_v_cache_pool,
        page_table,
        context_lens_arr
    )
    mx.eval(output2)

    # --- Assertions ---
    assert output1.shape == output2.shape, f"Output shapes differ: {output1.shape} vs {output2.shape}"
    assert output1.dtype == output2.dtype, f"Output dtypes differ: {output1.dtype} vs {output2.dtype}"

    # For debugging, print if they are not equal
    if not mx.array_equal(output1, output2).item():
        logger.error("Non-deterministic output detected!")
        logger.error(f"Output 1 sample: {output1[0, : min(output1.shape[1], 4)] if output1.size > 0 else 'empty'}")
        logger.error(f"Output 2 sample: {output2[0, : min(output2.shape[1], 4)] if output2.size > 0 else 'empty'}")
        if output1.size > 0 and output2.size > 0 and output1.size == output2.size:
            diff_mask: mx.array = mx.abs(output1 - output2) > 1e-5
            diff_loc = np.nonzero(diff_mask.tolist())
            if diff_loc[0].size > 0:
                first_diff_idx_flat = int(diff_loc[0][0])
                row_idx = first_diff_idx_flat // head_dim
                col_idx = first_diff_idx_flat % head_dim
                output1_val = output1[row_idx, col_idx].item()
                output2_val = output2[row_idx, col_idx].item()
                logger.error(f"First difference at index [{row_idx}, {col_idx}] (flat: {first_diff_idx_flat})")
                logger.error(f"  Output1[{row_idx}, {col_idx}]: {output1_val}")
                logger.error(f"  Output2[{row_idx}, {col_idx}]: {output2_val}")

    assert mx.array_equal(output1, output2).item(), (
        f"Paged attention prefill output is not deterministic for dtype={dtype}. Outputs differ between two identical calls."
    )

    logger.info("  Result: Outputs are identical - prefill determinism verified.")
