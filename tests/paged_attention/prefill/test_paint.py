# tests/paged_attention/prefill/test_prompt_attention.py
import logging

import mlx.core as mx
import numpy as np
import pytest

from proxy_attention_lab import (
    get_k_cache_shape,
    get_v_cache_shape,
    paged_attention_prefill,
)

logger = logging.getLogger(__name__)


def manual_prompt_attention(q_prompt, k_prompt, v_prompt, inv_sqrt_head_dim):
    """
    A simple NumPy implementation of the prompt-on-prompt attention logic
    to verify the kernel's intermediate output.
    """
    num_prompt_tokens, num_q_heads, head_dim = q_prompt.shape

    # Use float32 for all calculations for precision
    q = np.array(q_prompt, dtype=np.float32) * inv_sqrt_head_dim
    k = np.array(k_prompt, dtype=np.float32)
    v = np.array(v_prompt, dtype=np.float32)

    output_accumulator = np.zeros_like(q, dtype=np.float32)

    for h in range(num_q_heads):
        # Online softmax stats for this head
        max_scores = np.full(num_prompt_tokens, -np.inf, dtype=np.float32)
        sum_exps = np.zeros(num_prompt_tokens, dtype=np.float32)

        # Accumulator for this head
        head_accumulator = np.zeros((num_prompt_tokens, head_dim), dtype=np.float32)

        # Loop over keys
        for j in range(num_prompt_tokens):
            k_vec = k[:, h, :]  # For GQA, this would map to kv_head
            v_vec = v[:, h, :]

            # Loop over queries
            for i in range(num_prompt_tokens):
                if j > i:  # Causal mask
                    continue

                # Compute score
                score = np.dot(q[i, h, :], k_vec[j, :])

                # Online softmax update
                old_max = max_scores[i]
                new_max = max(old_max, score)

                if new_max > old_max:
                    scale = np.exp(old_max - new_max)
                    head_accumulator[i, :] *= scale
                    sum_exps[i] *= scale

                prob = np.exp(score - new_max)
                sum_exps[i] += prob
                head_accumulator[i, :] += prob * v_vec[j, :]
                max_scores[i] = new_max

        output_accumulator[:, h, :] = head_accumulator

    return output_accumulator


@pytest.mark.parametrize("dtype", [mx.float16])
def test_prompt_on_prompt_logic(dtype):
    """
    Verifies the intermediate result of the prompt-on-prompt attention loop.
    Compares the kernel's unnormalized output against a manual NumPy calculation.
    """
    logger.info(f"Test: {test_prompt_on_prompt_logic.__name__} (dtype={dtype})")

    # Use small, deterministic values for easy debugging
    num_prompt_tokens = 4
    num_q_heads = 1
    head_dim = 32
    num_kv_heads = 1

    # Create simple, non-random inputs
    q_prompt = (
        mx.arange(num_prompt_tokens * num_q_heads * head_dim, dtype=dtype).reshape(
            num_prompt_tokens, num_q_heads, head_dim
        )
        * 0.1
    )
    k_prompt = (
        mx.arange(num_prompt_tokens * num_kv_heads * head_dim, dtype=dtype).reshape(
            num_prompt_tokens, num_kv_heads, head_dim
        )
        * 0.1
    )
    v_prompt = mx.ones_like(k_prompt)  # Use ones for V to make accumulation obvious

    # Dummy history inputs (will be ignored by the current kernel state)
    k_cache_paged = mx.zeros(get_k_cache_shape(1, num_kv_heads, head_dim, 16, dtype))
    v_cache_paged = mx.zeros(get_v_cache_shape(1, num_kv_heads, head_dim, 16, dtype))
    page_table = mx.array([[0]], dtype=mx.uint32)
    context_len_arr = mx.array([0], dtype=mx.int32)  # No history

    # Calculate expected result on CPU
    inv_sqrt_head_dim = 1.0 / np.sqrt(head_dim)
    expected_np = manual_prompt_attention(q_prompt, k_prompt, v_prompt, inv_sqrt_head_dim)
    expected_mx = mx.array(expected_np, dtype=mx.float32)

    # Run the kernel
    out_unnormalized = paged_attention_prefill(
        q_prompt,
        k_prompt,
        v_prompt,
        k_cache_paged,
        v_cache_paged,
        page_table,
        context_len_arr,
    )
    mx.eval(out_unnormalized)

    # Reshape kernel output to match expected [tokens, heads, dim]
    out_reshaped = out_unnormalized.reshape(num_prompt_tokens, num_q_heads, head_dim)

    for i in range(num_prompt_tokens):
        logger.info(f"First 3 elements of Expected ({i} vec): {expected_mx[i].tolist()[0][:3]}")
        logger.info(f"First 3 elements of Actual ({i} vec):   {out_reshaped[i].astype(mx.float32).tolist()[0][:3]}")

    assert mx.allclose(out_reshaped.astype(mx.float32), expected_mx, atol=1e-2), (
        "Kernel output does not match manual calculation."
    )
