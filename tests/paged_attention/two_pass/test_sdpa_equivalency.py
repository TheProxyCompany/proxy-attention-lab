import logging

import mlx.core as mx
import mlx.nn as nn
import pytest

from proxy_attention_lab import paged_attention
from proxy_attention_lab.pal_core import get_optimal_page_size

logger = logging.getLogger(__name__)


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
@pytest.mark.parametrize(
    "seq_len, num_heads, head_dim",
    [
        (16, (1, 1), 32),  # Original case
        (32, (1, 1), 32),  # Longer sequence
        (16, (4, 1), 32),  # MQA (num_q_heads > num_kv_heads)
        (16, (4, 2), 32),  # GQA (num_q_heads > num_kv_heads, num_kv_heads > 1)
        (16, (1, 1), 64),  # Different head dimension
        (16, (2, 2), 32),  # num q = num kv heads
        (16, (4, 2), 128),  # 128 head dim
        (123, (32, 16), 128),  # long sequence
        (2048, (32, 16), 128),  # Gemma 3 27b, long sequence
        (17, (32, 8), 128),  # llama 3.1 8b failing "paris question"
    ],
)
def test_pal_vs_sdpa_equivalency(seq_len, num_heads, head_dim, dtype):
    """Compare PAL paged_attention with MLX SDPA kernel for single sequence prefill.

    This test verifies that our paged_attention implementation produces numerically
    equivalent results to MLX's scaled_dot_product_attention (SDPA) function when
    processing a single sequence in prefill mode.

    The test simulates how the PIE scheduler would call paged_attention during
    prefill: processing one sequence at a time with all tokens as "new" queries.

    We test various configurations of:
    - Sequence lengths (shorter and longer)
    - Head configurations (including MQA and GQA variants)
    - Head dimensions

    The test:
    1. Runs MLX SDPA with random inputs as the reference implementation
    2. Converts those same inputs to the format expected by paged_attention
    3. Runs paged_attention with the new kernel's expected layout
    4. Compares the outputs to ensure they're numerically equivalent within tolerance
    """
    mx.clear_cache()
    mx.random.seed(11)

    logger.info(f"Test: {test_pal_vs_sdpa_equivalency.__name__}")
    num_q_heads, num_kv_heads = num_heads

    logger.info("  Test Configuration:")
    logger.info(f"    Sequence length: {seq_len}")
    logger.info(f"    Query heads: {num_q_heads}, KV heads: {num_kv_heads}, Head dim: {head_dim}")
    logger.info(f"    Data type: {dtype}")

    tokens_per_page = get_optimal_page_size()
    logger.info(f"    Tokens per page: {tokens_per_page}")

    # --- 1. Setup Inputs & Run MLX SDPA (Reference) ---
    # For single sequence: batch_size = 1
    sdpa_q_shape = (1, num_q_heads, seq_len, head_dim)
    sdpa_kv_shape = (1, num_kv_heads, seq_len, head_dim)

    sdpa_queries = mx.random.normal(sdpa_q_shape, dtype=dtype)
    sdpa_keys = mx.random.normal(sdpa_kv_shape, dtype=dtype)
    sdpa_values = mx.random.normal(sdpa_kv_shape, dtype=dtype)

    causal_mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len).astype(dtype)
    scale = 1.0 / mx.sqrt(float(head_dim))

    logger.info("  Running MLX SDPA (Reference):")
    logger.info(f"    Input shapes - Q: {sdpa_q_shape}, K/V: {sdpa_kv_shape}")

    sdpa_output = mx.fast.scaled_dot_product_attention(
        sdpa_queries,
        sdpa_keys,
        sdpa_values,
        scale=scale,
        mask=causal_mask,
    )
    logger.info(f"    SDPA output shape: {sdpa_output.shape}")

    # --- 2. Prepare Inputs for PAL's paged_attention ---
    # For prefill: all tokens in the sequence are queries
    # Shape: [seq_len, num_q_heads, head_dim]
    pal_queries = sdpa_queries[0].transpose(1, 0, 2)  # [seq_len, num_q_heads, head_dim]

    num_logical_pages_per_seq = (seq_len + tokens_per_page - 1) // tokens_per_page

    # Create empty KV cache pools with new layout: [pages, kv_heads, tokens, head_dim]
    pal_k_cache_pool = mx.zeros((num_logical_pages_per_seq, num_kv_heads, tokens_per_page, head_dim), dtype=dtype)
    pal_v_cache_pool = mx.zeros((num_logical_pages_per_seq, num_kv_heads, tokens_per_page, head_dim), dtype=dtype)

    logger.info("  Preparing PAL paged_attention inputs:")
    logger.info(f"    Queries shape: {pal_queries.shape}")
    logger.info(f"    KV cache shape: {pal_k_cache_pool.shape}")
    logger.info(f"    Logical pages: {num_logical_pages_per_seq}")

    # Populate KV cache from SDPA inputs
    keys_to_cache = sdpa_keys[0]  # [num_kv_heads, seq_len, head_dim]
    values_to_cache = sdpa_values[0]  # [num_kv_heads, seq_len, head_dim]

    for page_idx in range(num_logical_pages_per_seq):
        token_start = page_idx * tokens_per_page
        token_end = min((page_idx + 1) * tokens_per_page, seq_len)
        tokens_in_page = token_end - token_start

        if tokens_in_page > 0:
            # Copy tokens for all KV heads at once
            pal_k_cache_pool[page_idx, :, :tokens_in_page, :] = keys_to_cache[:, token_start:token_end, :]
            pal_v_cache_pool[page_idx, :, :tokens_in_page, :] = values_to_cache[:, token_start:token_end, :]

    # Create page table - single sequence uses all pages
    page_indices = list(range(num_logical_pages_per_seq))
    pal_page_table = mx.array([page_indices], dtype=mx.uint32)  # Shape: [1, num_pages]

    # For prefill: each query position i can see tokens 0...i (causal attention)
    pal_sequence_lengths = mx.arange(1, seq_len + 1, dtype=mx.int32)

    # All queries belong to sequence 0
    pal_query_to_seq_map = mx.zeros(seq_len, dtype=mx.int32)

    # Query token offset: each query's position in the sequence
    pal_query_token_offset = mx.arange(seq_len, dtype=mx.int32)

    logger.info("  PAL metadata arrays:")
    logger.info(f"    Page table shape: {pal_page_table.shape}")
    logger.info(
        f"    Sequence lengths: {pal_sequence_lengths[:10]}..."
        if seq_len > 10
        else f"    Sequence lengths: {pal_sequence_lengths}"
    )
    logger.info(f"    Query to sequence map shape: {pal_query_to_seq_map.shape}")
    logger.info(f"    Query token offset shape: {pal_query_token_offset.shape}")

    # --- 3. Run PAL paged_attention ---
    logger.info("  Running PAL paged_attention (prefill mode):")
    pal_output = paged_attention(
        pal_queries,
        pal_k_cache_pool,
        pal_v_cache_pool,
        pal_page_table,
        pal_sequence_lengths,
        pal_query_to_seq_map,
        pal_query_token_offset,
        use_fused_kernel=False,
    )
    logger.info(f"    PAL output shape: {pal_output.shape}")

    # --- 4. Compare PAL output with SDPA output ---
    # SDPA output is [1, num_q_heads, seq_len, head_dim]
    # PAL output is [seq_len * num_q_heads, head_dim] (flattened by the kernel)
    # Reshape SDPA output to match
    sdpa_output_for_comparison = sdpa_output[0].transpose(1, 0, 2).reshape(seq_len * num_q_heads, head_dim)

    logger.info("  Comparing outputs:")
    logger.info(f"    PAL output shape: {pal_output.shape}")
    logger.info(f"    SDPA output shape for comparison: {sdpa_output_for_comparison.shape}")

    assert pal_output.shape == sdpa_output_for_comparison.shape, (
        f"Shape mismatch: PAL output {pal_output.shape}, SDPA {sdpa_output_for_comparison.shape}"
    )

    # Calculate differences between outputs
    diff = mx.abs(pal_output - sdpa_output_for_comparison)
    max_diff = mx.max(diff).item()
    mean_diff = mx.mean(diff).item()
    logger.info(f"    Difference metrics - Max: {max_diff:.6f}, Mean: {mean_diff:.6f}")

    # For FP16, we allow slightly larger differences due to numerical precision & different implementation
    current_atol = 1e-2
    current_rtol = 1e-4
    logger.info(f"    Tolerance values - Absolute: {current_atol}, Relative: {current_rtol}")

    # Assert outputs match within tolerance
    assert mx.allclose(pal_output, sdpa_output_for_comparison, atol=current_atol, rtol=current_rtol), (
        f"Numerical mismatch between PAL paged_attention and MLX SDPA for params: "
        f"sl={seq_len}, nqh={num_q_heads}, nkvh={num_kv_heads}, hd={head_dim}, dt={dtype}. "
        f"Max diff: {max_diff}, Mean diff: {mean_diff}"
    )


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
@pytest.mark.parametrize(
    "batch_size, seq_len, num_heads, head_dim",
    [
        (2, 64, (4, 4), 32),  # Batched Example
        (4, 32, (8, 2), 64),  # Larger batch with GQA
        (3, 128, (16, 8), 128),  # Mixed configuration
    ],
)
def test_pal_vs_sdpa_batched(batch_size, seq_len, num_heads, head_dim, dtype):
    """Test batched prefill scenario where multiple sequences are processed together.

    This test simulates a batched prefill operation where multiple sequences
    are processed simultaneously, each with all their tokens as queries.
    """
    mx.clear_cache()
    mx.random.seed(11)

    logger.info(f"Test: {test_pal_vs_sdpa_batched.__name__}")
    num_q_heads, num_kv_heads = num_heads

    logger.info("  Test Configuration:")
    logger.info(f"    Batch size: {batch_size}, Sequence length: {seq_len}")
    logger.info(f"    Query heads: {num_q_heads}, KV heads: {num_kv_heads}, Head dim: {head_dim}")
    logger.info(f"    Data type: {dtype}")

    tokens_per_page = get_optimal_page_size()
    logger.info(f"    Tokens per page: {tokens_per_page}")

    # --- 1. Setup Inputs & Run MLX SDPA (Reference) ---
    # For batched prefill, all tokens are queries
    sdpa_q_shape = (batch_size, num_q_heads, seq_len, head_dim)
    sdpa_kv_shape = (batch_size, num_kv_heads, seq_len, head_dim)

    sdpa_queries = mx.random.normal(sdpa_q_shape, dtype=dtype)
    sdpa_keys = mx.random.normal(sdpa_kv_shape, dtype=dtype)
    sdpa_values = mx.random.normal(sdpa_kv_shape, dtype=dtype)

    # Create causal mask for prefill
    causal_mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len).astype(dtype)
    scale = 1.0 / mx.sqrt(float(head_dim))

    logger.info("  Running MLX SDPA (Reference):")
    logger.info(f"    Input shapes - Q: {sdpa_q_shape}, K/V: {sdpa_kv_shape}")

    sdpa_output = mx.fast.scaled_dot_product_attention(
        sdpa_queries,
        sdpa_keys,
        sdpa_values,
        scale=scale,
        mask=causal_mask,
    )
    logger.info(f"    SDPA output shape: {sdpa_output.shape}")

    # --- 2. Prepare Inputs for PAL's paged_attention ---
    # For batched prefill: reshape to [batch_size * seq_len, num_q_heads, head_dim]
    # This represents all tokens from all sequences as queries
    pal_queries = sdpa_queries.transpose(0, 2, 1, 3).reshape(batch_size * seq_len, num_q_heads, head_dim)

    num_logical_pages_per_seq = (seq_len + tokens_per_page - 1) // tokens_per_page
    num_total_physical_pages = batch_size * num_logical_pages_per_seq

    # Create empty KV cache pools
    pal_k_cache_pool = mx.zeros((num_total_physical_pages, num_kv_heads, tokens_per_page, head_dim), dtype=dtype)
    pal_v_cache_pool = mx.zeros((num_total_physical_pages, num_kv_heads, tokens_per_page, head_dim), dtype=dtype)

    logger.info("  Preparing PAL paged_attention inputs:")
    logger.info(f"    Queries shape: {pal_queries.shape}")
    logger.info(f"    KV cache shape: {pal_k_cache_pool.shape}")
    logger.info(f"    Total physical pages: {num_total_physical_pages}")

    # Populate KV cache from SDPA inputs
    for b_idx in range(batch_size):
        keys_to_cache_b = sdpa_keys[b_idx]  # [num_kv_heads, seq_len, head_dim]
        values_to_cache_b = sdpa_values[b_idx]  # [num_kv_heads, seq_len, head_dim]

        for l_idx in range(num_logical_pages_per_seq):
            physical_page_idx = b_idx * num_logical_pages_per_seq + l_idx

            token_start = l_idx * tokens_per_page
            token_end = min((l_idx + 1) * tokens_per_page, seq_len)
            tokens_in_page = token_end - token_start

            if tokens_in_page > 0:
                pal_k_cache_pool[physical_page_idx, :, :tokens_in_page, :] = keys_to_cache_b[
                    :, token_start:token_end, :
                ]
                pal_v_cache_pool[physical_page_idx, :, :tokens_in_page, :] = values_to_cache_b[
                    :, token_start:token_end, :
                ]

    # Create page table mapping
    pal_page_table_list = []
    for b_idx in range(batch_size):
        sequence_physical_page_indices = [
            b_idx * num_logical_pages_per_seq + l_idx for l_idx in range(num_logical_pages_per_seq)
        ]
        pal_page_table_list.append(sequence_physical_page_indices)
    pal_page_table = mx.array(pal_page_table_list, dtype=mx.uint32)

    # For batched prefill: each query position i in sequence b can see tokens 0...i
    # We need to repeat the pattern [1, 2, ..., seq_len] for each sequence
    pal_sequence_lengths = mx.tile(mx.arange(1, seq_len + 1, dtype=mx.int32), batch_size)

    # Map each query to its sequence: [0, 0, ..., 0, 1, 1, ..., 1, ...]
    pal_query_to_seq_map = mx.repeat(mx.arange(batch_size, dtype=mx.int32), seq_len)

    # Query token offset: [0, 1, ..., seq_len-1, 0, 1, ..., seq_len-1, ...]
    pal_query_token_offset = mx.tile(mx.arange(seq_len, dtype=mx.int32), batch_size)

    logger.info("  PAL metadata arrays:")
    logger.info(f"    Page table shape: {pal_page_table.shape}")
    logger.info(f"    Sequence lengths shape: {pal_sequence_lengths.shape}")
    logger.info(f"    Query to sequence map shape: {pal_query_to_seq_map.shape}")
    logger.info(f"    Query token offset shape: {pal_query_token_offset.shape}")

    # --- 3. Run PAL paged_attention ---
    logger.info("  Running PAL paged_attention (batched prefill mode with two-pass kernel):")
    pal_output = paged_attention(
        pal_queries,
        pal_k_cache_pool,
        pal_v_cache_pool,
        pal_page_table,
        pal_sequence_lengths,
        pal_query_to_seq_map,
        pal_query_token_offset,
        use_fused_kernel=False,  # Use two-pass kernel for prefill
    )
    logger.info(f"    PAL output shape: {pal_output.shape}")

    # --- 4. Compare PAL output with SDPA output ---
    # SDPA output is [batch_size, num_q_heads, seq_len, head_dim]
    # PAL output is [batch_size * seq_len * num_q_heads, head_dim] (flattened by the kernel)
    # Reshape SDPA output to match
    sdpa_output_for_comparison = sdpa_output.transpose(0, 2, 1, 3).reshape(batch_size * seq_len * num_q_heads, head_dim)

    logger.info("  Comparing outputs:")
    logger.info(f"    PAL output shape: {pal_output.shape}")
    logger.info(f"    SDPA output shape for comparison: {sdpa_output_for_comparison.shape}")

    assert pal_output.shape == sdpa_output_for_comparison.shape, (
        f"Shape mismatch: PAL output {pal_output.shape}, SDPA {sdpa_output_for_comparison.shape}"
    )

    # Calculate differences between outputs
    diff = mx.abs(pal_output - sdpa_output_for_comparison)
    max_diff = mx.max(diff).item()
    mean_diff = mx.mean(diff).item()
    logger.info(f"    Difference metrics - Max: {max_diff:.6f}, Mean: {mean_diff:.6f}")

    # For FP16, we allow slightly larger differences due to numerical precision & different implementation
    current_atol = 1e-2
    current_rtol = 1e-4
    logger.info(f"    Tolerance values - Absolute: {current_atol}, Relative: {current_rtol}")

    # Assert outputs match within tolerance
    assert mx.allclose(pal_output, sdpa_output_for_comparison, atol=current_atol, rtol=current_rtol), (
        f"Numerical mismatch between PAL paged_attention and MLX SDPA for params: "
        f"bs={batch_size}, sl={seq_len}, nqh={num_q_heads}, nkvh={num_kv_heads}, hd={head_dim}, dt={dtype}. "
        f"Max diff: {max_diff}, Mean diff: {mean_diff}"
    )
