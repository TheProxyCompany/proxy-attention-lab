import logging

import mlx.core as mx
import mlx.nn as nn
import pytest

from proxy_attention_lab import calculate_page_size, paged_attention

logger = logging.getLogger(__name__)


@pytest.mark.parametrize(
    "batch_size, seq_len, num_heads, head_dim, dtype",
    [
        (1, 16, (1, 1), 32, mx.float16),  # Original case
        (1, 32, (1, 1), 32, mx.float16),  # Longer sequence
        (1, 16, (4, 1), 32, mx.float16),  # MQA (num_q_heads > num_kv_heads)
        (1, 16, (4, 2), 32, mx.float16),  # GQA (num_q_heads > num_kv_heads, num_kv_heads > 1)
        (1, 16, (1, 1), 64, mx.float16),  # Different head dimension
        (1, 16, (2, 2), 32, mx.float16),  # num q = num kv heads
        (1, 16, (4, 2), 128, mx.float16),  # 128 head dim
        (1, 123, (32, 16), 128, mx.float16),  # long sequence
        (1, 2048, (32, 16), 128, mx.float16),  # Gemma 3 27b, long sequence
        (2, 64, (4, 4), 32, mx.float16),  # Batched Example
    ],
)
def test_pal_vs_sdpa_equivalency_mha(batch_size, seq_len, num_heads, head_dim, dtype):
    """Compare PAL paged_attention with MLX SDPA kernel.

    This test verifies that our paged_attention implementation produces numerically
    equivalent results to MLX's scaled_dot_product_attention (SDPA) function across
    different configurations. We test various combinations of:

    - Batch sizes (single item and batched)
    - Sequence lengths (shorter and longer)
    - Head configurations (including MQA and GQA variants)
    - Head dimensions

    The test:
    1. Runs MLX SDPA with random inputs as the reference implementation
    2. Converts those same inputs to the format expected by paged_attention
    3. Runs paged_attention and reshapes the output to match SDPA's output format
    4. Compares the outputs to ensure they're numerically equivalent within tolerance

    This ensures that our implementation matches the standard attention mechanism
    when the inputs are directly comparable.
    """
    mx.random.seed(11)

    logger.info(f"Test: {test_pal_vs_sdpa_equivalency_mha.__name__}")
    num_q_heads, num_kv_heads = num_heads

    logger.info("  Test Configuration:")
    logger.info(f"    Batch size: {batch_size}, Sequence length: {seq_len}")
    logger.info(f"    Query heads: {num_q_heads}, KV heads: {num_kv_heads}, Head dim: {head_dim}")
    logger.info(f"    Data type: {dtype}")

    tokens_per_page = calculate_page_size(head_dim, num_q_heads, num_kv_heads)
    logger.info(f"    Tokens per page: {tokens_per_page}")

    # --- 1. Setup Inputs & Run MLX SDPA (Reference) ---
    sdpa_q_shape = (batch_size, num_q_heads, seq_len, head_dim)
    sdpa_kv_shape = (batch_size, num_kv_heads, seq_len, head_dim)

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
    mx.eval(sdpa_output)
    logger.info(f"    SDPA output shape: {sdpa_output.shape}")

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

    logger.info("  Preparing PAL paged_attention inputs:")
    logger.info(f"    Queries shape: {pal_queries.shape}")
    logger.info(f"    KV cache shape: {pal_k_cache_pool.shape}")
    logger.info(f"    Logical pages per sequence: {num_logical_pages_per_seq}")
    logger.info(f"    Total physical pages: {num_total_physical_pages}")

    # Populate KV cache from SDPA inputs
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

    # Create page table mapping
    pal_page_table_list = []
    for b_idx in range(batch_size):
        sequence_physical_page_indices = [
            b_idx * num_logical_pages_per_seq + l_idx for l_idx in range(num_logical_pages_per_seq)
        ]
        pal_page_table_list.append(sequence_physical_page_indices)
    pal_page_table = mx.array(pal_page_table_list, dtype=mx.uint32)

    # Set sequence length for each batch item
    pal_sequence_lengths = mx.array([seq_len] * batch_size, dtype=mx.int32)

    # query_to_seq_map: maps each token in pal_queries to its sequence index
    # pal_queries has tokens ordered as [seq0_tokens, seq1_tokens, ...]
    pal_query_to_seq_map = mx.repeat(mx.arange(batch_size, dtype=mx.int32), repeats=seq_len)

    # query_token_offset: for causal attention, 0-indexed position within the sequence
    # Offsets are [0, 1, ..., SL-1, 0, 1, ..., SL-1, ...]
    pal_query_token_offset = mx.tile(mx.arange(seq_len, dtype=mx.int32), batch_size)

    pal_queries = mx.contiguous(pal_queries)
    pal_k_cache_pool = mx.contiguous(pal_k_cache_pool)
    pal_v_cache_pool = mx.contiguous(pal_v_cache_pool)
    pal_page_table = mx.contiguous(pal_page_table)
    pal_sequence_lengths = mx.contiguous(pal_sequence_lengths)
    pal_query_to_seq_map = mx.contiguous(pal_query_to_seq_map)
    pal_query_token_offset = mx.contiguous(pal_query_token_offset)

    mx.eval(pal_queries)
    mx.eval(pal_k_cache_pool)
    mx.eval(pal_v_cache_pool)
    mx.eval(pal_page_table)
    mx.eval(pal_sequence_lengths)
    mx.eval(pal_query_to_seq_map)
    mx.eval(pal_query_token_offset)

    logger.info("  PAL metadata arrays:")
    logger.info(f"    Page table shape: {pal_page_table.shape}")
    logger.info(f"    Sequence lengths: {pal_sequence_lengths}")
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
        is_prefill=True,  # explicitly use prefill mode
    )
    mx.eval(pal_output)
    logger.info(f"    PAL output shape: {pal_output.shape}")

    # --- 4. Compare PAL output with SDPA output ---
    # SDPA output is (batch_size, num_q_heads, seq_len, head_dim)
    # PAL output (from C++ op, given pal_queries shape (B*SL, NQ, HD)) is (B*SL*NQ, HD)
    # We need to reshape sdpa_output to match this.
    # (B, NQ, SL, HD) -> transpose(0, 2, 1, 3) -> (B, SL, NQ, HD) -> reshape(-1, HD) -> (B*SL*NQ, HD)
    sdpa_output_reshaped = sdpa_output.transpose(0, 2, 1, 3).reshape(-1, head_dim)

    logger.info("  Comparing outputs:")
    logger.info(f"    PAL output shape: {pal_output.shape}")
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
    current_atol = 1e-2 if seq_len < 2048 else 1e-1
    current_rtol = 1e-5
    logger.info(f"    Tolerance values - Absolute: {current_atol}, Relative: {current_rtol}")

    # Assert outputs match within tolerance
    assert mx.allclose(pal_output, sdpa_output_reshaped, atol=current_atol, rtol=current_rtol), (
        f"Numerical mismatch between PAL paged_attention and MLX SDPA for params: "
        f"bs={batch_size}, sl={seq_len}, nqh={num_q_heads}, nkvh={num_kv_heads}, hd={head_dim}, dt={dtype}. "
        f"Max diff: {max_diff}, Mean diff: {mean_diff}"
    )
