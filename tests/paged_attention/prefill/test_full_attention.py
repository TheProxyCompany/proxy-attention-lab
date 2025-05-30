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
"""Tests for full attention computation.

This module contains tests that verify the complete paged attention operation,
including matrix multiplication, softmax, and value aggregation.
"""

import logging

import mlx.core as mx

from proxy_attention_lab import calculate_page_size, paged_attention  # Import calculate_page_size

logger = logging.getLogger(__name__)


def test_full_attention_in_one_block_prefill() -> None:  # Renamed for clarity
    """Test prefill full attention computation for multiple query tokens in a single sequence,
    with history contained within one logical block.

    This test verifies that the prefill kernel correctly computes attention for
    each query token in a sequence against its causal history.
    """
    test_name = "test_full_attention_in_one_block_prefill"
    logger.info(f"Test: {test_name}")

    # --- Configuration ---
    num_q_heads = 1  # For simplicity, matching original test's single query focus per step
    num_kv_heads = 1
    head_dim = 4
    dtype = mx.float16

    # Determine D_s (tokens_per_page for kernel) based on config
    # This test's K/V cache will be set up with this page size.
    tokens_per_page = calculate_page_size(head_dim, num_q_heads, num_kv_heads)
    logger.info(f"  Calculated tokens_per_page (D_s for kernel): {tokens_per_page}")

    # Scenario: Prefill a sequence of 3 tokens.
    # Query at pos 0 attends to self (K_0, V_0)
    # Query at pos 1 attends to self and K_0 (K_1,V_1 and K_0,V_0)
    # Query at pos 2 attends to self and K_0, K_1 (K_2,V_2 and K_0,V_0 and K_1,V_1)
    # The original test effectively looked at the output for the query at current_position = 3 (0-indexed, so 4th token)
    # Let's prefill a sequence of length 3 (tokens at pos 0, 1, 2).
    # We will then check the output for the token at pos 2 (the last one).

    prefill_sequence_length = 3  # We are prefilling 3 tokens (at pos 0, 1, 2)

    logger.info("  Test Configuration:")
    logger.info(f"    head_dim: {head_dim}, num_q_heads: {num_q_heads}, num_kv_heads: {num_kv_heads}")
    logger.info(f"    Prefilling sequence of length: {prefill_sequence_length}")
    logger.info(f"    Using K/V cache page size (tokens_per_page): {tokens_per_page}")

    # --- Setup Test Inputs ---
    # 1. Queries: Shape [TotalNumQueryTokensInBatch, NumQHeads, HeadDim]
    #    TotalNumQueryTokensInBatch = prefill_sequence_length for this single sequence test.
    #    We need distinct Q vectors for each token being prefilled.
    #    Original test used Q = [1.0, 2.0, 3.0, 4.0]. Let's make them slightly different for each position.
    q_data_list = [
        [1.0, 2.0, 3.0, 4.0],  # Q for token at pos 0
        [1.1, 2.1, 3.1, 4.1],  # Q for token at pos 1
        [1.2, 2.2, 3.2, 4.2],  # Q for token at pos 2
    ]
    py_queries_list = [mx.array(q_d, dtype=dtype).reshape(1, head_dim) for q_d in q_data_list]

    # If num_q_heads > 1, we'd replicate or make them distinct per head. For num_q_heads=1:
    py_queries_stacked = mx.stack(py_queries_list, axis=0)  # Shape [prefill_sequence_length, 1, head_dim]
    if num_q_heads == 1:
        py_queries = py_queries_stacked.reshape(prefill_sequence_length, head_dim)  # Make it 2D [TotalTokens, HD]
    else:  # For multi-head, ensure shape [TotalTokens, NumQHeads, HD]
        py_queries = py_queries_stacked.reshape(prefill_sequence_length, 1, head_dim)
        py_queries = mx.repeat(py_queries, repeats=num_q_heads, axis=1)

    # 2. K/V Cache Pools:
    #    Need enough physical pages for `prefill_sequence_length` tokens.
    num_logical_pages_for_seq = (prefill_sequence_length + tokens_per_page - 1) // tokens_per_page
    num_physical_pages = num_logical_pages_for_seq

    kv_cache_shape = (num_physical_pages, tokens_per_page, num_kv_heads, head_dim)
    py_k_cache_pool = mx.zeros(kv_cache_shape, dtype=dtype)
    py_v_cache_pool = mx.zeros(kv_cache_shape, dtype=dtype)

    # K-vectors for history positions (these are the actual keys in the cache)
    # These match the original test's K values for positions 0, 1, 2.
    k_history_data = {
        0: [1.0, 1.0, 1.0, 1.0],  # K for token at pos 0
        1: [2.0, 2.0, 2.0, 2.0],  # K for token at pos 1
        2: [0.5, 0.5, 0.5, 0.5],  # K for token at pos 2
    }
    v_history_data = {
        0: [10.0, 11.0, 12.0, 13.0],  # V for token at pos 0
        1: [20.0, 21.0, 22.0, 23.0],  # V for token at pos 1
        2: [30.0, 31.0, 32.0, 33.0],  # V for token at pos 2
    }

    logger.info("  KV Cache Setup:")
    for i in range(prefill_sequence_length):
        page_idx = i // tokens_per_page
        slot_idx = i % tokens_per_page
        if page_idx < num_physical_pages:  # Should always be true with correct num_physical_pages
            py_k_cache_pool[page_idx, slot_idx, 0, :] = mx.array(k_history_data[i], dtype=dtype)
            py_v_cache_pool[page_idx, slot_idx, 0, :] = mx.array(v_history_data[i], dtype=dtype)
            logger.info(
                f"    K at history pos {i} (page {page_idx}, slot {slot_idx}): {py_k_cache_pool[page_idx, slot_idx, 0, :]}"
            )
            logger.info(
                f"    V at history pos {i} (page {page_idx}, slot {slot_idx}): {py_v_cache_pool[page_idx, slot_idx, 0, :]}"
            )

    # 3. Page Table: [NumSequencesInBatch, MaxLogicalBlocksPerSeq]
    max_logical_blocks_per_seq = num_logical_pages_for_seq
    py_page_table_list = []
    # Sequence 0 uses physical pages 0, 1, ... (up to num_logical_pages_for_seq - 1)
    pages_for_this_seq = list(range(num_logical_pages_for_seq))
    while (
        len(pages_for_this_seq) < max_logical_blocks_per_seq
    ):  # Should not happen if max_logical_blocks_per_seq is correct
        pages_for_this_seq.append(0)  # Dummy
    py_page_table_list.append(pages_for_this_seq)
    py_page_table = mx.array(py_page_table_list, dtype=mx.uint32)

    # 4. Sequence Lengths:
    py_sequence_lengths = mx.array([prefill_sequence_length], dtype=mx.int32)

    # 5. Query to Sequence Map:
    py_query_to_seq_map = mx.zeros(prefill_sequence_length, dtype=mx.int32)  # All queries map to seq 0

    # 6. Query Token Offset: For prefill, these are 0, 1, ..., L-1
    py_query_token_offset = mx.arange(prefill_sequence_length, dtype=mx.int32)

    mx.eval(
        py_queries,
        py_k_cache_pool,
        py_v_cache_pool,
        py_page_table,
        py_sequence_lengths,
        py_query_to_seq_map,
        py_query_token_offset,
    )

    # --- Run paged attention ---
    output_arr = paged_attention(
        py_queries,
        py_k_cache_pool,
        py_v_cache_pool,
        py_page_table,
        py_sequence_lengths,
        py_query_to_seq_map,
        py_query_token_offset,
        is_prefill=True,  # CRITICAL: Testing prefill path
    )
    mx.eval(output_arr)

    # --- Calculate expected output (Python reference) ---
    # The output_arr will have shape [TotalQueryTokensInBatch * NumQHeads, HeadDim]
    # For this test, TotalQueryTokensInBatch = 3, NumQHeads = 1. So, shape [3, 4].
    # We are interested in the output for the last query token (at logical position 2).

    expected_outputs_all_tokens = []
    py_scale = 1.0 / mx.sqrt(mx.array(float(head_dim))).item()

    for q_idx_in_prefill in range(prefill_sequence_length):
        current_q_logical_pos = q_idx_in_prefill  # 0, 1, or 2

        # Select the correct query vector based on num_q_heads and input query shape
        if py_queries.ndim == 3:  # [TotalTokens, NumQHeads, HD]
            current_q_vector = py_queries[q_idx_in_prefill, 0, :].astype(mx.float32)  # Assuming checking for q_head 0
        else:  # [TotalTokens, HD]
            current_q_vector = py_queries[q_idx_in_prefill, :].astype(mx.float32)

        scores_for_current_q = []
        # A query at current_q_logical_pos attends to history 0 up to current_q_logical_pos (inclusive due to causal mask fix)
        # Corrected causal mask: history_token_logical_pos <= current_q_logical_pos
        for hist_idx in range(current_q_logical_pos + 1):
            page_h = hist_idx // tokens_per_page
            slot_h = hist_idx % tokens_per_page
            k_vec = py_k_cache_pool[page_h, slot_h, 0, :].astype(mx.float32)  # Assuming kv_head 0
            score = (mx.sum(current_q_vector * k_vec) * py_scale).item()
            scores_for_current_q.append(score)

        if not scores_for_current_q:  # Should not happen if current_q_logical_pos >= 0
            max_score_for_current_q = -float("inf")  # Or 0.0f if truly no attention
            sum_exp_for_current_q = 0.0
            probs_for_current_q = []
        else:
            max_score_for_current_q = max(scores_for_current_q)
            exp_scores_minus_max = [mx.exp(s - max_score_for_current_q).item() for s in scores_for_current_q]
            sum_exp_for_current_q = sum(exp_scores_minus_max)
            if sum_exp_for_current_q == 0:  # Avoid division by zero
                probs_for_current_q = [0.0] * len(scores_for_current_q)
            else:
                probs_for_current_q = [es / sum_exp_for_current_q for es in exp_scores_minus_max]

        # Accumulate in higher precision to avoid intermediate fp16 rounding errors
        expected_v_for_current_q_f32 = mx.zeros(head_dim, dtype=mx.float32)
        for i, prob in enumerate(probs_for_current_q):
            hist_idx_for_v = i  # Probs are in order of history 0, 1, ... current_q_logical_pos
            page_v = hist_idx_for_v // tokens_per_page
            slot_v = hist_idx_for_v % tokens_per_page
            v_hist = py_v_cache_pool[page_v, slot_v, 0, :].astype(mx.float16)  # Assuming kv_head 0
            # Perform accumulation in float32
            expected_v_for_current_q_f32 += v_hist.astype(mx.float32) * prob

        # Convert to float16 at the end to match kernel output dtype
        expected_outputs_all_tokens.append(expected_v_for_current_q_f32.astype(mx.float16))

    # Stack the expected outputs for all prefill tokens
    # If num_q_heads > 1, this would need to be flattened or reshaped to match kernel output.
    # For num_q_heads=1, and 2D query input, output is [TotalTokens, HD]
    # If query input was 3D [TotalTokens, 1, HD], output is [TotalTokens*1, HD]
    expected_V_output_all_tokens_stacked = mx.stack(expected_outputs_all_tokens, axis=0)

    # The original test was interested in the state after "current_position = 3" (which is query at index 2)
    # So we compare against the output for the last token prefilled (q_idx_in_prefill = 2)
    # The kernel output `output_arr` will have results for all `prefill_sequence_length * num_q_heads` items.
    # If num_q_heads is 1, then output_arr[q_idx_in_prefill] is the one we want.

    # Let's check the output for the query token at logical position 2 (the last one prefilled)
    # This corresponds to the original test's intent of checking a query at a certain position with its history.
    idx_to_check = prefill_sequence_length - 1  # Output for the last prefilled token

    # The kernel output `output_arr` has shape [TotalQueryTokensInBatch * NumQHeads, HeadDim]
    # For this test, TotalQueryTokensInBatch = prefill_sequence_length, NumQHeads = num_q_heads
    # So, the item corresponding to q_idx_in_prefill for q_head 0 is at row:
    # (q_idx_in_prefill * num_q_heads) + 0 (for q_head_idx 0)
    output_row_to_check = (idx_to_check * num_q_heads) + 0  # Assuming we check for q_head 0
    actual_output_for_token = output_arr[output_row_to_check, :]
    expected_output_for_token = expected_outputs_all_tokens[idx_to_check]

    logger.info(f"  Checking output for prefill token at logical position: {idx_to_check}")
    logger.info(
        f"    Scores for this token: {scores_for_current_q if idx_to_check == prefill_sequence_length - 1 else 'not last token'}"
    )  # Log scores for last token
    logger.info(
        f"    Max score for this token: {max_score_for_current_q if idx_to_check == prefill_sequence_length - 1 else 'not last token'}"
    )
    logger.info(
        f"    Softmax Probs for this token: {probs_for_current_q if idx_to_check == prefill_sequence_length - 1 else 'not last token'}"
    )
    logger.info(f"    Expected V-aggregation for this token: {expected_output_for_token}")
    logger.info(f"    Actual output from kernel for this token: {actual_output_for_token}")
    logger.info(f"    Full kernel output_arr shape: {output_arr.shape}")
    logger.info(f"    Full expected_V_output_all_tokens_stacked shape: {expected_V_output_all_tokens_stacked.shape}")

    # Verify the output for ALL prefilled tokens
    expected_output_shape_overall = (prefill_sequence_length * num_q_heads, head_dim)
    assert output_arr.shape == expected_output_shape_overall, (
        f"Overall output shape {output_arr.shape} does not match expected {expected_output_shape_overall}"
    )
    assert output_arr.dtype == dtype, f"Output dtype {output_arr.dtype} does not match {dtype}"

    # If num_q_heads > 1, expected_V_output_all_tokens_stacked might need reshaping or careful indexing
    # Assuming num_q_heads = 1 for direct comparison here based on original test intent
    if num_q_heads == 1:
        assert mx.allclose(output_arr, expected_V_output_all_tokens_stacked, atol=2e-2), (
            f"Value mismatch for full prefill. Expected: {expected_V_output_all_tokens_stacked}, Got: {output_arr}"
        )
    else:
        # For multi-head, need to interleave/reshape expected or compare head by head
        logger.warning("Multi-head output comparison not fully implemented in this refactored test's assertion yet.")
        # Basic check for the first head of the last token
        assert mx.allclose(actual_output_for_token, expected_output_for_token, atol=2e-2), (
            f"Value mismatch for last token, head 0. Expected: {expected_output_for_token}, Got: {actual_output_for_token}"
        )

    logger.info(f"{test_name} PASSED (pending full multi-head validation if NQ>1)")
