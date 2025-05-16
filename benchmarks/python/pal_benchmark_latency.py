#!/usr/bin/env python3
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
"""Benchmark script for measuring PAL paged_attention kernel latency."""

import logging
import time

import mlx.core as mx

from proxy_attention_lab import paged_attention

# Configure basic logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def run_latency_benchmark(params: dict) -> float:
    """
    Run a latency benchmark for the paged_attention kernel.

    Args:
        params: Dictionary containing benchmark parameters:
            - num_query_items: Total number of query-head items to process
            - num_q_heads: Number of query heads
            - num_kv_heads: Number of K/V heads
            - head_dim: Dimension of each head
            - seq_len: Effective history length for all items
            - tokens_per_page: Number of tokens per page (typically 64)
            - num_sequences_in_batch: Number of sequences in the batch
            - dtype: Data type for tensors (e.g., mx.float16)

    Returns:
        float: Average execution time per iteration in milliseconds
    """
    logger.info("Starting paged_attention latency benchmark with the following parameters:")
    for k, v in params.items():
        logger.info(f"  {k}: {v}")

    # Derived parameters
    num_tokens = params["num_query_items"] // params["num_q_heads"]
    assert params["num_query_items"] % params["num_q_heads"] == 0, "num_query_items must be a multiple of num_q_heads"

    num_logical_pages_per_seq = (params["seq_len"] + params["tokens_per_page"] - 1) // params["tokens_per_page"]
    num_total_physical_pages = params["num_sequences_in_batch"] * num_logical_pages_per_seq

    logger.info("Derived parameters:")
    logger.info(f"  num_tokens: {num_tokens}")
    logger.info(f"  num_logical_pages_per_seq: {num_logical_pages_per_seq}")
    logger.info(f"  num_total_physical_pages: {num_total_physical_pages}")

    # Create input tensors
    logger.info("Creating input tensors...")

    # Queries: [num_tokens, num_q_heads, head_dim]
    py_queries = mx.random.normal((num_tokens, params["num_q_heads"], params["head_dim"]), dtype=params["dtype"])

    # K/V cache pools: [num_total_physical_pages, tokens_per_page, num_kv_heads, head_dim]
    py_k_cache_pool = mx.random.normal(
        (num_total_physical_pages, params["tokens_per_page"], params["num_kv_heads"], params["head_dim"]),
        dtype=params["dtype"],
    )

    py_v_cache_pool = mx.random.normal(
        (num_total_physical_pages, params["tokens_per_page"], params["num_kv_heads"], params["head_dim"]),
        dtype=params["dtype"],
    )

    # Page table: [num_sequences_in_batch, num_logical_pages_per_seq]
    # Each sequence maps to a unique contiguous block of physical pages
    pal_page_table_list = []
    for b_idx in range(params["num_sequences_in_batch"]):
        sequence_physical_page_indices = [
            b_idx * num_logical_pages_per_seq + l_idx for l_idx in range(num_logical_pages_per_seq)
        ]
        pal_page_table_list.append(sequence_physical_page_indices)
    py_page_table = mx.array(pal_page_table_list, dtype=mx.uint32)

    # Sequence lengths: [num_sequences_in_batch]
    py_sequence_lengths = mx.array([params["seq_len"]] * params["num_sequences_in_batch"], dtype=mx.int32)

    # Query to sequence map: [num_tokens]
    if params["num_sequences_in_batch"] == 1:
        py_query_to_seq_map = mx.zeros(num_tokens, dtype=mx.int32)
    else:
        # Ensure num_tokens is multiple of num_sequences_in_batch for even distribution
        assert num_tokens % params["num_sequences_in_batch"] == 0, (
            "For multi-sequence batch, num_tokens should be a multiple of num_sequences_in_batch"
        )
        tokens_per_seq_in_map = num_tokens // params["num_sequences_in_batch"]
        py_query_to_seq_map = mx.repeat(
            mx.arange(params["num_sequences_in_batch"], dtype=mx.int32), repeats=tokens_per_seq_in_map
        )

    # Query token offset: [num_tokens]
    # Set all offsets to seq_len so each token attends to a full history
    py_query_token_offset = mx.array([params["seq_len"]] * num_tokens, dtype=mx.int32)

    logger.info("Input tensor shapes:")
    logger.info(f"  queries: {py_queries.shape}")
    logger.info(f"  k_cache_pool: {py_k_cache_pool.shape}")
    logger.info(f"  v_cache_pool: {py_v_cache_pool.shape}")
    logger.info(f"  page_table: {py_page_table.shape}")
    logger.info(f"  sequence_lengths: {py_sequence_lengths.shape}")
    logger.info(f"  query_to_seq_map: {py_query_to_seq_map.shape}")
    logger.info(f"  query_token_offset: {py_query_token_offset.shape}")

    # Warm-up
    logger.info("Performing warm-up run...")
    warmup_out = paged_attention(
        py_queries,
        py_k_cache_pool,
        py_v_cache_pool,
        py_page_table,
        py_sequence_lengths,
        py_query_to_seq_map,
        py_query_token_offset,
    )
    mx.eval(warmup_out)

    # Timing loop
    num_iterations = 100
    logger.info(f"Starting timing loop with {num_iterations} iterations...")

    start_time = time.perf_counter()
    last_out = None

    for _i in range(num_iterations):
        last_out = paged_attention(
            py_queries,
            py_k_cache_pool,
            py_v_cache_pool,
            py_page_table,
            py_sequence_lengths,
            py_query_to_seq_map,
            py_query_token_offset,
        )

    # Ensure all operations are complete
    mx.eval(last_out)
    end_time = time.perf_counter()

    # Calculate average time in milliseconds
    total_time_ms = (end_time - start_time) * 1000
    avg_time_ms = total_time_ms / num_iterations

    logger.info("Benchmark completed.")
    logger.info(f"Total time for {num_iterations} iterations: {total_time_ms:.2f} ms")
    logger.info(f"Average time per iteration: {avg_time_ms:.4f} ms")

    # Verify output shape
    expected_items = num_tokens * params["num_q_heads"]
    expected_shape = (expected_items, params["head_dim"])
    logger.info(f"Output shape: {last_out.shape} (Expected: {expected_shape})")

    return avg_time_ms


if __name__ == "__main__":
    # Define baseline configuration
    baseline_config = {
        "num_query_items": 64,  # e.g., 64 tokens * 1 q_head, or 32 tokens * 2 q_heads
        "num_q_heads": 1,
        "num_kv_heads": 1,
        "head_dim": 128,
        "seq_len": 128,  # History length
        "tokens_per_page": 64,
        "num_sequences_in_batch": 1,
        "dtype": mx.float16,
    }

    # Print CSV header for structured output
    print(
        "num_query_items,num_q_heads,num_kv_heads,head_dim,seq_len,tokens_per_page,num_sequences_in_batch,avg_latency_ms"
    )

    # --- Sweep 1: Varying seq_len ---
    logger.info(
        f"--- Starting sweep for seq_len (fixed head_dim={baseline_config['head_dim']}, fixed num_query_items={baseline_config['num_query_items']}) ---"
    )
    seq_len_values = [64, 128, 256, 512, 1024, 2048]

    for sl_val in seq_len_values:
        current_params = baseline_config.copy()
        current_params["seq_len"] = sl_val

        # Check for parameter compatibility
        if current_params["num_q_heads"] > 0 and current_params["num_query_items"] % current_params["num_q_heads"] != 0:
            logger.warning(
                f"Skipping invalid combination: num_query_items {current_params['num_query_items']} not divisible by num_q_heads {current_params['num_q_heads']}"
            )
            continue

        try:
            avg_time_ms = run_latency_benchmark(current_params)

            # Print CSV-formatted result
            print(
                f"{current_params['num_query_items']},{current_params['num_q_heads']},{current_params['num_kv_heads']},{current_params['head_dim']},{current_params['seq_len']},{current_params['tokens_per_page']},{current_params['num_sequences_in_batch']},{avg_time_ms:.4f}"
            )
        except Exception as e:
            logger.error(f"Error during benchmark with seq_len={sl_val}: {e!s}")
            print(
                f"{current_params['num_query_items']},{current_params['num_q_heads']},{current_params['num_kv_heads']},{current_params['head_dim']},{current_params['seq_len']},{current_params['tokens_per_page']},{current_params['num_sequences_in_batch']},ERROR"
            )

    # --- Sweep 2: Varying head_dim ---
    logger.info(
        f"--- Starting sweep for head_dim (fixed seq_len={baseline_config['seq_len']}, fixed num_query_items={baseline_config['num_query_items']}) ---"
    )
    head_dim_values = [64, 128, 192, 256]

    for hd_val in head_dim_values:
        current_params = baseline_config.copy()
        current_params["head_dim"] = hd_val

        # Check for parameter compatibility
        if current_params["num_q_heads"] > 0 and current_params["num_query_items"] % current_params["num_q_heads"] != 0:
            logger.warning(
                f"Skipping invalid combination: num_query_items {current_params['num_query_items']} not divisible by num_q_heads {current_params['num_q_heads']}"
            )
            continue

        try:
            avg_time_ms = run_latency_benchmark(current_params)

            # Print CSV-formatted result
            print(
                f"{current_params['num_query_items']},{current_params['num_q_heads']},{current_params['num_kv_heads']},{current_params['head_dim']},{current_params['seq_len']},{current_params['tokens_per_page']},{current_params['num_sequences_in_batch']},{avg_time_ms:.4f}"
            )
        except Exception as e:
            logger.error(f"Error during benchmark with head_dim={hd_val}: {e!s}")
            print(
                f"{current_params['num_query_items']},{current_params['num_q_heads']},{current_params['num_kv_heads']},{current_params['head_dim']},{current_params['seq_len']},{current_params['tokens_per_page']},{current_params['num_sequences_in_batch']},ERROR"
            )

    # --- Sweep 3: Varying num_query_items ---
    logger.info(
        f"--- Starting sweep for num_query_items (fixed seq_len={baseline_config['seq_len']}, fixed head_dim={baseline_config['head_dim']}) ---"
    )
    num_query_items_values = [32, 64, 128, 256, 512]

    for nqi_val in num_query_items_values:
        current_params = baseline_config.copy()
        current_params["num_query_items"] = nqi_val

        # Check for parameter compatibility
        if current_params["num_q_heads"] > 0 and current_params["num_query_items"] % current_params["num_q_heads"] != 0:
            logger.warning(
                f"Skipping invalid combination: num_query_items {current_params['num_query_items']} not divisible by num_q_heads {current_params['num_q_heads']}"
            )
            continue

        try:
            avg_time_ms = run_latency_benchmark(current_params)

            # Print CSV-formatted result
            print(
                f"{current_params['num_query_items']},{current_params['num_q_heads']},{current_params['num_kv_heads']},{current_params['head_dim']},{current_params['seq_len']},{current_params['tokens_per_page']},{current_params['num_sequences_in_batch']},{avg_time_ms:.4f}"
            )
        except Exception as e:
            logger.error(f"Error during benchmark with num_query_items={nqi_val}: {e!s}")
            print(
                f"{current_params['num_query_items']},{current_params['num_q_heads']},{current_params['num_kv_heads']},{current_params['head_dim']},{current_params['seq_len']},{current_params['tokens_per_page']},{current_params['num_sequences_in_batch']},ERROR"
            )
