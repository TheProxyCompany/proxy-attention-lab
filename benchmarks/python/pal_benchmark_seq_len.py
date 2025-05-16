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
"""Benchmark script for measuring PAL paged_attention kernel latency vs. sequence length."""

import logging
import time

import mlx.core as mx

from proxy_attention_lab import paged_attention

# Configure basic logging - suppress most logs for cleaner output
logging.basicConfig(level=logging.WARNING, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def run_latency_benchmark(params: dict) -> float:
    """
    Run a latency benchmark for the paged_attention kernel.

    Args:
        params: Dictionary containing benchmark parameters

    Returns:
        float: Average execution time per iteration in milliseconds
    """
    # Derived parameters
    num_tokens = params["num_query_items"] // params["num_q_heads"]
    num_logical_pages_per_seq = (params["seq_len"] + params["tokens_per_page"] - 1) // params["tokens_per_page"]
    num_total_physical_pages = params["num_sequences_in_batch"] * num_logical_pages_per_seq

    # Create input tensors
    py_queries = mx.random.normal((num_tokens, params["num_q_heads"], params["head_dim"]), dtype=params["dtype"])

    py_k_cache_pool = mx.random.normal(
        (num_total_physical_pages, params["tokens_per_page"], params["num_kv_heads"], params["head_dim"]),
        dtype=params["dtype"],
    )

    py_v_cache_pool = mx.random.normal(
        (num_total_physical_pages, params["tokens_per_page"], params["num_kv_heads"], params["head_dim"]),
        dtype=params["dtype"],
    )

    # Page table: [num_sequences_in_batch, num_logical_pages_per_seq]
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
        tokens_per_seq_in_map = num_tokens // params["num_sequences_in_batch"]
        py_query_to_seq_map = mx.repeat(
            mx.arange(params["num_sequences_in_batch"], dtype=mx.int32), repeats=tokens_per_seq_in_map
        )

    # Query token offset: [num_tokens]
    py_query_token_offset = mx.array([params["seq_len"]] * num_tokens, dtype=mx.int32)

    # Warm-up
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

    # --- Sweep: Varying seq_len ---
    seq_len_values = [64, 128, 256, 512, 1024, 2048]

    for sl_val in seq_len_values:
        current_params = baseline_config.copy()
        current_params["seq_len"] = sl_val

        try:
            avg_time_ms = run_latency_benchmark(current_params)

            # Print CSV-formatted result
            print(
                f"{current_params['num_query_items']},{current_params['num_q_heads']},{current_params['num_kv_heads']},{current_params['head_dim']},{current_params['seq_len']},{current_params['tokens_per_page']},{current_params['num_sequences_in_batch']},{avg_time_ms:.4f}"
            )
        except Exception:
            print(
                f"{current_params['num_query_items']},{current_params['num_q_heads']},{current_params['num_kv_heads']},{current_params['head_dim']},{current_params['seq_len']},{current_params['tokens_per_page']},{current_params['num_sequences_in_batch']},ERROR"
            )
