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
"""Operations for paged attention in MLX."""

import mlx.core as mx

from proxy_attention_lab.pal_core import (
    fill_kv_pages as cpp_fill_kv_pages_kernel,
)
from proxy_attention_lab.pal_core import (
    get_k_cache_stripe_size as cpp_get_k_cache_stripe_size,
)
from proxy_attention_lab.pal_core import (
    paged_attention as cpp_paged_attention_kernel,
)


def get_k_cache_stripe_size(dtype: mx.Dtype) -> int:
    """Calculates the stripe size for the K cache.

    Args:
        dtype: The data type of the K cache.
    """
    return cpp_get_k_cache_stripe_size(dtype)


def paged_attention(
    queries: mx.array,
    k_cache_pool: mx.array,
    v_cache_pool: mx.array,
    page_table: mx.array,
    sequence_lengths: mx.array,
    stream: mx.Stream | mx.Device | None = None,
) -> mx.array:
    """Performs paged attention using the custom C++ primitive and Metal kernel.

    Args:
        queries: Queries array. May be 1D, 2D, or 3D:
            - 1D: [NumItems] with HeadDim=1
            - 2D: [NumItems, HeadDim] (NumQHeads implicitly 1)
            - 3D: [NumTokens, NumQHeads, HeadDim]
        k_cache_pool: The entire K cache buffer.
            Shape: [NumTotalPages, NumKVHeads, NumTokensPerPage, HeadDim]
        v_cache_pool: The entire V cache buffer.
            Shape: [NumTotalPages, NumKVHeads, NumTokensPerPage, HeadDim]
        page_table: Page table mapping logical blocks for each sequence
            to physical page IDs in the k_cache_pool/v_cache_pool.
            Shape: [NumSequencesInBatch, MaxLogicalBlocksPerSequence]
        sequence_lengths: Actual length of each sequence in the batch.
            Shape: [NumSequencesInBatch]
        stream: Optional stream or device for the operation.

    Returns:
        mx.array: The result of the paged attention operation:
            - If queries are 3D [NumTokens, NumQHeads, HeadDim], output is [NumTokens*NumQHeads, HeadDim]
            - If queries are 2D [NumItems, HeadDim], output is [NumItems, HeadDim]
            - If queries are 1D [NumItems], output is [NumItems, HeadDim]

    Note:
        The output HeadDim is always taken from the KV cache head dimension, regardless of query dimensions.
    """
    return cpp_paged_attention_kernel(
        queries,
        k_cache_pool,
        v_cache_pool,
        page_table,
        sequence_lengths,
        stream=stream,
    )


def fill_kv_pages(
    new_keys: mx.array,
    new_values: mx.array,
    global_key_pool: mx.array,
    global_value_pool: mx.array,
    page_table: mx.array,
    current_token_write_positions: mx.array,
    query_to_seq_map: mx.array,
    stream: mx.Stream | mx.Device | None = None,
) -> tuple[mx.array, mx.array]:
    """Fills the KV cache pages with the new keys and values.

    Args:
        new_keys: The new keys to fill the KV cache with.
        new_values: The new values to fill the KV cache with.
        global_key_pool: The global key pool to fill the KV cache with.
        global_value_pool: The global value pool to fill the KV cache with.
        page_table: Page table mapping logical blocks for each sequence
            to physical page IDs in the k_cache_pool/v_cache_pool.
            Shape: [NumSequencesInBatch, MaxLogicalBlocksPerSequence]
        current_token_write_positions: Logical token index within its sequence where the new K/V should be written
            Shape: [TotalCurrentTokensInBatch]
        query_to_seq_map: Maps each query token to its sequence index.
            Shape: [TotalQueryTokens]
        stream: Optional stream or device for the operation.

    Returns:
        tuple[mx.array, mx.array]: The updated global key pool and global value pool.
    """
    return cpp_fill_kv_pages_kernel(
        new_keys,
        new_values,
        global_key_pool,
        global_value_pool,
        page_table,
        current_token_write_positions,
        query_to_seq_map,
        stream=stream,
    )
