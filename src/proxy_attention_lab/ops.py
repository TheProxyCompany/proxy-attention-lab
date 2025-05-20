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
"""Operations for paged attention in MLX."""

import mlx.core as mx

from proxy_attention_lab.pal_core import paged_attention as cpp_paged_attention_kernel


def paged_attention(
    queries: mx.array,
    k_cache_pool: mx.array,
    v_cache_pool: mx.array,
    page_table: mx.array,
    sequence_lengths: mx.array,
    query_to_seq_map: mx.array,
    query_token_offset: mx.array,
    is_prefill: bool = True,
    stream: mx.Stream | mx.Device | None = None,
) -> mx.array:
    """Performs paged attention using the custom C++ primitive and Metal kernel.

    Args:
        queries: Queries array. May be 1D, 2D, or 3D:
            - 1D: [NumItems] with HeadDim=1
            - 2D: [NumItems, HeadDim] (NumQHeads implicitly 1)
            - 3D: [NumTokens, NumQHeads, HeadDim]
        k_cache_pool: The entire K cache buffer.
            Shape: [NumTotalPages, TokensPerPage, NumKVHeads, HeadDim]
        v_cache_pool: The entire V cache buffer.
            Shape: [NumTotalPages, TokensPerPage, NumKVHeads, HeadDim]
        page_table: Page table mapping logical blocks for each sequence
            to physical page IDs in the k_cache_pool/v_cache_pool.
            Shape: [NumSequencesInBatch, MaxLogicalBlocksPerSequence]
        sequence_lengths: Actual length of each sequence in the batch.
            Shape: [NumSequencesInBatch]
        query_to_seq_map: Maps each query token to its sequence index.
            Shape: [TotalQueryTokens]
        query_token_offset: Logical offset of each query token within its sequence.
            Shape: [TotalQueryTokens]
        is_prefill: Whether to perform prefill or decoding.
            - When True (prefill mode): One threadgroup processes all heads for a query token
            - When False (decode mode): One threadgroup processes one query-token-head pair
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
        query_to_seq_map,
        query_token_offset,
        is_prefill=is_prefill,
        stream=stream,
    )
