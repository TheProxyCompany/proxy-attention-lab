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
    stream: mx.Stream | mx.Device | None = None,
) -> mx.array:
    """
    Performs paged attention using the custom C++ primitive and Metal kernel.

    Args:
        queries (mlx.core.array): Queries array.
            Shape: e.g., [TotalQueryTokens, NumQHeads, HeadDim] or [TotalQueryTokens, ModelDim]
        k_cache_pool (mlx.core.array): The entire K cache buffer.
            Shape: e.g., [NumTotalPages, TokensPerPage, NumKVHeads, HeadDim]
        v_cache_pool (mlx.core.array): The entire V cache buffer.
            Shape: e.g., [NumTotalPages, TokensPerPage, NumKVHeads, HeadDim]
        page_table (mlx.core.array): Page table mapping logical blocks for each sequence
            to physical page IDs in the k_cache_pool/v_cache_pool.
            Shape: e.g., [NumSequencesInBatch, MaxLogicalBlocksPerSequence] (flattened or 2D)
        sequence_lengths (mlx.core.array): Actual length of each sequence in the batch.
            Shape: e.g., [NumSequencesInBatch]
        query_to_seq_map (mlx.core.array): Maps each query token to its sequence index.
            Shape: e.g., [TotalQueryTokens]
        query_token_offset (mlx.core.array): Logical offset of each query token within its sequence.
            Shape: e.g., [TotalQueryTokens]
        stream (mlx.core.Stream | mx.core.Device | None): Stream or device.

    Returns:
        mlx.core.array: The attention output array.
    """
    output_array = cpp_paged_attention_kernel(
        queries,
        k_cache_pool,
        v_cache_pool,
        page_table,
        sequence_lengths,
        query_to_seq_map,
        query_token_offset,
        stream=stream,
    )
    return output_array
