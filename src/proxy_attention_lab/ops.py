import mlx.core as mx

from proxy_attention_lab.pal_core import paged_attention as cpp_paged_attention_kernel


def paged_attention(
    queries: mx.array,
    # kv_cache: mx.array,
    # page_table: mx.array,
    # stream: mx.Stream | mx.Device | None = None,
) -> int:
    """
    Performs paged attention using the custom C++ primitive and Metal kernel.

    Args:
        queries (mlx.core.array): Queries array. Shape depends on batching,
                                 e.g., [batch_size, num_heads, seq_len, head_dim] or flattened.
        kv_cache (mlx.core.array): The entire KV cache buffer allocated by the PageAllocator.
                                 Shape e.g., [num_pages, tokens_per_page, 2, num_kv_heads, head_dim].
        page_table (mlx.core.array): Page table mapping logical blocks for each sequence
                                   to physical pages in kv_cache. Shape depends on kernel design.
        stream (mlx.core.Stream | mx.core.Device | None, optional): Stream or device to evaluate on.
                                                                    Defaults to None (MLX default).

    Returns:
        mlx.core.array: The attention output array. Shape likely matches 'q' or a derivative.
    """
    # --- Call the C++ Bound Operation ---
    # Pass arguments directly. Nanobind handles type conversions.
    # The C++ operation `pal::cpp::paged_attention` creates the primitive
    # and returns the output array object, adding it to the MLX graph.
    # output_array = cpp_paged_attention_kernel(
    #     queries,
    #     kv_cache,
    #     page_table,
    #     # stream=stream,
    # )
    breakpoint()
    test = cpp_paged_attention_kernel(queries)
    print(test)
    return test
