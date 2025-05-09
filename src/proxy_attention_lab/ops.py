import mlx.core as mx

from proxy_attention_lab.pal_core import paged_attention as cpp_paged_attention_kernel


def paged_attention(
    queries: mx.array,
    kv_cache: mx.array,
    page_table: mx.array,
    stream: mx.Stream | mx.Device | None = None,
) -> int:
    """
    Performs paged attention using the custom C++ primitive and Metal kernel.

    Args:
        queries (mlx.core.array): Queries array.
        kv_cache (mlx.core.array): The entire KV cache buffer.
        page_table (mlx.core.array): Page table mapping logical blocks for each sequence
                                   to physical pages in kv_cache.
        stream (mlx.core.Stream | mx.core.Device | None): Stream or device.

    Returns:
        mlx.core.array: The attention output array.
    """
    # --- Call the C++ Bound Operation ---
    # Pass arguments directly. Nanobind handles type conversions.
    # The C++ operation `pal::cpp::paged_attention` creates the primitive
    # and returns the output array object, adding it to the MLX graph.
    output_array = cpp_paged_attention_kernel(
        queries,
        kv_cache,
        page_table,
        stream=stream,
    )
    return output_array
