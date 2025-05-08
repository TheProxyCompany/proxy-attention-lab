
# src/python/ops.py
import mlx.core as mx
# Import the C++ binding module built by CMake/Nanobind
# The name '_pal_cpp_binding' must match NB_MODULE in binding.cpp
# and the target name in CMakeLists.txt
from . import _pal_cpp_binding

def paged_attention(
    q: mx.array,
    kv_cache: mx.array, # Full KV cache buffer
    page_table: mx.array, # Logical block -> physical page mapping
    stream: mx.Stream | mx.Device | None = None # Optional stream/device
) -> mx.array:
    """
    Performs paged attention using the custom C++ primitive and Metal kernel.

    Args:
        q (mlx.core.array): Queries array. Shape depends on batching,
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
    # --- Input Validation (Optional but Recommended) ---
    # Example checks (adapt based on actual kernel requirements)
    if q.dtype not in [mx.float16, mx.bfloat16, mx.float32]:
         raise TypeError(f"Query dtype {q.dtype} not supported.")
    if kv_cache.dtype != q.dtype:
         raise TypeError(f"KV cache dtype {kv_cache.dtype} must match query dtype {q.dtype}.")
    if page_table.dtype != mx.uint32:
         raise TypeError(f"Page table dtype must be uint32, got {page_table.dtype}.")
    # Add shape checks as needed

    # --- Call the C++ Bound Operation ---
    # Pass arguments directly. Nanobind handles type conversions.
    # The C++ operation `pal::cpp::paged_attention` creates the primitive
    # and returns the output array object, adding it to the MLX graph.
    output_array = _pal_cpp_binding.paged_attention(
        q,
        kv_cache,
        page_table,
        stream=stream # Pass the stream/device context
    )

    return output_array
