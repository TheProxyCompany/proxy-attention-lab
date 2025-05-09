import logging
import time

import mlx.core as mx

from proxy_attention_lab import paged_attention

logger = logging.getLogger(__name__)


def test_forward():
    """
    Robust smoke test for paged_attention forward pass with various shapes and dtypes.
    """
    # --- Mock Inputs ---
    mock_queries = mx.arange(0, 16, dtype=mx.float16).reshape((4, 4))
    mock_kv_cache = mx.ones_like(mock_queries)
    mock_page_table = mx.zeros((4,), dtype=mx.uint32)

    # --- Run C++ Kernel (without eval) ---
    start = time.time()
    out = paged_attention(
        mock_queries,
        mock_kv_cache,
        mock_page_table,
    )
    mid = time.time()
    raw_cpp_time = mid - start
    logger.info("Time for C++: %s", raw_cpp_time)

    # --- Evaluate Output ---
    mx.eval(out)
    end = time.time()
    overall_eval_time = end - mid
    logger.info("Time for eval: %s", overall_eval_time)

    assert out.shape == mock_queries.shape
    assert out.dtype == mock_queries.dtype
