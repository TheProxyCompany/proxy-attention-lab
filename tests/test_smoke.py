import logging
import time

import mlx.core as mx
from proxy_attention_lab import paged_attention

logger = logging.getLogger(__name__)


def test_forward():
    """Smoke test for paged_attention forward pass."""

    # --- Mock Inputs ---
    mock_queries = mx.arange(16, dtype=mx.float16).reshape(4, 4)
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
    logger.info(f"Time taken for C++ kernel (no eval yet): {raw_cpp_time:.6f} seconds")

    # --- Evaluate Output ---
    mx.eval(out)
    end = time.time()
    overall_eval_time = end - mid
    logger.info(f"Time taken for eval: {overall_eval_time:.6f} seconds")

    # --- Assert Output Correctness ---
    expected = mock_queries + mock_kv_cache
    max_diff = mx.max(mx.abs(out - expected))
    assert max_diff < 1e-3, f"Max difference {max_diff} exceeds tolerance"
