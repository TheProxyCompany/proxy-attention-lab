import mlx.core as mx
from src.python.ops import paged_attention


def test_forward():
    q = mx.arange(16, dtype=mx.float16).reshape(4, 4)
    kv_cache = mx.ones_like(q)
    page_table = mx.zeros((4,), dtype=mx.uint32)
    out = paged_attention(q, kv_cache, page_table)
    mx.eval(out)
    assert mx.max(mx.abs(out - (q + kv_cache))) < 1e-3
