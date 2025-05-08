import mlx.core as mx
from lab.ops import paged_attention


def test_forward():
    q = mx.arange(16, dtype=mx.float16).reshape(4, 4)
    kv = mx.ones_like(q)
    tbl = mx.zeros((4,), dtype=mx.uint32)
    out = paged_attention(q, kv, tbl)
    mx.eval(out)
    assert mx.max(mx.abs(out - (q + kv))) < 1e-3
