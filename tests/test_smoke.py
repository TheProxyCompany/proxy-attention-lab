import mlx.core as mx
from proxy_attention_lab.ops import paged_attention


def test_forward():
    q = mx.arange(16, dtype=mx.bfloat16).reshape(4, 4)
    kv_cache = mx.ones_like(q)
    page_table = mx.zeros((4,), dtype=mx.uint32)
    out = paged_attention(
        q,
        kv_cache,
        page_table,
        stream=None,
    )
    mx.eval(out)
    assert mx.max(mx.abs(out - (q + kv_cache))) < 1e-3
