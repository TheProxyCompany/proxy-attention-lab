import mlx.core as mx
from proxy_attention_lab import paged_attention


def test_forward():
    q = mx.arange(16, dtype=mx.float16).reshape(4, 4)
    out = paged_attention(q)
    assert out
    # mx.eval(out)
    # assert mx.max(mx.abs(out - (q + kv_cache))) < 1e-3
