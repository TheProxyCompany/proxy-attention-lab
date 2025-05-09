import mlx.core as mx

from proxy_attention_lab import paged_attention


def test_forward():
    mock_queries = mx.arange(16, dtype=mx.float16).reshape(4, 4)
    mock_kv_cache = mx.ones_like(mock_queries)
    mock_page_table = mx.zeros((4,), dtype=mx.uint32)

    out = paged_attention(
        mock_queries,
        mock_kv_cache,
        mock_page_table,
    )
    mx.eval(out)
    assert mx.max(mx.abs(out - (mock_queries + mock_kv_cache))) < 1e-3
