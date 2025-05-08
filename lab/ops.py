
from pathlib import Path

import mlx.core as mx

_METALLIB = Path(__file__).with_name("attn.metallib")

def _kernel():
    src = """
    extern "C" kernel void paged_attn(
        device const half* q      [[buffer(0)]],
        device const half* kv     [[buffer(1)]],
        device const uint* tbl    [[buffer(2)]],
        device half* out          [[buffer(3)]],
        uint tid                  [[thread_position_in_grid]])
    {
        //  **stub**  – replace with real math soon
        out[tid] = q[tid] + kv[tid];
    }
    """
    return mx.fast.metal_kernel(
        name="paged_attn",
        input_names=["q", "kv", "tbl"],
        output_names=["out"],
        source=src,
    )


_paged_attn = _kernel()


def paged_attention(
    q: mx.array,
    kv: mx.array,
    ptable: mx.array,
) -> mx.array:
    return _paged_attn(q, kv, ptable)[0]
