from functools import cache
from pathlib import Path

import torch


@cache
def _kernel(dtype, dim, kh, kw, th):
    scalar = {torch.float16: "half", torch.bfloat16: "bfloat", torch.float32: "float"}[
        dtype
    ]
    return torch.mps.compile_shader(
        f"#define D {dim}\n#define KH {kh}\n#define KW {kw}\n#define T {scalar}\n#define TH {th}\n"
        + f"#define S {'half' if dtype == torch.float16 else 'float'}\n"
        + Path(__file__).with_suffix(".metal").read_text()
    ).attention


@torch.library.custom_op("surge::neighborhood_attention2d_mps", mutates_args=())
def _forward(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, kh: int, kw: int, scale: float
) -> torch.Tensor:
    b, h, w, heads, dim = q.shape
    out = torch.empty(q.shape, device=q.device, dtype=q.dtype)
    th = 2 if dim <= 64 else 1
    _kernel(q.dtype, dim, kh, kw, th)(
        q,
        k,
        v,
        out,
        h,
        w,
        heads,
        scale,
        q.stride(),
        k.stride(),
        v.stride(),
        threads=(128 * ((w + 7) // 8), (h + th - 1) // th, b * heads),
        group_size=(128, 1, 1),
    )
    return out


@_forward.register_fake
def _(q, k, v, kh, kw, scale):
    return torch.empty(q.shape, device=q.device, dtype=q.dtype)


def _backward(ctx, grad):
    raise RuntimeError(
        "MPS neighborhood attention supports forward only; use CUDA for backward"
    )


_forward.register_autograd(_backward)
