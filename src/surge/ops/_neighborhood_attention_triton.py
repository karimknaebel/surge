import os

import torch
import triton
import triton.language as tl

os.environ.setdefault("TRITON_DEFAULT_FP_FUSION", "0")


@triton.jit
def _coordinates(
    H,
    W,
    KH: tl.constexpr,
    KW: tl.constexpr,
    TH: tl.constexpr,
    TW: tl.constexpr,
    N: tl.constexpr,
    offset=0,
):
    tile = tl.program_id(0)
    y, x = tile // tl.cdiv(W, TW) * TH, tile % tl.cdiv(W, TW) * TW
    m = tl.arange(0, TH * TW)
    qy, qx = y + m // TW, x + m % TW
    # Cover the union of shifted neighborhoods, including windows at image boundaries.
    hy = tl.minimum(tl.maximum(y - KH // 2, 0), tl.maximum(H - TH - KH + 1, 0))
    hx = tl.minimum(tl.maximum(x - KW // 2, 0), tl.maximum(W - TW - KW + 1, 0))
    n = tl.arange(0, N) + offset
    ky, kx = hy + n // (TW + KW - 1), hx + n % (TW + KW - 1)
    valid_q = (qy < H) & (qx < W)
    valid_k = (n < (TH + KH - 1) * (TW + KW - 1)) & (ky < H) & (kx < W)
    sy = tl.minimum(tl.maximum(qy - KH // 2, 0), H - KH)
    sx = tl.minimum(tl.maximum(qx - KW // 2, 0), W - KW)
    mask = (
        valid_k[None, :]
        & (ky[None, :] >= sy[:, None])
        & (ky[None, :] < sy[:, None] + KH)
        & (kx[None, :] >= sx[:, None])
        & (kx[None, :] < sx[:, None] + KW)
    )
    return qy * W + qx, ky * W + kx, valid_q, valid_k, mask


@triton.jit(do_not_specialize=["H", "W"])
def _forward_kernel(
    Q,
    K,
    V,
    O,
    L,
    H,
    W,
    HEADS: tl.constexpr,
    D: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    SCALE: tl.constexpr,
    TH: tl.constexpr,
    TW: tl.constexpr,
    BLOCK_N: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
):
    bh = tl.program_id(1)
    batch, head = bh // HEADS, bh % HEADS
    qi, ki, valid_q, valid_k, mask = _coordinates(H, W, KH, KW, TH, TW, BLOCK_N)
    d = tl.arange(0, D)
    qo = ((batch * H * W + qi[:, None]) * HEADS + head) * D + d[None, :]
    q = tl.load(Q + qo, valid_q[:, None], 0)
    maximum = tl.full((TH * TW,), float("-inf"), tl.float32)
    normalizer = tl.full((TH * TW,), 0, tl.float32)
    acc = tl.full((TH * TW, D), 0, tl.float32)
    for start in range(0, (TH + KH - 1) * (TW + KW - 1), BLOCK_N):
        qi, ki, valid_q, valid_k, mask = _coordinates(
            H, W, KH, KW, TH, TW, BLOCK_N, start
        )
        ko = ((batch * H * W + ki[:, None]) * HEADS + head) * D + d[None, :]
        k = tl.load(K + ko, valid_k[:, None], 0)
        v = tl.load(V + ko, valid_k[:, None], 0)
        scores = tl.dot(q, tl.trans(k), input_precision=INPUT_PRECISION) * (
            SCALE * 1.4426950408889634
        )
        scores = tl.where(mask, scores, float("-inf"))
        new_maximum = tl.maximum(maximum, tl.max(scores, 1))
        safe_maximum = tl.where(new_maximum == float("-inf"), 0, new_maximum)
        alpha = tl.exp2(maximum - safe_maximum)
        p = tl.exp2(scores - safe_maximum[:, None])
        acc = acc * alpha[:, None] + tl.dot(
            p.to(q.dtype), v, input_precision=INPUT_PRECISION
        )
        normalizer = normalizer * alpha + tl.sum(p, 1)
        maximum = new_maximum
    tl.store(O + qo, acc / normalizer[:, None], valid_q[:, None])
    tl.store(L + bh * H * W + qi, maximum + tl.log2(normalizer), valid_q)


@triton.jit(do_not_specialize=["H", "W"])
def _backward_kernel(
    Q,
    K,
    V,
    O,
    DO,
    L,
    DQ,
    DK,
    DV,
    H,
    W,
    HEADS: tl.constexpr,
    D: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    SCALE: tl.constexpr,
    TH: tl.constexpr,
    TW: tl.constexpr,
    BLOCK_N: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
):
    bh = tl.program_id(1)
    batch, head = bh // HEADS, bh % HEADS
    qi, ki, valid_q, valid_k, mask = _coordinates(H, W, KH, KW, TH, TW, BLOCK_N)
    d = tl.arange(0, D)
    qo = ((batch * H * W + qi[:, None]) * HEADS + head) * D + d[None, :]
    q = tl.load(Q + qo, valid_q[:, None], 0)
    o = tl.load(O + qo, valid_q[:, None], 0)
    do = tl.load(DO + qo, valid_q[:, None], 0)
    l = tl.load(L + bh * H * W + qi, valid_q, float("inf"))
    delta = tl.sum(o.to(tl.float32) * do.to(tl.float32), 1)
    dq = tl.full((TH * TW, D), 0, tl.float32)
    for start in range(0, (TH + KH - 1) * (TW + KW - 1), BLOCK_N):
        qi, ki, valid_q, valid_k, mask = _coordinates(
            H, W, KH, KW, TH, TW, BLOCK_N, start
        )
        ko = ((batch * H * W + ki[:, None]) * HEADS + head) * D + d[None, :]
        k = tl.load(K + ko, valid_k[:, None], 0)
        v = tl.load(V + ko, valid_k[:, None], 0)
        scores = tl.dot(q, tl.trans(k), input_precision=INPUT_PRECISION) * (
            SCALE * 1.4426950408889634
        )
        p = tl.where(mask & valid_q[:, None], tl.exp2(scores - l[:, None]), 0.0)
        dp = tl.dot(do, tl.trans(v), input_precision=INPUT_PRECISION)
        ds = (p * (dp - delta[:, None]) * SCALE).to(q.dtype)
        dq += tl.dot(ds, k, input_precision=INPUT_PRECISION)
        dk = tl.dot(tl.trans(ds), q, input_precision=INPUT_PRECISION)
        dv = tl.dot(tl.trans(p.to(q.dtype)), do, input_precision=INPUT_PRECISION)
        # Overlapping query tiles accumulate key/value gradients in FP32.
        tl.atomic_add(DK + ko, dk, valid_k[:, None], sem="relaxed")
        tl.atomic_add(DV + ko, dv, valid_k[:, None], sem="relaxed")
    tl.store(DQ + qo, dq, valid_q[:, None])


# Hopper-tuned (query height, query width, key chunk, warps); also fit SM86 shared memory.
_CONFIGS = {
    (torch.bfloat16, 16, 9): ((8, 8, 128, 4), (8, 8, 64, 4)),
    (torch.bfloat16, 32, 9): ((8, 8, 64, 4), (8, 8, 64, 4)),
    (torch.bfloat16, 64, 3): ((4, 8, 64, 4), (8, 8, 128, 4)),
    (torch.bfloat16, 64, 9): ((8, 8, 64, 4), (8, 8, 64, 4)),
    (torch.bfloat16, 64, 13): ((8, 8, 64, 4), (8, 8, 32, 4)),
    (torch.bfloat16, 128, 3): ((4, 8, 64, 4), (8, 8, 32, 4)),
    (torch.bfloat16, 128, 9): ((8, 8, 128, 4), (8, 8, 32, 4)),
    (torch.bfloat16, 128, 13): ((8, 8, 32, 4), (8, 8, 32, 4)),
    (torch.float16, 16, 9): ((8, 8, 64, 4), (8, 8, 64, 4)),
    (torch.float16, 32, 9): ((8, 8, 128, 4), (8, 8, 64, 4)),
    (torch.float16, 64, 9): ((8, 8, 128, 4), (8, 8, 64, 4)),
    (torch.float16, 128, 9): ((8, 8, 128, 4), (8, 8, 32, 4)),
    (torch.float32, 16, 9): ((8, 8, 32, 4), (8, 8, 64, 4)),
    (torch.float32, 32, 9): ((8, 8, 64, 4), (8, 8, 64, 4)),
    (torch.float32, 64, 3): ((4, 4, 32, 2), (4, 4, 32, 4)),
    (torch.float32, 64, 9): ((4, 4, 32, 2), (4, 8, 64, 4)),
    (torch.float32, 64, 13): ((4, 8, 64, 4), (4, 4, 32, 4)),
    (torch.float32, 128, 3): ((4, 8, 32, 4), (4, 4, 32, 4)),
    (torch.float32, 128, 9): ((4, 8, 64, 4), (4, 4, 32, 4)),
    (torch.float32, 128, 13): ((4, 4, 64, 4), (4, 4, 32, 4)),
}


def _launch_configs(dtype, dim, kh, kw):
    if kh == kw and (dtype, dim, kh) in _CONFIGS:
        return _CONFIGS[dtype, dim, kh]
    if dtype == torch.float32:
        return (4, 4, 32, 4), (4, 4, 32, 4)
    return (
        ((8, 8, 32, 4), (8, 8, 32, 4)) if dim == 128 else ((8, 8, 64, 4), (8, 8, 64, 4))
    )


@torch.library.custom_op("surge::neighborhood_attention2d_forward", mutates_args=())
def _forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kh: int,
    kw: int,
    scale: float,
    fp32_precision: str = "tf32x3",
) -> tuple[torch.Tensor, torch.Tensor]:
    b, h, w, heads, dim = q.shape
    out = torch.empty_like(q)
    lse = torch.empty((b, heads, h, w), device=q.device, dtype=torch.float32)
    th, tw, chunk, warps = _launch_configs(q.dtype, dim, kh, kw)[0]
    with torch.cuda.device(q.device):
        _forward_kernel[(triton.cdiv(h, th) * triton.cdiv(w, tw), b * heads)](
            q,
            k,
            v,
            out,
            lse,
            h,
            w,
            heads,
            dim,
            kh,
            kw,
            scale,
            th,
            tw,
            chunk,
            fp32_precision,
            num_warps=warps,
            num_stages=1,
        )
    return out, lse


@_forward.register_fake
def _(q, k, v, kh, kw, scale, fp32_precision="tf32x3"):
    b, h, w, heads, _ = q.shape
    return torch.empty_like(q), q.new_empty((b, heads, h, w), dtype=torch.float32)


@torch.library.custom_op("surge::neighborhood_attention2d_backward", mutates_args=())
def _backward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    grad: torch.Tensor,
    lse: torch.Tensor,
    kh: int,
    kw: int,
    scale: float,
    fp32_precision: str = "tf32x3",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    b, h, w, heads, dim = q.shape
    dq = torch.empty_like(q)
    dk = torch.zeros(q.shape, device=q.device, dtype=torch.float32)
    dv = torch.zeros_like(dk)
    th, tw, chunk, warps = _launch_configs(q.dtype, dim, kh, kw)[1]
    with torch.cuda.device(q.device):
        _backward_kernel[(triton.cdiv(h, th) * triton.cdiv(w, tw), b * heads)](
            q,
            k,
            v,
            out,
            grad.contiguous(),
            lse,
            dq,
            dk,
            dv,
            h,
            w,
            heads,
            dim,
            kh,
            kw,
            scale,
            th,
            tw,
            chunk,
            fp32_precision,
            num_warps=warps,
            num_stages=1,
        )
    return dq, dk.to(q.dtype), dv.to(q.dtype)


@_backward.register_fake
def _(q, k, v, out, grad, lse, kh, kw, scale, fp32_precision="tf32x3"):
    return torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)


def _setup_context(ctx, inputs, output):
    q, k, v, *config = inputs
    out, lse = output
    ctx.save_for_backward(q, k, v, out, lse)
    ctx.config = config
    ctx.mark_non_differentiable(lse)


def _autograd(ctx, grad, grad_lse):
    q, k, v, out, lse = ctx.saved_tensors
    return *_backward(q, k, v, out, grad, lse, *ctx.config), None, None, None, None


_forward.register_autograd(_autograd, setup_context=_setup_context)
