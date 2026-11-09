import math

import torch


def neighborhood_attention2d(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kernel_size: int | tuple[int, int] = 9,
    scale: float | None = None,
    *,
    fp32_precision: str = "tf32x3",
) -> torch.Tensor:
    """Shifted-window attention on tensors shaped (batch, height, width, heads, head_dim).

    Boundary windows shift inward to retain the full neighborhood, matching NATTEN.
    Inputs must share shape, CUDA or MPS device and dtype (FP16, BF16 or FP32). Arbitrary
    strides are accepted. Head dimensions are 16/32/64/128; neighborhoods have odd
    dimensions from 3 through 13. Dilation and stride are 1; attention is noncausal.
    The output has the same shape and dtype as the inputs and is contiguous.

    ``scale=None`` means ``head_dim**-0.5``; zero and negative scales are supported.
    FP32 defaults to ``"tf32x3"``; ``fp32_precision`` also accepts ``"ieee"``,
    ``"tf32"``, ``"bf16x3"`` and ``"bf16x6"``, independently of Torch's TF32 flags.
    This setting has no effect on FP16/BF16 or MPS. MPS uses FP32 accumulation
    and supports forward only. On CUDA, first-order gradients are supported;
    backward uses nondeterministic FP32 atomic accumulation.
    Spatial dimensions remain runtime inputs under ``torch.compile``.
    Inputs whose flattened offsets could exceed signed 32-bit indexing are
    rejected, including a conservative allowance for masked tile lanes.
    """
    if any(t.ndim != 5 for t in (q, k, v)) or q.shape != k.shape or q.shape != v.shape:
        raise ValueError(
            "q, k and v must have the same (batch, height, width, heads, head_dim) shape"
        )
    if q.device != k.device or q.device != v.device:
        raise ValueError("q, k and v must be on the same CUDA or MPS device")
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise ValueError("q, k and v must have the same dtype")
    if q.device.type not in ("cuda", "mps") or q.dtype not in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
    ):
        raise ValueError(
            "neighborhood_attention2d requires CUDA or MPS FP16, BF16 or FP32 tensors"
        )
    if q.shape[-1] not in (16, 32, 64, 128):
        raise ValueError("head_dim must be 16, 32, 64 or 128")
    if q.shape[0] == 0 or q.shape[-2] == 0:
        raise ValueError("batch and heads must be nonempty")
    kernel = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
    if len(kernel) != 2 or any(
        not isinstance(n, int) or n < 3 or n > 13 or n % 2 == 0 for n in kernel
    ):
        raise ValueError("kernel_size must contain two odd integers from 3 through 13")
    kh, kw = kernel
    if q.shape[1] < kh or q.shape[2] < kw:
        raise ValueError("the neighborhood cannot exceed the input size")
    if fp32_precision not in ("tf32x3", "ieee", "tf32", "bf16x3", "bf16x6"):
        raise ValueError(
            'fp32_precision must be "tf32x3", "ieee", "tf32", "bf16x3" or "bf16x6"'
        )
    if scale is not None and not math.isfinite(scale):
        raise ValueError("scale must be finite")

    b, h, w, heads, dim = q.shape
    # Masked lanes also form offsets; tiles/chunks contain at most 128 lanes.
    if (b * h * w + 128 * (w + 1)) * heads * dim >= 2**31:
        raise ValueError(
            "input exceeds the safe 32-bit indexing limit for neighborhood attention"
        )

    if q.device.type == "mps":
        from ._neighborhood_attention_metal import _forward as metal_forward

        return metal_forward(
            q, k, v, kh, kw, q.shape[-1] ** -0.5 if scale is None else scale
        )

    from ._neighborhood_attention_triton import _forward

    return _forward(
        q.contiguous(),
        k.contiguous(),
        v.contiguous(),
        kh,
        kw,
        q.shape[-1] ** -0.5 if scale is None else scale,
        fp32_precision,
    )[0]
