import math
from typing import Callable

import torch
import torch.nn as nn
from natten.functional import neighborhood_attention_generic
from natten.utils.checks import check_all_args
from specbuild import REGISTRY
from timm.layers import LayerNorm, LayerScale, RmsNorm
from timm.layers.pos_embed_sincos import RotaryEmbeddingCat, apply_rot_embed_cat
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
)


@REGISTRY.register()
class NAD(nn.Module):
    def __init__(
        self,
        dim_in: list[int | None],
        dim_out: list[int | None],
        embed_dim: list[int] = [1024, 512, 256, 128, 64],
        depth: int | list[int] = 3,
        head_dim: int | list[int] = 64,
        mlp_ratio: float = 4.0,
        kernel_size: int | list[int] = 9,
        dilation: int | list[int] = 1,
        qk_norm: bool | list[bool] = True,
        rope: bool = True,
        rope_temperature_scale: float = 1.0 / math.pi,
        init_values: float | None | list[float | None] = None,
        input_init_values: float | None | list[float | None] = None,
        norm_layer: str | None = None,
        act_layer: str | Callable = "relu",
        upsample_mode: str | list[str] = "conv_transpose",
        upsample_refine_kernel_size: int | None | list[int | None] = 3,
        upsample_refine_padding_mode: str | list[str] = "zeros",
    ):
        super().__init__()

        num_stages = len(embed_dim)

        def repeat_stage_arg(value, length=num_stages):
            if isinstance(value, list):
                assert len(value) == length
                return value
            return [value] * length

        assert len(dim_in) == num_stages
        assert len(dim_out) == num_stages
        input_init_values = repeat_stage_arg(input_init_values)
        depth = repeat_stage_arg(depth)
        head_dim = repeat_stage_arg(head_dim)
        kernel_size = repeat_stage_arg(kernel_size)
        dilation = repeat_stage_arg(dilation)
        init_values = repeat_stage_arg(init_values)
        qk_norm = repeat_stage_arg(qk_norm)
        upsample_mode = repeat_stage_arg(upsample_mode, num_stages - 1)
        upsample_refine_padding_mode = repeat_stage_arg(
            upsample_refine_padding_mode, num_stages - 1
        )
        upsample_refine_kernel_size = repeat_stage_arg(
            upsample_refine_kernel_size, num_stages - 1
        )

        self.stages = nn.ModuleList(
            [
                NADStage(
                    dim_in=dim_in[i],
                    dim_out=dim_out[i],
                    embed_dim=embed_dim[i],
                    depth=depth[i],
                    head_dim=head_dim[i],
                    kernel_size=kernel_size[i],
                    dilation=dilation[i],
                    qk_norm=qk_norm[i],
                    init_values=init_values[i],
                    input_init_values=input_init_values[i],
                    mlp_ratio=mlp_ratio,
                    rope=rope,
                    rope_temperature_scale=rope_temperature_scale,
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    next_embed_dim=embed_dim[i + 1] if i < num_stages - 1 else None,
                    upsample_mode=upsample_mode[i] if i < num_stages - 1 else None,
                    upsample_refine_kernel_size=upsample_refine_kernel_size[i]
                    if i < num_stages - 1
                    else None,
                    upsample_refine_padding_mode=upsample_refine_padding_mode[i]
                    if i < num_stages - 1
                    else None,
                )
                for i in range(num_stages)
            ]
        )

    def enable_gradient_checkpointing(self):
        for stage in self.stages:
            stage.enable_gradient_checkpointing()

    def forward(self, in_features: list[torch.Tensor]):
        first_feature = next(x for x in in_features if x is not None)
        batch_shape = first_feature.shape[:-3]
        in_features = [
            x.reshape(-1, *x.shape[-3:]).permute(0, 2, 3, 1).contiguous()
            if x is not None
            else None
            for x in in_features
        ]

        out_features = []
        x = None
        for stage, feature in zip(self.stages, in_features, strict=True):
            output, x = stage(feature, x)
            out_features.append(output)

        out_features = [
            x.permute(0, 3, 1, 2).unflatten(0, batch_shape) if x is not None else None
            for x in out_features
        ]
        return out_features


class NADStage(nn.Module):
    def __init__(
        self,
        dim_in: int | None,
        dim_out: int | None,
        embed_dim: int,
        next_embed_dim: int | None,
        depth: int,
        head_dim: int,
        mlp_ratio: float,
        kernel_size: int | tuple[int, int],
        dilation: int | tuple[int, int],
        qk_norm: bool,
        rope: bool,
        rope_temperature_scale: float,
        init_values: float | None,
        input_init_values: float | None,
        norm_layer: str | None,
        act_layer: str | Callable,
        upsample_mode: str | None,
        upsample_refine_kernel_size: int | None,
        upsample_refine_padding_mode: str | None,
    ):
        super().__init__()
        assert embed_dim % head_dim == 0

        kernel_size, _, dilation, _ = check_all_args(2, kernel_size, 1, dilation, False)
        if rope:
            effective_kernel_size = max(
                dilation_dim * (kernel_dim - 1) + 1
                for kernel_dim, dilation_dim in zip(kernel_size, dilation, strict=True)
            )
            rope_temperature = max(
                1.0, rope_temperature_scale * float(effective_kernel_size)
            )
        else:
            rope_temperature = None
        if rope_temperature is not None:
            assert head_dim % 4 == 0
        self.rope = (
            RotaryEmbeddingCat(
                dim=head_dim,
                temperature=rope_temperature,
                in_pixels=False,
            )
            if rope_temperature is not None
            else None
        )
        self.rope_half = getattr(self.rope, "rotate_half", False)

        self.input_projection = (
            nn.Linear(dim_in, embed_dim) if dim_in is not None else nn.Identity()
        )
        self.input_scale = (
            LayerScale(embed_dim, init_values=input_init_values)
            if input_init_values is not None
            else nn.Identity()
        )
        self.blocks = nn.Sequential(
            *(
                NADBlock(
                    embed_dim,
                    num_heads=embed_dim // head_dim,
                    mlp_ratio=mlp_ratio,
                    qk_norm=qk_norm,
                    init_values=init_values,
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    rope_half=self.rope_half,
                    kernel_size=kernel_size,
                    dilation=dilation,
                )
                for _ in range(depth)
            )
        )
        self.output_projection = (
            nn.Linear(embed_dim, dim_out) if dim_out is not None else None
        )
        self.upsample = (
            UpsampleBlock(
                embed_dim,
                next_embed_dim,
                scale_factor=2,
                mode=upsample_mode,
                refine_kernel_size=upsample_refine_kernel_size,
                refine_padding_mode=upsample_refine_padding_mode,
            )
            if upsample_mode is not None and next_embed_dim is not None
            else None
        )

    def enable_gradient_checkpointing(self):
        if self.upsample is not None:
            self.upsample = checkpoint_wrapper(self.upsample)
        for i in range(len(self.blocks)):
            self.blocks[i] = checkpoint_wrapper(self.blocks[i])

    def _get_rope_embed(self, x: torch.Tensor) -> torch.Tensor | None:
        if self.rope is None:
            return None
        height, width = x.shape[1:3]
        return self.rope.get_embed([height, width]).reshape(height, width, 1, -1)

    def forward(
        self,
        feature: torch.Tensor | None,
        x: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        feature = self.input_scale(self.input_projection(feature))
        if x is None:
            x = feature
        elif feature is not None:
            x = x + feature
        rope_embed = self._get_rope_embed(x) if len(self.blocks) > 0 else None
        for block in self.blocks:
            x = block(x, rope_embed)
        output = (
            self.output_projection(x) if self.output_projection is not None else None
        )
        if self.upsample is not None:
            x = self.upsample(x)
        return output, x


class NADBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        proj_drop: float = 0.0,
        init_values: float | None = None,
        act_layer: str | Callable = "relu",
        norm_layer: str | None = None,
        rope_half: bool = False,
        kernel_size: int | tuple[int, int] = 9,
        dilation: int | tuple[int, int] = 1,
    ):
        super().__init__()

        self.norm1 = _make_norm_layer(norm_layer, dim)
        self.attn = NeighborhoodAttention2d(
            dim,
            num_heads=num_heads,
            kernel_size=kernel_size,
            dilation=dilation,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            proj_drop=proj_drop,
            rope_half=rope_half,
        )
        self.ls1 = (
            LayerScale(dim, init_values=init_values)
            if init_values is not None
            else nn.Identity()
        )

        self.norm2 = _make_norm_layer(norm_layer, dim)
        hidden_channels = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_channels),
            _make_activation_layer(act_layer),
            nn.Linear(hidden_channels, dim),
        )
        self.ls2 = (
            LayerScale(dim, init_values=init_values)
            if init_values is not None
            else nn.Identity()
        )

    def forward(
        self,
        x: torch.Tensor,
        rope_embed: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.ls1(self.attn(self.norm1(x), rope_embed))
        x = x + self.ls2(self.mlp(self.norm2(x)))
        return x


# adjusted from https://github.com/SHI-Labs/NATTEN/blob/92750c3cf837652d58b6091e5ebd37dcad46e753/src/natten/modules.py#L42
class NeighborhoodAttention2d(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        kernel_size: int | tuple[int, int],
        dilation: int | tuple[int, int] = 1,
        qkv_bias: bool = True,
        qk_scale: float | None = None,
        qk_norm: bool = False,
        proj_drop: float = 0.0,
        rope_half: bool = False,
    ):
        super().__init__()
        kernel_size, _, dilation, is_causal = check_all_args(
            2, kernel_size, 1, dilation, False
        )
        assert embed_dim % num_heads == 0

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = qk_scale or self.head_dim**-0.5
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.is_causal = is_causal
        self.rope_half = rope_half
        self.q_norm = LayerNorm(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = LayerNorm(self.head_dim) if qk_norm else nn.Identity()
        self.qk_norm = qk_norm

        self.qkv = nn.Linear(self.embed_dim, self.embed_dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
        self,
        x: torch.Tensor,
        rope_embed: torch.Tensor | None = None,
    ) -> torch.Tensor:
        b, h, w, c = x.shape

        qkv = self.qkv(x).reshape(b, h, w, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=3)
        q = self.q_norm(q).to(v.dtype)
        k = self.k_norm(k).to(v.dtype)

        if rope_embed is not None:
            rope_embed = rope_embed.to(dtype=q.dtype)
            q = apply_rot_embed_cat(q, rope_embed, half=self.rope_half)
            k = apply_rot_embed_cat(k, rope_embed, half=self.rope_half)

        x = neighborhood_attention_generic(
            q,
            k,
            v,
            kernel_size=self.kernel_size,
            dilation=self.dilation,
            is_causal=self.is_causal,
            scale=self.scale,
        )
        x = x.reshape(b, h, w, c)

        x = self.proj_drop(self.proj(x))
        return x

    def extra_repr(self) -> str:
        return f"kernel_size={self.kernel_size}, dilation={self.dilation}"


class UpsampleBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mode: str,
        scale_factor: int = 2,
        refine_kernel_size: int | None = 3,
        refine_padding_mode: str = "zeros",
    ):
        super().__init__()
        if mode == "conv_transpose":
            self.resample = nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=scale_factor,
                stride=scale_factor,
            )
            with torch.no_grad():
                self.resample.weight[:] = self.resample.weight[:, :, :1, :1]
            resampled_channels = out_channels
        elif mode == "pixel_shuffle":
            # NOTE: this is effectively the same as conv_transpose (except for the bias)
            self.resample = PixelShuffle2d(in_channels, out_channels, scale_factor)
            resampled_channels = out_channels
        elif mode in ["nearest", "bilinear"]:
            self.resample = nn.Upsample(
                scale_factor=scale_factor,
                mode=mode,
                align_corners=False if mode == "bilinear" else None,
            )
            resampled_channels = in_channels
        else:
            raise ValueError(f"Unsupported upsample mode: {mode}")
        if refine_kernel_size is None:
            assert resampled_channels == out_channels
            self.refine = nn.Identity()
        else:
            self.refine = nn.Conv2d(
                resampled_channels,
                out_channels,
                kernel_size=refine_kernel_size,
                stride=1,
                padding=refine_kernel_size // 2,
                padding_mode=refine_padding_mode,
            )
            if refine_padding_mode == "replicate":
                # otherwise cuda conv2d with replicate returns contiguous NCHW, not channels last
                self.refine = self.refine.to(memory_format=torch.channels_last)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 3, 1, 2)
        x = self.refine(self.resample(x))
        # assert x.is_contiguous(memory_format=torch.channels_last)  # uncomment for perf debugging
        return x.contiguous(memory_format=torch.channels_last).permute(0, 2, 3, 1)


class PixelShuffle2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, scale_factor: int):
        super().__init__()
        self.scale_factor = scale_factor
        self.proj = nn.Linear(in_channels, out_channels * scale_factor**2)
        with torch.no_grad():
            self.proj.weight.view(
                scale_factor, scale_factor, out_channels, in_channels
            )[:] = self.proj.weight.view(
                scale_factor, scale_factor, out_channels, in_channels
            )[:1, :1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)
        b, h, w, _ = x.shape
        s = self.scale_factor
        x = self.proj(x).reshape(b, h, w, s, s, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).reshape(b, h * s, w * s, -1)
        return x.permute(0, 3, 1, 2)


def _make_norm_layer(
    norm_layer: str | None,
    channels: int,
) -> nn.Module:
    if norm_layer is None:
        return nn.Identity()
    if norm_layer == "layer_norm":
        return LayerNorm(channels)
    if norm_layer == "rms_norm":
        return RmsNorm(channels)
    raise ValueError(f"Unsupported norm layer: {norm_layer}")


def _make_activation_layer(
    act_layer: str | Callable,
) -> nn.Module:
    if callable(act_layer):
        return act_layer()
    if act_layer == "relu":
        return nn.ReLU()
    if act_layer == "elu":
        return nn.ELU()
    if act_layer == "gelu":
        return nn.GELU()
    raise ValueError(f"Unsupported activation function: {act_layer}")
