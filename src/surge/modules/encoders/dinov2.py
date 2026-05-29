import importlib
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
from specbuild import REGISTRY

from surge._vendor.dinov2.models.vision_transformer import DinoVisionTransformer
from surge.modules.encoders.base import BaseEncoder


@REGISTRY.register()
class DINOv2Encoder(BaseEncoder):
    backbone: DinoVisionTransformer
    image_mean: torch.Tensor
    image_std: torch.Tensor

    def __init__(
        self,
        backbone: str,
        pretrained: bool = True,
        frozen: bool = False,
        **kwargs,
    ):
        hub_loader = getattr(
            importlib.import_module("surge._vendor.dinov2.hub.backbones"),
            backbone,
        )
        _backbone = hub_loader(pretrained=pretrained)
        super().__init__(
            image_mean=[0.485, 0.456, 0.406],
            image_std=[0.229, 0.224, 0.225],
            num_features=_backbone.blocks[0].attn.qkv.in_features,
            **kwargs,
        )
        self.backbone = _backbone
        self._enable_pytorch_native_sdpa()

        self.frozen = frozen
        if self.frozen:
            self.backbone.requires_grad_(False)
            self.backbone.eval()

    def train(self, mode: bool = True):
        out = super().train(mode)
        if self.frozen:
            self.backbone.eval()
        return out

    @property
    def patch_size(self) -> tuple[int, int]:
        return self.backbone.patch_embed.patch_size

    def enable_gradient_checkpointing(self):
        for i in range(len(self.backbone.blocks)):
            # NOTE: torch.compile doesn't work with this
            wrap_module_with_gradient_checkpointing(self.backbone.blocks[i])

    def _enable_pytorch_native_sdpa(self):
        for i in range(len(self.backbone.blocks)):
            # NOTE: torch.compile (probably) doesn't work with this
            wrap_dinov2_attention_with_sdpa(self.backbone.blocks[i].attn)

    def enable_compile(self):
        pass

    def forward_backbone(
        self, image: torch.Tensor, intermediate_layers: int | list[int]
    ):
        with torch.no_grad() if self.frozen else nullcontext():
            features = self.backbone.get_intermediate_layers(
                image,
                n=intermediate_layers,
                reshape=True,
                return_class_token=True,
            )
        return [feat for (feat, cls_token) in features], features[-1][1]


# adjusted from https://github.com/microsoft/MoGe/blob/07444410f1e33f402353b99d6ccd26bd31e469e8/moge/model/utils.py#L7-L15
def wrap_module_with_gradient_checkpointing(module: nn.Module):
    from torch.utils.checkpoint import checkpoint

    class _CheckpointingWrapper(module.__class__):
        _restore_cls = module.__class__

        def forward(self, *args, **kwargs):
            return checkpoint(super().forward, *args, use_reentrant=False, **kwargs)

    module.__class__ = _CheckpointingWrapper
    return module


# adjusted from https://github.com/microsoft/MoGe/blob/07444410f1e33f402353b99d6ccd26bd31e469e8/moge/model/utils.py#L22-L38
def wrap_dinov2_attention_with_sdpa(module: nn.Module):
    assert torch.__version__ >= "2.0", "SDPA requires PyTorch 2.0 or later"

    class _AttentionWrapper(module.__class__):
        def forward(self, x: torch.Tensor, attn_bias=None) -> torch.Tensor:
            B, N, C = x.shape
            qkv = (
                self.qkv(x)
                .reshape(B, N, 3, self.num_heads, C // self.num_heads)
                .permute(2, 0, 3, 1, 4)
            )  # (3, B, H, N, C // H)

            q, k, v = torch.unbind(qkv, 0)  # (B, H, N, C // H)

            x = F.scaled_dot_product_attention(q, k, v, attn_bias)
            x = x.permute(0, 2, 1, 3).reshape(B, N, C)

            x = self.proj(x)
            x = self.proj_drop(x)
            return x

    module.__class__ = _AttentionWrapper
    return module
