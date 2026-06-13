from contextlib import nullcontext

import torch
import torch.nn as nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
)
from specbuild import REGISTRY

from surge.modules.encoders.base import BaseEncoder


@REGISTRY.register()
class DINOv2Encoder(BaseEncoder):
    backbone: nn.Module
    image_mean: torch.Tensor
    image_std: torch.Tensor

    def __init__(
        self,
        backbone: str,
        pretrained: bool = True,
        frozen: bool = False,
        **kwargs,
    ):
        _backbone = torch.hub.load(
            "karimknaebel/dinov2-core:v0.1.2",
            backbone,
            verbose=False,
            trust_repo=True,
            pretrained=pretrained,
        )
        super().__init__(
            image_mean=[0.485, 0.456, 0.406],
            image_std=[0.229, 0.224, 0.225],
            num_features=_backbone.blocks[0].attn.qkv.in_features,
            **kwargs,
        )
        self.backbone = _backbone

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
            self.backbone.blocks[i] = checkpoint_wrapper(self.backbone.blocks[i])

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
