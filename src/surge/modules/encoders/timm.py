import timm
import torch
from specbuild import REGISTRY
from timm.models.vision_transformer import VisionTransformer

from surge.modules.encoders.base import BaseEncoder


@REGISTRY.register()
class TimmEncoder(BaseEncoder):
    image_mean: torch.Tensor
    image_std: torch.Tensor

    def __init__(
        self,
        backbone: str,  # e.g. "vit_large_patch14_dinov2",
        pretrained: bool = True,
        **kwargs,
    ):
        _backbone: VisionTransformer = timm.create_model(
            backbone,
            pretrained=pretrained,
            dynamic_img_size=True,
            num_classes=0,
            global_pool="",
        )
        super().__init__(
            image_mean=_backbone.pretrained_cfg["mean"],
            image_std=_backbone.pretrained_cfg["std"],
            num_features=_backbone.num_features,
            **kwargs,
        )
        self.backbone = _backbone

    @property
    def patch_size(self) -> tuple[int, int]:
        return self.backbone.patch_embed.patch_size

    def enable_gradient_checkpointing(self):
        self.backbone.set_grad_checkpointing(True)

    def enable_compile(self):
        pass
        # NOTE: at least with the current dynamic res training, compile seems to *slow down* this module
        # for block in self.backbone.blocks:
        #     block.compile(dynamic=True)

    def forward_backbone(
        self, image: torch.Tensor, intermediate_layers: int | list[int]
    ):
        features = self.backbone.forward_intermediates(
            image,
            indices=intermediate_layers,
            return_prefix_tokens=True,
            norm=True,
            intermediates_only=True,
        )
        return [feat for (feat, cls_token) in features], features[-1][1]
