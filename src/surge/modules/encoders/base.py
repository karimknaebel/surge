import itertools
from typing import Literal, Sequence

import torch
import torch.amp
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.utils.checkpoint
import torch.version


class BaseEncoder(nn.Module):
    image_mean: torch.Tensor
    image_std: torch.Tensor

    def __init__(
        self,
        image_mean: list[float] | torch.Tensor,
        image_std: list[float] | torch.Tensor,
        num_features: int,
        intermediate_layers: int | list[int],
        dim_out: int | list[int | None] | None,
        reduction: Literal["mean", "sum"] | None = "sum",
    ):
        super().__init__()

        self.num_features = num_features
        self.intermediate_layers = intermediate_layers
        self.output_projections = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=num_features,
                    out_channels=d_out,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
                if d_out is not None
                else nn.Identity()
                for _, d_out in zip(
                    range(
                        intermediate_layers
                        if isinstance(intermediate_layers, int)
                        else len(intermediate_layers)
                    ),
                    dim_out
                    if isinstance(dim_out, Sequence)
                    else itertools.repeat(dim_out),
                )
            ]
        )
        self.reduction = reduction

        self.register_buffer("image_mean", torch.tensor(image_mean).view(1, 3, 1, 1))
        self.register_buffer("image_std", torch.tensor(image_std).view(1, 3, 1, 1))

    @property
    def patch_size(self) -> tuple[int, int]:
        raise NotImplementedError

    def enable_gradient_checkpointing(self):
        pass

    def enable_compile(self):
        pass

    def forward_backbone(
        self, image: torch.Tensor, intermediate_layers: int | list[int]
    ):
        raise NotImplementedError

    def forward(
        self,
        image: torch.Tensor,
        token_rows: int,
        token_cols: int,
        return_class_token: bool = False,
    ) -> (
        tuple[torch.Tensor | list[torch.Tensor], torch.Tensor]
        | torch.Tensor
        | list[torch.Tensor]
    ):
        image = F.interpolate(
            image,
            (token_rows * self.patch_size[0], token_cols * self.patch_size[1]),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        image = (image - self.image_mean) / self.image_std

        x, cls_token = self.forward_backbone(image, self.intermediate_layers)

        # Project features to the desired dimensionality
        x = [proj(feat) for proj, feat in zip(self.output_projections, x, strict=True)]
        match self.reduction:
            case "mean":
                x = torch.stack(x).mean(dim=0)
            case "sum":
                x = torch.stack(x).sum(dim=0)
            case None:
                pass
            case _:
                raise ValueError(f"Unknown reduction: {self.reduction}")

        if return_class_token:
            return x, cls_token
        else:
            return x
