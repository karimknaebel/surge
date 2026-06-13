import itertools
from typing import Any, Literal, Sequence

import torch
import torch.amp
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.utils.checkpoint
import torch.version
import utils3d
from einops import rearrange
from huggingface_hub import PyTorchModelHubMixin
from specbuild import build

from surge.modules.encoders.base import BaseEncoder
from surge.modules.heads.nad import NAD
from surge.utils.geometry_torch import normalized_view_plane_uv, recover_focal_shift


class SurGe(nn.Module, PyTorchModelHubMixin):
    def __init__(
        self,
        encoder: dict[str, Any],
        points_head: dict[str, Any] | None = None,
        remap_output: Literal["linear", "exp"] = "exp",
        concat_uv: list[bool] | bool = True,
        num_tokens_range: tuple[int, int] = (1024, 2802),
    ):
        super().__init__()

        self.encoder: BaseEncoder = build(encoder)
        if points_head is not None:
            self.points_head: NAD = build(points_head)
        self.remap_output = remap_output
        self.concat_uv = concat_uv
        self.num_tokens_range = num_tokens_range

    def enable_gradient_checkpointing(self):
        self.encoder.enable_gradient_checkpointing()
        self.points_head.enable_gradient_checkpointing()

    def _remap_points(self, points: torch.Tensor) -> torch.Tensor:
        if self.remap_output == "linear":
            pass
        elif self.remap_output == "exp":
            xy, z = points.split([2, 1], dim=-1)
            z = torch.exp(z)
            points = torch.cat([xy * z, z], dim=-1)
        else:
            raise ValueError(f"Invalid remap output type: {self.remap_output}")
        return points

    def forward(
        self,
        image: torch.Tensor,
        num_tokens: int,
        resize_output: bool = True,
    ) -> dict[str, torch.Tensor]:
        batch_size, _, img_h, img_w = image.shape
        aspect_ratio = img_w / img_h
        base_h, base_w = (
            int((num_tokens / aspect_ratio) ** 0.5),
            int((num_tokens * aspect_ratio) ** 0.5),
        )

        # Backbone
        features, _ = self.encoder(image, base_h, base_w, return_class_token=True)
        if not isinstance(features, list):
            features = [features, None, None, None, None]

        # Concat UV
        for level, concat_uv in zip(
            range(len(features)),
            self.concat_uv
            if isinstance(self.concat_uv, Sequence)
            else itertools.repeat(self.concat_uv),
        ):
            if not concat_uv:
                continue
            uv = normalized_view_plane_uv(
                width=base_w * 2**level,
                height=base_h * 2**level,
                aspect_ratio=aspect_ratio,
                dtype=image.dtype,
                device=image.device,
            )
            uv = uv.permute(2, 0, 1).unsqueeze(0).expand(batch_size, -1, -1, -1)
            if features[level] is None:
                features[level] = uv
            else:
                features[level] = torch.concat([features[level], uv], dim=1)

        with torch.autocast(device_type=image.device.type, dtype=torch.float32):
            # Head
            points = self.points_head(features)[-1]

            # Resize
            if resize_output:
                points = F.interpolate(
                    points,
                    (img_h, img_w),
                    mode="bilinear",
                    align_corners=False,
                    antialias=False,
                )

            # Remap output
            points = rearrange(points, "b c h w -> b h w c")
            points = self._remap_points(points)

            return {"points": points}

    # adjusted from https://github.com/microsoft/MoGe/blob/07444410f1e33f402353b99d6ccd26bd31e469e8/moge/model/v2.py#L194-L303
    @torch.inference_mode()
    def infer(
        self,
        image: torch.Tensor,
        num_tokens: int | Literal["min", "max"] = "max",
        resize_output: bool = True,
        force_projection: bool = True,
        fov_x: float | torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        User-friendly inference function

        ### Parameters
        - `image`: input image tensor of shape (B, 3, H, W)
        - `num_tokens`: base ViT token budget for inference. Suggested range: 1024--2802.
            More tokens will result in significantly higher accuracy and finer details, but slower inference time. Default: 2802.
        - `resize_output`: if True, resize the output point map to the input image size. Default: True
        - `force_projection`: if True, the output point map will be computed using the actual depth map. Default: True
        - `fov_x`: the horizontal camera FoV in degrees. If None, it will be inferred from the predicted point map. Default: None

        ### Returns

        A dictionary containing the following keys:
        - `points`: output tensor of shape (B, H, W, 3).
        - `depth`: tensor of shape (B, H, W) containing the depth map.
        - `intrinsics`: tensor of shape (B, 3, 3) containing the camera intrinsics.
        """
        device = next(self.parameters()).device

        image = image.to(device)
        H, W = image.shape[-2:]
        aspect_ratio = W / H

        if num_tokens in {"min", "max"}:
            num_tokens = self.num_tokens_range[0 if num_tokens == "min" else 1]

        # Forward pass
        output = self(image, num_tokens=num_tokens, resize_output=resize_output)
        points = output["points"]

        with torch.autocast(device_type=device.type, dtype=torch.float32):
            # Convert affine point map to camera-space. Recover depth and intrinsics from point map.
            # NOTE: Focal here is the focal length relative to half the image diagonal
            if fov_x is None:
                # Recover focal and shift from predicted point map
                focal, shift = recover_focal_shift(points)
            else:
                # Focal is known, recover shift only
                focal = aspect_ratio / (1 + aspect_ratio**2) ** 0.5 / torch.tan(torch.deg2rad(torch.as_tensor(fov_x, device=points.device, dtype=points.dtype) / 2))  # fmt: off
                if focal.ndim == 0:
                    focal = focal[None].expand(points.shape[0])
                _, shift = recover_focal_shift(points, focal=focal)
            fx, fy = (
                focal / 2 * (1 + aspect_ratio**2) ** 0.5 / aspect_ratio,
                focal / 2 * (1 + aspect_ratio**2) ** 0.5,
            )
            intrinsics = utils3d.torch.intrinsics_from_focal_center(fx, fy, 0.5, 0.5)
            points[..., 2] += shift[..., None, None]
            depth = points[..., 2].clone()

            # If projection constraint is forced, recompute the point map using the actual depth map & intrinsics
            if force_projection:
                points = utils3d.torch.depth_to_points(
                    depth.double(),
                    intrinsics=intrinsics.double(),
                ).to(points.dtype)

        return {"points": points, "depth": depth, "intrinsics": intrinsics}
