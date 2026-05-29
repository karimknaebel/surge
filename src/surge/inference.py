import os
from typing import Literal

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import cv2
import numpy as np
import torch
import utils3d

from surge import SurGe
from surge.utils.geometry_numpy import depth_occlusion_edge_numpy

DEFAULT_CHECKPOINT = "karimknaebel/surge-large"
DEFAULT_TOKEN_RANGE = (1024, 2802)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}

# Model/Rerun camera space is RDF; trimesh/glTF export space is RUB.
RDF_RUB_SCALE = np.array([1, -1, -1], dtype=np.float32)
TRIMESH_UV_SCALE = np.array([1, -1], dtype=np.float32)
TRIMESH_UV_OFFSET = np.array([0, 1], dtype=np.float32)


def resize_to_max_size(image: np.ndarray, max_size: int | None) -> np.ndarray:
    if max_size is None:
        return image

    height, width = image.shape[:2]
    scale = max_size / max(height, width)
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    return cv2.resize(
        image,
        (round(width * scale), round(height * scale)),
        interpolation=interpolation,
    )


def image_to_tensor(image: np.ndarray, device: torch.device) -> torch.Tensor:
    return (
        torch.from_numpy(image.astype(np.float32) / 255.0)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(device)
    )


def postprocess_output(
    image: np.ndarray,
    output: dict[str, torch.Tensor],
    unit_scale: bool = False,
) -> dict[str, np.ndarray]:
    points = output["points"][0].float().cpu().numpy()
    depth = output["depth"][0].float().cpu().numpy()

    if image.shape[:2] != points.shape[:2]:
        image = cv2.resize(
            image,
            (points.shape[1], points.shape[0]),
            interpolation=cv2.INTER_AREA,
        )

    mask = np.isfinite(depth) & (depth > 0) & np.isfinite(points).all(axis=-1)
    if unit_scale:
        scale = np.linalg.norm(points[mask], axis=-1).mean()
        points = points / scale
        depth = depth / scale

    normal, normal_mask = utils3d.numpy.points_to_normals(points, mask=mask)
    return {
        "image": image,
        "points": points,
        "depth": depth,
        "mask": mask,
        "normal": normal,
        "normal_mask": normal_mask,
        "intrinsics": output["intrinsics"][0].float().cpu().numpy(),
    }


def infer_image(
    model: SurGe,
    image: np.ndarray,
    device: torch.device,
    num_tokens: int | Literal["min", "max"],
    resize_output: bool = True,
    force_projection: bool = True,
    fov_x: float | None = None,
    unit_scale: bool = False,
) -> dict[str, np.ndarray]:
    return postprocess_output(
        image,
        model.infer(
            image_to_tensor(image, device),
            num_tokens=num_tokens,
            resize_output=resize_output,
            force_projection=force_projection,
            fov_x=fov_x,
        ),
        unit_scale=unit_scale,
    )


def build_mesh(
    result: dict[str, np.ndarray],
    edge_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    image = result["image"]
    height, width = image.shape[:2]
    mask = result["mask"] & result["normal_mask"]
    if not np.isinf(edge_threshold):
        mask = mask & ~depth_occlusion_edge_numpy(
            result["depth"],
            mask,
            tol=edge_threshold,
            thickness=2,
        )

    faces, vertices, vertex_colors, vertex_uvs, vertex_normals = (
        utils3d.numpy.image_mesh(
            result["points"],
            image.astype(np.float32) / 255.0,
            utils3d.numpy.image_uv(width=width, height=height),
            result["normal"],
            mask=mask,
            tri=True,
        )
    )

    vertices = vertices * RDF_RUB_SCALE
    vertex_uvs = vertex_uvs * TRIMESH_UV_SCALE + TRIMESH_UV_OFFSET
    vertex_normals = vertex_normals * RDF_RUB_SCALE
    return faces, vertices, vertex_colors, vertex_uvs, vertex_normals


def point_cloud_from_result(
    result: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mask = result["mask"] & result["normal_mask"]
    return (
        result["points"][mask] * RDF_RUB_SCALE,
        result["image"][mask],
        result["normal"][mask] * RDF_RUB_SCALE,
    )
