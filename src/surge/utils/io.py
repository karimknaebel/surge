# adjusted from https://github.com/microsoft/MoGe/blob/main/moge/utils/io.py

import os

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
from pathlib import Path
from typing import IO

import cv2
import numpy as np
from PIL import Image


def save_glb(
    save_path: str | os.PathLike,
    vertices: np.ndarray,
    faces: np.ndarray,
    vertex_uvs: np.ndarray,
    texture: np.ndarray,
    vertex_normals: np.ndarray | None = None,
):
    import trimesh
    import trimesh.visual

    trimesh.Trimesh(
        vertices=vertices,
        vertex_normals=vertex_normals,
        faces=faces,
        visual=trimesh.visual.texture.TextureVisuals(
            uv=vertex_uvs,
            material=trimesh.visual.material.PBRMaterial(
                baseColorTexture=Image.fromarray(texture),
                metallicFactor=0.5,
                roughnessFactor=1.0,
            ),
        ),
        process=False,
    ).export(save_path)


def save_ply(
    save_path: str | os.PathLike,
    vertices: np.ndarray,
    faces: np.ndarray,
    vertex_colors: np.ndarray,
    vertex_normals: np.ndarray | None = None,
):
    import trimesh
    import trimesh.visual

    trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        vertex_colors=vertex_colors,
        vertex_normals=vertex_normals,
        process=False,
    ).export(save_path)


def read_image(path: str | os.PathLike | IO) -> np.ndarray:
    """
    Read a image, return uint8 RGB array of shape (H, W, 3).
    """
    if isinstance(path, (str, os.PathLike)):
        data = Path(path).read_bytes()
    else:
        data = path.read()
    image = cv2.cvtColor(
        cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB
    )
    return image
