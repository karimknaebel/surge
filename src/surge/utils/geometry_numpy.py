# adjusted from https://github.com/microsoft/MoGe/blob/07444410f1e33f402353b99d6ccd26bd31e469e8/moge/utils/geometry_numpy.py

from functools import partial

import cv2
import numpy as np
import utils3d


def weighted_mean_numpy(
    x: np.ndarray,
    w: np.ndarray = None,
    axis: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
    eps: float = 1e-7,
) -> np.ndarray:
    if w is None:
        return np.mean(x, axis=axis)
    else:
        w = w.astype(x.dtype)
        return (x * w).mean(axis=axis) / np.clip(w.mean(axis=axis), eps, None)


def solve_optimal_focal_shift(uv: np.ndarray, xyz: np.ndarray):
    "Solve `min |focal * xy / (z + shift) - uv|` with respect to shift and focal"
    from scipy.optimize import least_squares

    uv, xy, z = uv.reshape(-1, 2), xyz[..., :2].reshape(-1, 2), xyz[..., 2].reshape(-1)

    def fn(uv: np.ndarray, xy: np.ndarray, z: np.ndarray, shift: np.ndarray):
        xy_proj = xy / (z + shift)[:, None]
        f = (xy_proj * uv).sum() / np.square(xy_proj).sum()
        err = (f * xy_proj - uv).ravel()
        return err

    solution = least_squares(partial(fn, uv, xy, z), x0=0, ftol=1e-3, method="lm")
    optim_shift = solution["x"].squeeze().astype(np.float32)

    xy_proj = xy / (z + optim_shift)[:, None]
    optim_focal = (xy_proj * uv).sum() / np.square(xy_proj).sum()

    return optim_shift, optim_focal


def solve_optimal_shift(uv: np.ndarray, xyz: np.ndarray, focal: float):
    "Solve `min |focal * xy / (z + shift) - uv|` with respect to shift"
    from scipy.optimize import least_squares

    uv, xy, z = uv.reshape(-1, 2), xyz[..., :2].reshape(-1, 2), xyz[..., 2].reshape(-1)

    def fn(uv: np.ndarray, xy: np.ndarray, z: np.ndarray, shift: np.ndarray):
        xy_proj = xy / (z + shift)[:, None]
        err = (focal * xy_proj - uv).ravel()
        return err

    solution = least_squares(partial(fn, uv, xy, z), x0=0, ftol=1e-3, method="lm")
    optim_shift = solution["x"].squeeze().astype(np.float32)

    return optim_shift


def depth_occlusion_edge_numpy(
    depth: np.ndarray, mask: np.ndarray, thickness: int = 1, tol: float = 0.1
):
    disp = np.where(mask, 1 / depth, 0)
    disp_pad = np.pad(disp, (thickness, thickness), constant_values=0)
    mask_pad = np.pad(mask, (thickness, thickness), constant_values=False)
    kernel_size = 2 * thickness + 1
    disp_window = utils3d.numpy.sliding_window_2d(
        disp_pad, (kernel_size, kernel_size), 1, axis=(-2, -1)
    )  # [..., H, W, kernel_size ** 2]
    mask_window = utils3d.numpy.sliding_window_2d(
        mask_pad, (kernel_size, kernel_size), 1, axis=(-2, -1)
    )  # [..., H, W, kernel_size ** 2]

    disp_mean = weighted_mean_numpy(disp_window, mask_window, axis=(-2, -1))
    fg_edge_mask = mask & (disp > (1 + tol) * disp_mean)
    bg_edge_mask = mask & (disp_mean > (1 + tol) * disp)

    edge_mask = (
        cv2.dilate(
            fg_edge_mask.astype(np.uint8),
            np.ones((3, 3), dtype=np.uint8),
            iterations=thickness,
        )
        > 0
    ) & (
        cv2.dilate(
            bg_edge_mask.astype(np.uint8),
            np.ones((3, 3), dtype=np.uint8),
            iterations=thickness,
        )
        > 0
    )

    return edge_mask
