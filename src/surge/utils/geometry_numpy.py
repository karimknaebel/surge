# adjusted from https://github.com/microsoft/MoGe/blob/07444410f1e33f402353b99d6ccd26bd31e469e8/moge/utils/geometry_numpy.py

from functools import partial

import cv2
import numpy as np


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
    # MoGe computes the masked mean inverse depth with sliding windows:
    # https://github.com/microsoft/MoGe/blob/07444410f1e33f402353b99d6ccd26bd31e469e8/moge/utils/geometry_numpy.py#L146-L161
    # Box filters compute the same mean. The full function was ~33x faster on a
    # 640x480 float32 depth map with thickness=2 and tol=0.01.
    disp = np.zeros_like(depth, dtype=np.result_type(depth.dtype, np.float32))
    np.divide(1, depth, out=disp, where=mask)
    kernel_size = 2 * thickness + 1
    disp_mean = cv2.boxFilter(
        disp, cv2.CV_64F, (kernel_size, kernel_size), normalize=False, borderType=cv2.BORDER_CONSTANT
    )
    valid_count = cv2.boxFilter(
        mask.astype(np.uint8), cv2.CV_32F, (kernel_size, kernel_size), normalize=False, borderType=cv2.BORDER_CONSTANT
    )
    np.divide(disp_mean, valid_count, out=disp_mean, where=mask)
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
