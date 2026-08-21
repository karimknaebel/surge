import torch
import utils3d
from moge.utils.alignment import (
    align_depth_affine,
    align_points_scale_xyz_shift,
)
from moge.utils.geometry_torch import (
    angle_diff_vec3,
    intrinsics_to_fov,
    mask_aware_nearest_resize,
)
from moge.utils.tools import key_average

DELTA_THRESHOLDS = (0.01, 0.05, 0.25)
MIN_SEGMENT_PIXELS = 10


def rel_depth(pred: torch.Tensor, gt: torch.Tensor, eps: float = 1e-6) -> float:
    return (torch.abs(pred - gt) / (gt + eps)).mean().item()


def delta_depth(pred: torch.Tensor, gt: torch.Tensor, threshold: float) -> float:
    error = torch.abs(pred - gt)
    scale = torch.minimum(torch.abs(pred), torch.abs(gt))
    return (error < threshold * scale).float().mean().item()


def depth_metrics_dict(pred: torch.Tensor, gt: torch.Tensor) -> dict[str, float]:
    return {
        "rel": rel_depth(pred, gt),
        **{
            f"delta_{threshold:.2f}": delta_depth(pred, gt, threshold)
            for threshold in DELTA_THRESHOLDS
        },
    }


def rel_point(pred: torch.Tensor, gt: torch.Tensor, eps: float = 1e-6) -> float:
    return (
        (torch.norm(pred - gt, dim=-1) / (torch.norm(gt, dim=-1) + eps)).mean().item()
    )


def delta_point(pred: torch.Tensor, gt: torch.Tensor, threshold: float) -> float:
    dist_pred = torch.norm(pred, dim=-1)
    dist_gt = torch.norm(gt, dim=-1)
    dist_err = torch.norm(pred - gt, dim=-1)
    return (
        (dist_err < threshold * torch.minimum(dist_gt, dist_pred)).float().mean().item()
    )


def point_metrics_dict(pred: torch.Tensor, gt: torch.Tensor) -> dict[str, float]:
    return {
        "rel": rel_point(pred, gt),
        **{
            f"delta_{threshold:.2f}": delta_point(pred, gt, threshold)
            for threshold in DELTA_THRESHOLDS
        },
    }


def rel_point_local(
    pred: torch.Tensor, gt: torch.Tensor, diameter: torch.Tensor
) -> float:
    return (torch.norm(pred - gt, dim=-1) / diameter).mean().item()


def delta_point_local(
    pred: torch.Tensor,
    gt: torch.Tensor,
    diameter: torch.Tensor,
    threshold: float,
) -> float:
    return (torch.norm(pred - gt, dim=-1) < threshold * diameter).float().mean().item()


def boundary_f1(
    pred: torch.Tensor,
    gt: torch.Tensor,
    mask: torch.Tensor,
    radius: int = 1,
) -> float:
    neighbor_x, neighbor_y = torch.meshgrid(
        torch.linspace(-radius, radius, 2 * radius + 1, device=pred.device),
        torch.linspace(-radius, radius, 2 * radius + 1, device=pred.device),
        indexing="xy",
    )
    neighbor_mask = (neighbor_x**2 + neighbor_y**2) <= radius**2 + 1e-5

    pred_window = utils3d.pt.sliding_window(
        pred, window_size=2 * radius + 1, stride=1, dim=(-2, -1)
    )
    gt_window = utils3d.pt.sliding_window(
        gt, window_size=2 * radius + 1, stride=1, dim=(-2, -1)
    )
    mask_window = neighbor_mask & utils3d.pt.sliding_window(
        mask, window_size=2 * radius + 1, stride=1, dim=(-2, -1)
    )

    pred_rel = pred_window / pred[radius:-radius, radius:-radius, None, None]
    gt_rel = gt_window / gt[radius:-radius, radius:-radius, None, None]
    valid = mask[radius:-radius, radius:-radius, None, None] & mask_window

    thresholds = torch.linspace(0.05, 0.25, 10).tolist()
    scores = []
    for threshold in thresholds:
        pred_label = pred_rel > 1 + threshold
        gt_label = gt_rel > 1 + threshold
        true_positive = (pred_label & gt_label & valid).float().sum()
        precision = true_positive / (pred_label & valid).float().sum().clamp_min(1e-12)
        recall = true_positive / (gt_label & valid).float().sum().clamp_min(1e-12)
        scores.append(
            (2 * precision * recall / (precision + recall).clamp_min(1e-12)).item()
        )

    return sum(weight * score for weight, score in zip(thresholds, scores)) / sum(
        thresholds
    )


def normal_metrics_dict(
    pred: torch.Tensor,
    gt: torch.Tensor,
    mask: torch.Tensor,
) -> dict[str, float]:
    error = torch.rad2deg(angle_diff_vec3(pred[mask], gt[mask])).abs()
    return {
        "mean": error.mean().item(),
        "median": error.median().item(),
        "11.25deg": (error < 11.25).float().mean().item(),
    }


def point_normal_metrics_dict(
    pred: torch.Tensor,
    gt: torch.Tensor,
    mask: torch.Tensor,
) -> dict[str, float]:
    pred_normal, _ = utils3d.pt.point_map_to_normal_map(pred, mask)
    gt_normal, gt_normal_mask = utils3d.pt.point_map_to_normal_map(gt, mask)
    return normal_metrics_dict(pred_normal, gt_normal, gt_normal_mask)


def _compute_point_local_metrics(
    pred_points: torch.Tensor,
    gt_points: torch.Tensor,
    mask: torch.Tensor,
    lr_mask: torch.Tensor,
    lr_index: tuple[torch.Tensor, ...],
    segmentation_mask: torch.Tensor,
    segmentation_labels: dict[str, int],
) -> dict | None:
    segmentation_mask_lr = segmentation_mask[lr_index]
    per_segment_metrics = []
    for segment_id in segmentation_labels.values():
        valid_mask_lr = (segmentation_mask_lr == segment_id) & lr_mask
        if valid_mask_lr.sum().item() < MIN_SEGMENT_PIXELS:
            continue
        valid_mask = (segmentation_mask == segment_id) & mask
        if not valid_mask.any():
            continue

        pred_points_masked = pred_points[valid_mask]
        gt_points_masked = gt_points[valid_mask]
        pred_points_masked_lr = pred_points[lr_index][valid_mask_lr]
        gt_points_masked_lr = gt_points[lr_index][valid_mask_lr]
        diameter = (
            gt_points_masked.max(dim=0).values - gt_points_masked.min(dim=0).values
        ).max()
        scale, shift = align_points_scale_xyz_shift(
            pred_points_masked_lr,
            gt_points_masked_lr,
            1 / diameter.expand(gt_points_masked_lr.shape[0]),
        )
        pred_points_masked = pred_points_masked * scale + shift
        per_segment_metrics.append(
            {
                "rel": rel_point_local(pred_points_masked, gt_points_masked, diameter),
                **{
                    f"delta_{threshold:.2f}": delta_point_local(
                        pred_points_masked,
                        gt_points_masked,
                        diameter,
                        threshold,
                    )
                    for threshold in DELTA_THRESHOLDS
                },
            }
        )

    return key_average(per_segment_metrics) if per_segment_metrics else None


@torch.no_grad()
def compute_metrics(
    pred: dict[str, torch.Tensor],
    gt: dict,
) -> dict[str, dict]:
    mask = gt["depth_mask"]
    gt_depth = gt["depth"]
    gt_points = gt["points"]
    pred_depth = pred["depth_affine_invariant"]
    pred_points = pred["points_affine_invariant"]
    _, lr_mask, lr_index = mask_aware_nearest_resize(
        None, mask, (64, 64), return_index=True
    )

    gt_depth_lr = gt_depth[lr_index][lr_mask]
    depth_scale, depth_shift = align_depth_affine(
        pred_depth[lr_index][lr_mask], gt_depth_lr, 1 / gt_depth_lr
    )
    aligned_depth = pred_depth * depth_scale + depth_shift

    gt_points_lr = gt_points[lr_index][lr_mask]
    point_scale, point_shift = align_points_scale_xyz_shift(
        pred_points[lr_index][lr_mask],
        gt_points_lr,
        1 / gt_points_lr.norm(dim=-1),
    )
    aligned_points = pred_points * point_scale + point_shift

    point_metrics = {
        "affine_invariant": point_metrics_dict(aligned_points[mask], gt_points[mask])
    }
    if gt.get("segmentation_mask") is not None and gt.get("segmentation_labels"):
        local_metrics = _compute_point_local_metrics(
            pred_points,
            gt_points,
            mask,
            lr_mask,
            lr_index,
            gt["segmentation_mask"],
            gt["segmentation_labels"],
        )
        if local_metrics is not None:
            point_metrics["local"] = local_metrics
    if gt["has_clean_normal"]:
        point_metrics["normal"] = point_normal_metrics_dict(
            pred_points, gt_points, mask
        )

    depth_metrics = {
        "affine_invariant": depth_metrics_dict(aligned_depth[mask], gt_depth[mask])
    }
    if gt["has_sharp_boundary"]:
        depth_metrics["boundary_f1"] = boundary_f1(
            aligned_depth, gt_depth, mask, radius=1
        )

    pred_fov_x, _ = intrinsics_to_fov(pred["intrinsics"])
    gt_fov_x, _ = intrinsics_to_fov(gt["intrinsics"])
    fov_x_error = torch.rad2deg(pred_fov_x - gt_fov_x)
    focal_x_relative_error = (
        pred["intrinsics"][..., 0, 0] - gt["intrinsics"][..., 0, 0]
    ) / gt["intrinsics"][..., 0, 0]

    return {
        "point": point_metrics,
        "depth": depth_metrics,
        "intrinsics": {
            "fov_x": {
                "absolute_error": fov_x_error.abs().mean().item(),
                "signed_error": fov_x_error.item(),
            },
            "focal_x": {
                "absolute_relative_error": focal_x_relative_error.abs().mean().item(),
                "signed_relative_error": focal_x_relative_error.item(),
            },
        },
    }
