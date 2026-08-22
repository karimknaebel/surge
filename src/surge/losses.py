import torch


def _point_gradient_axis_matching(
    pred_0: torch.Tensor,
    pred_1: torch.Tensor,
    gt_0: torch.Tensor,
    gt_1: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    pred_scale = torch.minimum(pred_0[..., 2], pred_1[..., 2]).clamp_min(1e-6).reciprocal()
    gt_scale = torch.minimum(gt_0[..., 2], gt_1[..., 2]).clamp_min(1e-6).reciprocal()
    loss = (
        (pred_0 - pred_1) * pred_scale[..., None]
        - (gt_0 - gt_1) * gt_scale[..., None]
    ).norm(dim=-1)
    return (loss * mask).mean(dim=(-2, -1))


def point_gradient_matching(
    pred_points: torch.Tensor,
    gt_points: torch.Tensor,
    mask: torch.Tensor,
    edge_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    r"""Compute point gradient matching loss :math:`\mathcal{L}_{\mathrm{pgm}}`.

    ``pred_points``: ``(..., H, W, 3)``
    ``gt_points``: ``(..., H, W, 3)``
    ``mask``: ``(..., H, W)``
    ``edge_mask``: ``(..., H, W)``, occlusion boundaries excluded from the loss
    Returns: ``(...)``
    """
    gt_points = torch.where(mask[..., None], gt_points, 1)

    if edge_mask is not None:
        mask = mask & ~edge_mask

    loss_y = _point_gradient_axis_matching(
        pred_points[..., :-1, :, :],
        pred_points[..., 1:, :, :],
        gt_points[..., :-1, :, :],
        gt_points[..., 1:, :, :],
        mask[..., :-1, :] & mask[..., 1:, :],
    )
    loss_x = _point_gradient_axis_matching(
        pred_points[..., :, :-1, :],
        pred_points[..., :, 1:, :],
        gt_points[..., :, :-1, :],
        gt_points[..., :, 1:, :],
        mask[..., :, :-1] & mask[..., :, 1:],
    )
    return (loss_x + loss_y) / 2
