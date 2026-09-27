"""Shared Chamfer-distance implementations.

Two bidirectional Chamfer-distance variants are provided, matching the exact
formulas used across the Source_Repository:

- :func:`chamfer_distance_l1` uses **unsquared** (L1 / Euclidean) nearest-neighbor
  distances via ``torch.cdist``. This is the default variant used by
  :class:`~src.losses.seed_displacement_loss.SeedDisplacementLoss`.
- :func:`chamfer_distance_squared` uses **squared** Euclidean nearest-neighbor
  distances. This is the variant used by the completion losses (PCN, label-aware)
  and the squared-CD ablation of the seed-displacement loss.

Each function compares two point sets and returns a scalar equal to the mean
nearest-neighbor distance from ``pred`` to ``gt`` plus the mean nearest-neighbor
distance from ``gt`` to ``pred``.
"""

import torch


def chamfer_distance_l1(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Bidirectional Chamfer distance using unsquared (L1/Euclidean) distances.

    Args:
        pred: ``(B, N, 3)`` predicted point set.
        gt: ``(B, M, 3)`` ground-truth point set.

    Returns:
        Scalar tensor: mean min-distance ``pred->gt`` plus mean min-distance
        ``gt->pred``, using unsquared Euclidean distances.
    """
    dist = torch.cdist(pred, gt)  # (B, N, M) - L2 unsquared
    d1 = dist.min(dim=2)[0].mean()
    d2 = dist.min(dim=1)[0].mean()
    return d1 + d2


def chamfer_distance_squared(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Bidirectional Chamfer distance using squared Euclidean distances.

    Args:
        pred: ``(B, N, 3)`` predicted point set.
        gt: ``(B, M, 3)`` ground-truth point set.

    Returns:
        Scalar tensor: mean min squared-distance ``pred->gt`` plus mean min
        squared-distance ``gt->pred``.
    """
    pred_exp = pred.unsqueeze(2)  # (B, N, 1, 3)
    gt_exp = gt.unsqueeze(1)      # (B, 1, M, 3)
    dist_sq = torch.sum((pred_exp - gt_exp) ** 2, dim=-1)  # (B, N, M)
    d1 = dist_sq.min(dim=2)[0].mean()
    d2 = dist_sq.min(dim=1)[0].mean()
    return d1 + d2
