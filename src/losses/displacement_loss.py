"""Loss for displacement-based reconstruction networks (M4/M5/label-guided).

The loss compares predicted corrected points against ground-truth points using
**per-point squared L2** distances, focused on the damaged region. An optional
intact-region term (also squared L2) penalizes moving intact points and is used
for the blind displacement baseline.

Compared quantities:
- ``damaged_loss``: mean squared L2 between output and ground truth over points
  labelled 1 (reconstruction targets).
- ``intact_loss``: mean squared L2 over points labelled 0, weighted by
  ``intact_weight`` (0 disables the term).
"""

from typing import Dict, Tuple

import torch
import torch.nn as nn


class DisplacementLoss(nn.Module):
    """Per-point squared-L2 loss for displacement-based reconstruction.

    Computes an L2 loss between predicted and ground-truth positions, focused on
    damaged regions, with an optional weighted intact-region penalty.
    """

    def __init__(self, intact_weight: float = 0.0) -> None:
        """Initialize the loss.

        Args:
            intact_weight: weight for the intact-point loss (0.0 = ignore intact
                points). Set > 0 for BlindDisplacementNet to penalize moving
                intact points.
        """
        super().__init__()
        self.intact_weight = intact_weight

    def forward(
        self,
        output_points: torch.Tensor,
        displacements: torch.Tensor,
        gt_points: torch.Tensor,
        gt_labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute the displacement loss.

        Args:
            output_points: ``(B, N, 3)`` predicted corrected points.
            displacements: ``(B, N, 3)`` predicted displacements.
            gt_points: ``(B, N, 3)`` ground-truth points.
            gt_labels: ``(B, N)`` labels (0=intact, 1=reconstruction target).

        Returns:
            A tuple ``(loss, metrics)`` where ``loss`` is a scalar tensor and
            ``metrics`` reports the loss components and displacement statistics.
        """
        B, N, _ = output_points.shape

        # Per-point L2 distance
        point_dist = torch.sum((output_points - gt_points) ** 2, dim=-1)  # (B, N)

        # Damaged region loss
        damaged_mask = (gt_labels == 1).float()
        n_damaged = damaged_mask.sum()

        if n_damaged > 0:
            damaged_loss = (point_dist * damaged_mask).sum() / n_damaged
        else:
            damaged_loss = torch.tensor(0.0, device=output_points.device)

        # Intact region loss (penalize moving intact points)
        intact_mask = (gt_labels == 0).float()
        n_intact = intact_mask.sum()

        if n_intact > 0 and self.intact_weight > 0:
            intact_loss = (point_dist * intact_mask).sum() / n_intact
        else:
            intact_loss = torch.tensor(0.0, device=output_points.device)

        # Total loss
        loss = damaged_loss + self.intact_weight * intact_loss

        # Displacement magnitude stats
        disp_magnitude = torch.norm(displacements, dim=-1)  # (B, N)

        metrics = {
            'loss': loss.item(),
            'damaged_loss': damaged_loss.item(),
            'intact_loss': intact_loss.item(),
            'mean_displacement': disp_magnitude.mean().item(),
            'max_displacement': disp_magnitude.max().item(),
        }

        return loss, metrics
