"""Label-aware loss for local reconstruction (LR-PCN, M3).

The loss is computed **only** on reconstruction regions (ground-truth points
labelled 1), never on intact points. It combines:

1. A **squared Chamfer distance** between predicted reconstruction points and
   the ground-truth reconstruction targets.
2. A **coverage** term (mean nearest predicted-point distance for each target),
   using unsquared Euclidean distance via ``torch.cdist``, ensuring every target
   is covered by at least one predicted point.
3. Optional **imbalance weighting** that scales the loss inversely to the damaged
   ratio, so partitions with few damaged points receive stronger gradients.
"""

from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from .chamfer import chamfer_distance_squared


class LabelAwareLoss(nn.Module):
    """Label-aware loss for local reconstruction.

    Only computes loss on reconstruction regions, not intact points.

    Args:
        coverage_weight: weight applied to the coverage term.
        imbalance_aware: when ``True``, scale per-sample loss inversely to the
            damaged ratio (clamped to ``[1, 5]``).
    """

    def __init__(self, coverage_weight: float = 0.5, imbalance_aware: bool = True) -> None:
        super().__init__()
        self.coverage_weight = coverage_weight
        self.imbalance_aware = imbalance_aware

    def forward(
        self,
        intact_points_list: List[torch.Tensor],
        reconstructed_points: torch.Tensor,
        gt_points: torch.Tensor,
        gt_labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute the label-aware reconstruction loss.

        Args:
            intact_points_list: list of ``(K_b, 3)`` intact points per batch.
            reconstructed_points: ``(B, M, 3)`` predicted reconstruction points.
            gt_points: ``(B, N, 3)`` ground-truth points.
            gt_labels: ``(B, N)`` GT labels (0=intact, 1=reconstruction target).

        Returns:
            A tuple ``(loss, metrics)`` where ``loss`` is a scalar tensor and
            ``metrics`` reports the Chamfer, coverage, and total loss components.
        """
        batch_size = reconstructed_points.shape[0]
        device = reconstructed_points.device

        # Extract GT reconstruction targets (label=1)
        cd_reconstruction = 0
        coverage_loss = 0
        num_valid_samples = 0

        for b in range(batch_size):
            # Get reconstruction targets from GT
            recon_mask = gt_labels[b] == 1

            if recon_mask.sum() == 0:
                # No reconstruction targets in this sample, skip
                continue

            gt_recon_targets = gt_points[b][recon_mask].unsqueeze(0)  # (1, K, 3)
            pred_recon = reconstructed_points[b:b + 1]  # (1, M, 3)

            # Imbalance weight: boost loss for partitions with few damaged points
            if self.imbalance_aware:
                n_total = gt_labels[b].shape[0]
                n_damaged = recon_mask.sum().float()
                # Weight inversely proportional to damaged ratio, clamped
                weight = torch.clamp(n_total / (2.0 * n_damaged), min=1.0, max=5.0)
            else:
                weight = 1.0

            # Chamfer Distance between predicted reconstruction and GT targets
            cd = chamfer_distance_squared(pred_recon, gt_recon_targets)
            cd_reconstruction += cd * weight

            # Coverage: Ensure all GT targets are close to at least one predicted point
            # This prevents the network from ignoring parts of the damage
            dist_matrix = torch.cdist(gt_recon_targets, pred_recon)  # (1, K, M)
            min_dist_to_pred = torch.min(dist_matrix, dim=2)[0]  # (1, K)
            coverage = torch.mean(min_dist_to_pred)
            coverage_loss += coverage * weight

            num_valid_samples += 1

        if num_valid_samples == 0:
            # No reconstruction targets in batch
            return torch.tensor(0.0, device=device, requires_grad=True), {
                'cd_reconstruction': 0.0,
                'coverage': 0.0,
                'loss': 0.0
            }

        # Average over valid samples
        cd_reconstruction = cd_reconstruction / num_valid_samples
        coverage_loss = coverage_loss / num_valid_samples

        # Total loss
        loss = cd_reconstruction + self.coverage_weight * coverage_loss

        metrics = {
            'cd_reconstruction': cd_reconstruction.item(),
            'coverage': coverage_loss.item(),
            'loss': loss.item()
        }

        return loss, metrics
