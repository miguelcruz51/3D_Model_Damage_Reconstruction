"""Loss for the seed-based displacement network (SeedDispNet, M7).

The loss compares the reconstructed damaged points against the ground-truth
reconstruction targets (points labelled 1 in the ground truth). Three branches
are supported, selected by two flags:

- ``use_cd=False`` (default): **per-point L2** regression between index-aligned
  reconstructed and ground-truth points (relies on the point-to-point
  correspondence). The compared quantity is the sum of squared coordinate
  differences per point, averaged over points.
- ``use_cd=True, use_squared_cd=False``: **L1 (unsquared) Chamfer distance**
  between the reconstructed set and the ground-truth target set.
- ``use_cd=True, use_squared_cd=True``: **squared Chamfer distance** ablation.

An intact-preservation error (squared L2 between the output and ground truth on
intact points) is reported as a diagnostic metric but is not part of the
optimized loss.
"""

from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from .chamfer import chamfer_distance_l1, chamfer_distance_squared


class SeedDisplacementLoss(nn.Module):
    """Loss for seed-based displacement reconstruction.

    Args:
        use_cd: if ``False`` (default), use per-point L2; if ``True`` use a
            Chamfer distance.
        use_squared_cd: when ``use_cd`` is ``True``, select the squared Chamfer
            distance (``True``) or the L1 Chamfer distance (``False``, default).
    """

    def __init__(self, use_cd: bool = False, use_squared_cd: bool = False) -> None:
        super().__init__()
        self.use_cd = use_cd
        self.use_squared_cd = use_squared_cd

    def forward(
        self,
        output_points: torch.Tensor,
        reconstructed_list: List[torch.Tensor],
        damaged_masks: List[torch.Tensor],
        gt_points: torch.Tensor,
        gt_labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute the seed-displacement loss.

        Args:
            output_points: ``(B, N, 3)`` full corrected cloud.
            reconstructed_list: list of ``(K_b, 3)`` reconstructed points.
            damaged_masks: list of boolean masks marking damaged points.
            gt_points: ``(B, N, 3)`` ground truth.
            gt_labels: ``(B, N)`` ground-truth labels (0=intact, 1=target).

        Returns:
            A tuple ``(loss, metrics)`` where ``loss`` is a scalar tensor and
            ``metrics`` is a dict with ``loss``, ``damaged_loss`` and
            ``intact_loss`` values.
        """
        B = gt_points.shape[0]
        device = gt_points.device

        total_loss = 0
        total_damaged_loss = 0
        total_intact_loss = 0
        n_valid = 0

        for b in range(B):
            recon = reconstructed_list[b]  # (K, 3)
            damaged_mask = damaged_masks[b]

            if len(recon) == 0:
                continue

            # GT targets for damaged region
            gt_targets = gt_points[b][gt_labels[b] == 1]  # (K, 3)

            if self.use_cd:
                # Chamfer Distance - L1 or squared depending on flag
                if self.use_squared_cd:
                    loss = chamfer_distance_squared(
                        recon.unsqueeze(0), gt_targets.unsqueeze(0)
                    )
                else:
                    loss = chamfer_distance_l1(
                        recon.unsqueeze(0), gt_targets.unsqueeze(0)
                    )
            else:
                # Per-point L2 (requires correspondence)
                # Match sizes if different
                min_len = min(len(recon), len(gt_targets))
                loss = torch.mean(
                    torch.sum((recon[:min_len] - gt_targets[:min_len]) ** 2, dim=-1)
                )

            total_damaged_loss += loss.item()
            total_loss += loss

            # Intact preservation check (should be ~0 for this method)
            intact_mask = gt_labels[b] == 0
            if intact_mask.sum() > 0:
                intact_error = torch.mean(
                    torch.sum(
                        (output_points[b][intact_mask] - gt_points[b][intact_mask]) ** 2,
                        dim=-1,
                    )
                )
                total_intact_loss += intact_error.item()

            n_valid += 1

        if n_valid == 0:
            return torch.tensor(0.0, device=device, requires_grad=True), {
                'loss': 0.0, 'damaged_loss': 0.0, 'intact_loss': 0.0
            }

        loss = total_loss / n_valid

        metrics = {
            'loss': loss.item(),
            'damaged_loss': total_damaged_loss / n_valid,
            'intact_loss': total_intact_loss / n_valid,
        }

        return loss, metrics
