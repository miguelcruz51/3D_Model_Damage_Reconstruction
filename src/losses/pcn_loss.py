"""Loss for the Pure PCN completion network (M1).

Standard PCN loss: a **squared Chamfer distance** computed on the full point
cloud for both the coarse and fine predictions, combined as
``cd_fine + coarse_weight * cd_coarse``.

Compared quantities: the coarse prediction (256 points) and the fine prediction
(``num_points``) are each compared against the full ground-truth cloud using the
squared Chamfer distance.
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from .chamfer import chamfer_distance_squared


class PurePCNLoss(nn.Module):
    """Standard PCN loss: squared Chamfer distance on the full point cloud.

    Combines a coarse-prediction term and a fine-prediction term.

    Args:
        coarse_weight: weight applied to the coarse Chamfer-distance term.
    """

    def __init__(self, coarse_weight: float = 0.5) -> None:
        super().__init__()
        self.coarse_weight = coarse_weight

    def forward(
        self,
        coarse_points: torch.Tensor,
        fine_points: torch.Tensor,
        gt_points: torch.Tensor,
        gt_labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute the PCN completion loss.

        Args:
            coarse_points: ``(B, 256, 3)`` coarse prediction.
            fine_points: ``(B, N, 3)`` fine prediction.
            gt_points: ``(B, N, 3)`` full ground truth.
            gt_labels: ignored, kept for API compatibility.

        Returns:
            A tuple ``(loss, metrics)`` where ``loss`` is a scalar tensor and
            ``metrics`` reports the coarse and fine Chamfer distances.
        """
        cd_coarse = chamfer_distance_squared(coarse_points, gt_points)
        cd_fine = chamfer_distance_squared(fine_points, gt_points)

        loss = cd_fine + self.coarse_weight * cd_coarse

        metrics = {
            'loss': loss.item(),
            'cd_coarse': cd_coarse.item(),
            'cd_fine': cd_fine.item(),
        }

        return loss, metrics
