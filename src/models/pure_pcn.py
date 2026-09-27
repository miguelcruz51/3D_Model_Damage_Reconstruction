"""Pure PCN (Point Completion Network) - Yuan et al., 2018 (baseline M1).

Reconstructs the ENTIRE point cloud from a partial input. No labels, no intact
preservation - pure completion.

Input:  Partial point cloud ``(N, 3)`` - damaged points erased.
Output: Complete point cloud ``(M, 3)`` - full reconstruction.

This module contains only the model definition; the associated loss lives in
``src/losses/pcn_loss.py``.
"""

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class PurePCN(nn.Module):
    """Standard PCN that reconstructs the full point cloud.

    Architecture:
    - PointNet encoder -> ``latent_dim`` global feature.
    - Coarse decoder -> 256 points.
    - Fine decoder (folding) -> ``num_points`` points.
    """

    def __init__(self, num_points: int = 2048, latent_dim: int = 512) -> None:
        super().__init__()

        self.num_points = num_points
        self.latent_dim = latent_dim

        # Encoder
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)
        self.conv4 = nn.Conv1d(256, latent_dim, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(latent_dim)

        # Coarse decoder: global feature -> 256 coarse points
        self.num_coarse = 256
        self.fc1 = nn.Linear(latent_dim, 1024)
        self.fc2 = nn.Linear(1024, self.num_coarse * 3)
        self.fc_bn1 = nn.BatchNorm1d(1024)

        # Fine decoder: refine coarse points to num_points
        # Each coarse point generates num_points/num_coarse fine points
        self.points_per_coarse = num_points // self.num_coarse  # 8

        # Fine MLP: takes coarse point (3) + global feature (latent_dim) -> fine offsets
        self.fine_fc1 = nn.Conv1d(3 + latent_dim, 256, 1)
        self.fine_fc2 = nn.Conv1d(256, 128, 1)
        self.fine_fc3 = nn.Conv1d(128, self.points_per_coarse * 3, 1)

        self.fine_bn1 = nn.BatchNorm1d(256)
        self.fine_bn2 = nn.BatchNorm1d(128)

    def forward(
        self,
        points: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        training: bool = True,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Run the PCN forward pass.

        Args:
            points: ``(B, N, 3)`` partial point cloud.
            labels: ignored, kept for API compatibility.
            training: if ``True`` return ``(coarse, fine)``, else just ``fine``.

        Returns:
            If ``training`` is ``True``: ``(coarse_points, fine_points)``.
            Otherwise: ``fine_points``.
        """
        B = points.shape[0]

        # Encode
        x = points.transpose(1, 2)  # (B, 3, N)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        global_feature = torch.max(x, dim=2)[0]  # (B, latent_dim)

        # Coarse decode
        coarse = F.relu(self.fc_bn1(self.fc1(global_feature)))
        coarse = self.fc2(coarse)
        coarse_points = coarse.reshape(B, self.num_coarse, 3)  # (B, 256, 3)

        # Fine decode
        # Expand global feature for each coarse point
        global_expanded = global_feature.unsqueeze(2).expand(-1, -1, self.num_coarse)  # (B, latent_dim, 256)
        coarse_t = coarse_points.transpose(1, 2)  # (B, 3, 256)

        fine_input = torch.cat([coarse_t, global_expanded], dim=1)  # (B, 3+latent_dim, 256)

        fine = F.relu(self.fine_bn1(self.fine_fc1(fine_input)))
        fine = F.relu(self.fine_bn2(self.fine_fc2(fine)))
        fine = self.fine_fc3(fine)  # (B, points_per_coarse*3, 256)

        # Reshape: each coarse point generates points_per_coarse fine points
        fine = fine.transpose(1, 2)  # (B, 256, points_per_coarse*3)
        fine = fine.reshape(B, self.num_coarse, self.points_per_coarse, 3)

        # Add coarse points as centers
        fine_points = fine + coarse_points.unsqueeze(2)
        fine_points = fine_points.reshape(B, self.num_points, 3)

        if training:
            return coarse_points, fine_points
        else:
            return fine_points
