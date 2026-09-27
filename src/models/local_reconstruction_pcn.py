"""Local Reconstruction PCN (LR-PCN, baseline M3).

This model learns to reconstruct ONLY damaged regions by:

1. Taking a damaged partition + damage labels as input.
2. Predicting reconstruction points for the damaged region.
3. Using GT labels to supervise reconstruction quality.

Key design:
- Encoder: processes ALL points (intact + damaged) to learn geometric context.
  Intact points (label=0) provide shape information; damaged points (label=1)
  indicate where reconstruction is needed.
- Decoder: generates ONLY reconstruction points for damaged regions, conditioned
  on the damage location (centroid + spread).
- Intact geometry is preserved and merged back at inference time.

This module contains only the model and its sub-modules; the associated loss
lives in ``src/losses/label_aware_loss.py``.
"""

from typing import List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class LocalReconstructionPCN(nn.Module):
    """Point completion network for local damage reconstruction.

    Only reconstructs damaged points (label=1) and preserves intact points
    (label=0).

    Architecture:
    - Encoder: extract features from the damaged partition (with labels).
    - Damage localizer: extract spatial info from the damaged region (centroid, spread).
    - Decoder: generate reconstruction points conditioned on the damage location.
    - Merge: combine intact points + reconstructed points (inference only).
    """

    def __init__(
        self,
        num_points: int = 2048,
        latent_dim: int = 512,
        predict_ratio: float = 0.5,
    ) -> None:
        super().__init__()

        self.num_points = num_points
        self.latent_dim = latent_dim
        self.predict_ratio = predict_ratio

        # Encoder: damaged points + labels -> latent features
        self.encoder = PointNetEncoder(latent_dim=latent_dim, input_dim=4)

        # Damage-conditioned decoder
        # Input: global features (latent_dim) + damage spatial info (6: centroid + std)
        self.num_predict = int(num_points * predict_ratio)
        self.decoder = SpatialDecoder(
            latent_dim=latent_dim,
            spatial_dim=6,
            num_points=self.num_predict,
        )

    def forward(
        self,
        damaged_points: torch.Tensor,
        damaged_labels: torch.Tensor,
        training: bool = True,
    ) -> Union[Tuple[List[torch.Tensor], torch.Tensor], torch.Tensor]:
        """Run the local reconstruction forward pass.

        Args:
            damaged_points: ``(B, N, 3)`` damaged partition.
            damaged_labels: ``(B, N)`` binary labels (0=intact, 1=damaged).
            training: if ``True`` return ``(intact_points_list, reconstructed_points)``,
                else return the merged ``(B, num_points, 3)`` cloud.

        Returns:
            Training mode: tuple of the per-batch intact points list and the
            ``(B, num_predict, 3)`` reconstructed points.
            Inference mode: ``(B, num_points, 3)`` merged output cloud.
        """
        batch_size = damaged_points.shape[0]

        # Concatenate points with labels
        damaged_with_labels = torch.cat([
            damaged_points,
            damaged_labels.unsqueeze(-1),
        ], dim=-1)

        # Encode
        features = self.encoder(damaged_with_labels)  # (B, latent_dim)

        # Extract spatial info from damaged regions
        spatial_info = []
        intact_points_list: List[torch.Tensor] = []
        for b in range(batch_size):
            damaged_mask = damaged_labels[b] == 1
            intact_mask = damaged_labels[b] == 0

            intact_pts = damaged_points[b][intact_mask]
            intact_points_list.append(intact_pts)

            if damaged_mask.sum() > 1:
                dmg_pts = damaged_points[b][damaged_mask]
                centroid = dmg_pts.mean(dim=0)  # (3,)
                spread = dmg_pts.std(dim=0).clamp(min=1e-3)  # (3,) avoid zero std
            elif damaged_mask.sum() == 1:
                centroid = damaged_points[b][damaged_mask][0]
                spread = damaged_points[b].std(dim=0).clamp(min=1e-3)
            else:
                centroid = damaged_points[b].mean(dim=0)
                spread = damaged_points[b].std(dim=0).clamp(min=1e-3)

            spatial_info.append(torch.cat([centroid, spread], dim=0))

        spatial_info = torch.stack(spatial_info, dim=0)  # (B, 6)

        # Decode with spatial conditioning
        reconstructed_points = self.decoder(features, spatial_info)

        if training:
            return intact_points_list, reconstructed_points
        else:
            output_list = []
            for b in range(batch_size):
                intact_pts = intact_points_list[b]       # (K_intact, 3)
                recon_pts = reconstructed_points[b]      # (K_recon, 3)
                n_intact = len(intact_pts)
                n_recon = len(recon_pts)
                n_total = n_intact + n_recon

                if n_total > self.num_points:
                    # Keep ALL intact points, truncate only the reconstructed ones.
                    # This preserves index alignment: pred[intact_mask] == input[intact_mask].
                    n_recon_keep = self.num_points - n_intact
                    if n_recon_keep > 0:
                        recon_pts = recon_pts[:n_recon_keep]
                    else:
                        # Edge case: more intact points than num_points - truncate intact too
                        intact_pts = intact_pts[:self.num_points]
                        recon_pts = recon_pts[:0]
                    combined = torch.cat([intact_pts, recon_pts], dim=0)
                elif n_total < self.num_points:
                    # Pad by repeating reconstructed points (never touch intact)
                    shortage = self.num_points - n_total
                    repeats = (shortage + n_recon - 1) // n_recon if n_recon > 0 else 1
                    recon_pad = recon_pts.repeat(repeats + 1, 1)[:n_recon + shortage]
                    combined = torch.cat([intact_pts, recon_pad], dim=0)
                else:
                    combined = torch.cat([intact_pts, recon_pts], dim=0)

                output_list.append(combined)

            return torch.stack(output_list, dim=0)


class PointNetEncoder(nn.Module):
    """PointNet-style encoder with label awareness."""

    def __init__(self, latent_dim: int = 512, input_dim: int = 4) -> None:
        super().__init__()

        # Shared MLPs
        self.conv1 = nn.Conv1d(input_dim, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)
        self.conv4 = nn.Conv1d(256, latent_dim, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a labelled point cloud into a global feature.

        Args:
            x: ``(B, N, input_dim)`` points with labels.

        Returns:
            ``(B, latent_dim)`` global features.
        """
        # Transpose for conv1d: (B, N, C) -> (B, C, N)
        x = x.transpose(1, 2)

        # Shared MLPs
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))

        # Max pooling: (B, latent_dim, N) -> (B, latent_dim)
        x = torch.max(x, dim=2)[0]

        return x


class MLPDecoder(nn.Module):
    """MLP decoder to generate a complete point cloud."""

    def __init__(self, latent_dim: int = 512, num_points: int = 2048) -> None:
        super().__init__()

        self.num_points = num_points

        self.fc1 = nn.Linear(latent_dim, 1024)
        self.fc2 = nn.Linear(1024, 2048)
        self.fc3 = nn.Linear(2048, num_points * 3)

        self.bn1 = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d(2048)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Decode global features into a ``(B, num_points, 3)`` cloud."""
        x = F.relu(self.bn1(self.fc1(features)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        x = x.reshape(-1, self.num_points, 3)
        return x


class SpatialDecoder(nn.Module):
    """Spatially-conditioned decoder.

    Takes global features + damage spatial info (centroid, spread) and generates
    points centered around the damage location. This ensures reconstruction
    points are placed WHERE the damage is, not spread across the whole partition.
    """

    def __init__(
        self,
        latent_dim: int = 512,
        spatial_dim: int = 6,
        num_points: int = 1024,
    ) -> None:
        super().__init__()

        self.num_points = num_points

        # Combine global features with spatial info
        combined_dim = latent_dim + spatial_dim

        self.fc1 = nn.Linear(combined_dim, 1024)
        self.fc2 = nn.Linear(1024, 2048)
        self.fc3 = nn.Linear(2048, num_points * 3)  # Predict offsets from centroid

        self.bn1 = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d(2048)

    def forward(self, features: torch.Tensor, spatial_info: torch.Tensor) -> torch.Tensor:
        """Decode features + spatial info into reconstruction points.

        Args:
            features: ``(B, latent_dim)`` global features.
            spatial_info: ``(B, 6)`` ``[centroid_xyz, spread_xyz]``.

        Returns:
            ``(B, num_points, 3)`` reconstructed points in original space.
        """
        # Extract centroid and spread
        centroid = spatial_info[:, :3]  # (B, 3)
        spread = spatial_info[:, 3:].clamp(min=1e-3)   # (B, 3) avoid zero

        # Concatenate features with spatial info
        x = torch.cat([features, spatial_info], dim=1)  # (B, latent_dim + 6)

        # Generate offsets from centroid
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = torch.tanh(self.fc3(x))  # Bound offsets to [-1, 1]

        # Reshape to point cloud offsets
        offsets = x.reshape(-1, self.num_points, 3)  # (B, num_points, 3)

        # Scale offsets by spread and shift to centroid
        points = offsets * spread.unsqueeze(1) + centroid.unsqueeze(1)

        return points
