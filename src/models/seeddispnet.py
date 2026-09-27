"""Seed-based Displacement Network (SeedDispNet, M7 - proposed method).

Each damaged point is treated as a "seed". The network predicts a per-seed
displacement that moves the seed to its reconstructed nominal position.

Key properties:
- Generates exactly ``N_damaged`` reconstructed points (no waste).
- Maintains a 1-to-1 correspondence with the ground-truth reconstruction targets.
- Intact points (label=0) are preserved exactly.

Architecture:
1. Global PointNet encoder on the full labelled cloud produces a global feature.
2. A per-point local encoder produces local features.
3. For each damaged seed, ``[seed_xyz, local_feat, global_feat]`` is fed through an
   MLP to predict a displacement; the reconstructed seed is ``seed + displacement``.

This module contains only the model definitions; the associated loss lives in
``src/losses/seed_displacement_loss.py``.
"""

from typing import List, Tuple

import torch
import torch.nn as nn


class SeedDisplacementNet(nn.Module):
    """Seed-based displacement network for local reconstruction (M7).

    Each damaged point (label=1) is used as a seed. The network predicts where
    that seed should move to. Intact points (label=0) are preserved unchanged.
    """

    def __init__(self, latent_dim: int = 512, feature_dim: int = 256) -> None:
        super().__init__()

        self.latent_dim = latent_dim

        # Global encoder: full cloud with labels -> global feature
        self.encoder = nn.Sequential(
            nn.Conv1d(4, 64, 1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, 256, 1), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Conv1d(256, latent_dim, 1), nn.BatchNorm1d(latent_dim), nn.ReLU(),
        )

        # Per-point local encoder: extract features per point
        self.local_encoder = nn.Sequential(
            nn.Conv1d(4, 64, 1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, feature_dim, 1), nn.BatchNorm1d(feature_dim), nn.ReLU(),
        )

        # Displacement predictor:
        # [seed_xyz(3) + local_feat(feature_dim) + global_feat(latent_dim)] -> displacement(3)
        input_dim = 3 + feature_dim + latent_dim
        self.displacement_mlp = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 3),
        )

    def forward(
        self,
        points: torch.Tensor,
        labels: torch.Tensor,
        training: bool = True,
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        """Run the seed-displacement forward pass.

        Args:
            points: ``(B, N, 3)`` damaged partition.
            labels: ``(B, N)`` binary labels (0=intact, 1=damaged).
            training: unused, kept for API compatibility.

        Returns:
            output_points: ``(B, N, 3)`` corrected partition.
            reconstructed_list: list of ``(K_b, 3)`` reconstructed points per batch.
            damaged_masks: list of boolean masks marking damaged points per batch.
        """
        B, N, _ = points.shape

        # Concatenate points with labels
        x = torch.cat([points, labels.unsqueeze(-1)], dim=-1)  # (B, N, 4)
        x_t = x.transpose(1, 2)  # (B, 4, N)

        # Global feature
        global_feat = self.encoder(x_t)  # (B, latent_dim, N)
        global_feat = torch.max(global_feat, dim=2)[0]  # (B, latent_dim)

        # Local per-point features
        local_feat = self.local_encoder(x_t)  # (B, feature_dim, N)
        local_feat = local_feat.transpose(1, 2)  # (B, N, feature_dim)

        # Process each sample in batch
        output_points = points.clone()
        reconstructed_list: List[torch.Tensor] = []
        damaged_masks: List[torch.Tensor] = []

        for b in range(B):
            damaged_mask = labels[b] == 1
            n_damaged = damaged_mask.sum().item()

            if n_damaged == 0:
                reconstructed_list.append(torch.zeros(0, 3, device=points.device))
                damaged_masks.append(damaged_mask)
                continue

            # Get seeds (damaged points)
            seeds = points[b][damaged_mask]  # (K, 3)
            seed_local = local_feat[b][damaged_mask]  # (K, feature_dim)

            # Expand global feature for each seed
            seed_global = global_feat[b].unsqueeze(0).expand(n_damaged, -1)  # (K, latent_dim)

            # Concatenate: [seed_xyz, local_feat, global_feat]
            seed_input = torch.cat([seeds, seed_local, seed_global], dim=-1)  # (K, input_dim)

            # Predict displacement per seed
            displacement = self.displacement_mlp(seed_input)  # (K, 3)

            # Apply displacement
            reconstructed = seeds + displacement  # (K, 3)
            reconstructed_list.append(reconstructed)
            damaged_masks.append(damaged_mask)

            # Update output
            output_points[b][damaged_mask] = reconstructed

        return output_points, reconstructed_list, damaged_masks


class SeedDisplacementNetNoLocal(nn.Module):
    """Ablation of :class:`SeedDisplacementNet` without local per-point features.

    Only uses ``seed_xyz`` and the global feature (no local encoder). Tests the
    contribution of the local features to reconstruction quality.
    """

    def __init__(self, latent_dim: int = 512, feature_dim: int = 256) -> None:
        super().__init__()
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Conv1d(4, 64, 1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, 256, 1), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Conv1d(256, latent_dim, 1), nn.BatchNorm1d(latent_dim), nn.ReLU(),
        )

        # No local encoder - only seed_xyz(3) + global_feat(latent_dim)
        input_dim = 3 + latent_dim
        self.displacement_mlp = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 3),
        )

    def forward(
        self,
        points: torch.Tensor,
        labels: torch.Tensor,
        training: bool = True,
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        """Run the no-local-feature seed-displacement forward pass.

        Args:
            points: ``(B, N, 3)`` damaged partition.
            labels: ``(B, N)`` binary labels (0=intact, 1=damaged).
            training: unused, kept for API compatibility.

        Returns:
            output_points: ``(B, N, 3)`` corrected partition.
            reconstructed_list: list of ``(K_b, 3)`` reconstructed points per batch.
            damaged_masks: list of boolean masks marking damaged points per batch.
        """
        B, N, _ = points.shape
        x = torch.cat([points, labels.unsqueeze(-1)], dim=-1)
        x_t = x.transpose(1, 2)

        global_feat = self.encoder(x_t)
        global_feat = torch.max(global_feat, dim=2)[0]

        output_points = points.clone()
        reconstructed_list: List[torch.Tensor] = []
        damaged_masks: List[torch.Tensor] = []

        for b in range(B):
            damaged_mask = labels[b] == 1
            n_damaged = damaged_mask.sum().item()

            if n_damaged == 0:
                reconstructed_list.append(torch.zeros(0, 3, device=points.device))
                damaged_masks.append(damaged_mask)
                continue

            seeds = points[b][damaged_mask]
            seed_global = global_feat[b].unsqueeze(0).expand(n_damaged, -1)
            seed_input = torch.cat([seeds, seed_global], dim=-1)

            displacement = self.displacement_mlp(seed_input)
            reconstructed = seeds + displacement
            reconstructed_list.append(reconstructed)
            damaged_masks.append(damaged_mask)
            output_points[b][damaged_mask] = reconstructed

        return output_points, reconstructed_list, damaged_masks
