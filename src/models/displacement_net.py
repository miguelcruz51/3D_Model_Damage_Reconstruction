"""Displacement-based networks for local surface reconstruction (baselines M4/M5).

Variants:
1. LabelGuidedDisplacementNet: uses damage labels to predict displacements ONLY
   for damaged points; intact points pass through unchanged.
2. BlindDisplacementNet (M4, PointCleanNet-style): predicts displacements for ALL
   points (no label awareness) - a denoising approach.
3. FlowDisplacementNet (M5, FlowNet3D-style): uses local k-NN neighborhood
   features to predict per-point displacements.

All variants share the PointNet backbone / displacement head defined here. The
associated loss lives in ``src/losses/displacement_loss.py``.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PointNetPerPointEncoder(nn.Module):
    """PointNet encoder producing per-point features (local + global context)."""

    def __init__(self, input_dim: int = 4, feature_dim: int = 256) -> None:
        super().__init__()

        # Local feature extraction
        self.conv1 = nn.Conv1d(input_dim, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)

        # After concatenating local (256) + global (256) features
        self.conv4 = nn.Conv1d(512, 256, 1)
        self.conv5 = nn.Conv1d(256, feature_dim, 1)

        self.bn4 = nn.BatchNorm1d(256)
        self.bn5 = nn.BatchNorm1d(feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute per-point features.

        Args:
            x: ``(B, N, input_dim)`` input points (optionally with labels).

        Returns:
            ``(B, N, feature_dim)`` per-point features.
        """
        B, N, _ = x.shape
        x = x.transpose(1, 2)  # (B, C, N)

        # Local features
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        local_features = F.relu(self.bn3(self.conv3(x)))  # (B, 256, N)

        # Global feature
        global_feature = torch.max(local_features, dim=2, keepdim=True)[0]  # (B, 256, 1)
        global_feature = global_feature.expand(-1, -1, N)  # (B, 256, N)

        # Concatenate local + global
        combined = torch.cat([local_features, global_feature], dim=1)  # (B, 512, N)

        # Per-point features
        combined = F.relu(self.bn4(self.conv4(combined)))
        combined = F.relu(self.bn5(self.conv5(combined)))  # (B, feature_dim, N)

        return combined.transpose(1, 2)  # (B, N, feature_dim)


class DisplacementHead(nn.Module):
    """Predicts a per-point 3D displacement from per-point features."""

    def __init__(self, feature_dim: int = 256) -> None:
        super().__init__()

        self.conv1 = nn.Conv1d(feature_dim, 128, 1)
        self.conv2 = nn.Conv1d(128, 64, 1)
        self.conv3 = nn.Conv1d(64, 3, 1)

        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(64)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Map per-point features to per-point displacements.

        Args:
            features: ``(B, N, feature_dim)`` per-point features.

        Returns:
            ``(B, N, 3)`` displacements.
        """
        x = features.transpose(1, 2)  # (B, feature_dim, N)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.conv3(x)  # (B, 3, N) - no activation, displacements can be any value
        return x.transpose(1, 2)  # (B, N, 3)


class LabelGuidedDisplacementNet(nn.Module):
    """Label-Guided Displacement Network.

    Uses damage labels to predict displacements ONLY for damaged points; intact
    points (label=0) pass through unchanged.

    For damaged points: ``output = input + predicted_displacement``.
    For intact points:  ``output = input``.
    """

    def __init__(self, feature_dim: int = 256) -> None:
        super().__init__()

        # Input: xyz (3) + label (1) = 4
        self.encoder = PointNetPerPointEncoder(input_dim=4, feature_dim=feature_dim)
        self.displacement_head = DisplacementHead(feature_dim=feature_dim)

    def forward(
        self,
        points: torch.Tensor,
        labels: torch.Tensor,
        training: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run the label-guided displacement forward pass.

        Args:
            points: ``(B, N, 3)`` damaged point cloud.
            labels: ``(B, N)`` binary labels (0=intact, 1=damaged).
            training: unused, kept for API compatibility.

        Returns:
            output_points: ``(B, N, 3)`` corrected point cloud.
            displacements: ``(B, N, 3)`` predicted displacements (for loss).
        """
        # Concatenate points with labels
        x = torch.cat([points, labels.unsqueeze(-1)], dim=-1)  # (B, N, 4)

        # Extract per-point features
        features = self.encoder(x)  # (B, N, feature_dim)

        # Predict displacements for all points
        displacements = self.displacement_head(features)  # (B, N, 3)

        # Apply displacements ONLY to damaged points (label=1)
        mask = labels.unsqueeze(-1)  # (B, N, 1)
        output_points = points + displacements * mask

        return output_points, displacements


class BlindDisplacementNet(nn.Module):
    """Blind Displacement Network (PointCleanNet-style baseline, M4).

    Predicts displacements for ALL points regardless of labels - a denoising
    approach with no label awareness.
    """

    def __init__(self, feature_dim: int = 256) -> None:
        super().__init__()

        # Input: xyz only (3) - no labels
        self.encoder = PointNetPerPointEncoder(input_dim=3, feature_dim=feature_dim)
        self.displacement_head = DisplacementHead(feature_dim=feature_dim)

    def forward(
        self,
        points: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        training: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run the blind displacement forward pass.

        Args:
            points: ``(B, N, 3)`` damaged point cloud.
            labels: ignored, kept for API compatibility.
            training: unused, kept for API compatibility.

        Returns:
            output_points: ``(B, N, 3)`` corrected point cloud.
            displacements: ``(B, N, 3)`` predicted displacements (for loss).
        """
        # Extract per-point features (no labels)
        features = self.encoder(points)  # (B, N, feature_dim)

        # Predict displacements for ALL points
        displacements = self.displacement_head(features)  # (B, N, 3)

        # Apply displacements to all points
        output_points = points + displacements

        return output_points, displacements


class FlowDisplacementNet(nn.Module):
    """FlowNet3D-style Displacement Network (baseline M5).

    Uses local neighborhood features (k-NN) to predict per-point displacements.
    No label awareness - relies on local geometric context. Each point's
    displacement is informed by its neighbors, similar to scene flow estimation.
    """

    def __init__(self, feature_dim: int = 256, k: int = 16) -> None:
        super().__init__()
        self.k = k

        # Local neighborhood feature extraction
        # Input: xyz (3) + neighbor_relative_xyz (3) = 6 per neighbor
        self.edge_conv1 = nn.Conv2d(6, 64, 1)
        self.edge_conv2 = nn.Conv2d(64, 128, 1)
        self.edge_bn1 = nn.BatchNorm2d(64)
        self.edge_bn2 = nn.BatchNorm2d(128)

        # Point feature refinement (after max-pooling over neighbors)
        # 128 (local) + 3 (xyz) = 131
        self.conv1 = nn.Conv1d(131, 256, 1)
        self.conv2 = nn.Conv1d(256, feature_dim, 1)
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(feature_dim)

        # Global context
        self.conv_global = nn.Conv1d(feature_dim, feature_dim, 1)
        self.bn_global = nn.BatchNorm1d(feature_dim)

        # After concat: local (feature_dim) + global (feature_dim) = 2*feature_dim
        self.conv3 = nn.Conv1d(feature_dim * 2, 256, 1)
        self.conv4 = nn.Conv1d(256, 128, 1)
        self.conv5 = nn.Conv1d(128, 3, 1)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(128)

    def _knn(self, points: torch.Tensor, k: int) -> torch.Tensor:
        """Find the indices of the ``k`` nearest neighbors for each point."""
        # points: (B, N, 3)
        dist = torch.cdist(points, points)  # (B, N, N)
        _, idx = dist.topk(k, dim=-1, largest=False)  # (B, N, k)
        return idx

    def forward(
        self,
        points: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        training: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run the flow-based displacement forward pass.

        Args:
            points: ``(B, N, 3)`` damaged point cloud.
            labels: ignored, kept for API compatibility.
            training: unused, kept for API compatibility.

        Returns:
            output_points: ``(B, N, 3)`` corrected point cloud.
            displacements: ``(B, N, 3)`` predicted displacements (for loss).
        """
        B, N, _ = points.shape

        # Find k-NN
        knn_idx = self._knn(points, self.k)  # (B, N, k)

        # Gather neighbor points
        knn_idx_expanded = knn_idx.unsqueeze(-1).expand(-1, -1, -1, 3)  # (B, N, k, 3)
        neighbors = torch.gather(
            points.unsqueeze(2).expand(-1, -1, self.k, -1),
            1,
            knn_idx_expanded.reshape(B, -1, 3).unsqueeze(2).expand(-1, -1, self.k, -1)
        )
        # Simpler gather
        neighbors = points.unsqueeze(2).expand(-1, -1, N, -1)  # (B, N, N, 3)
        neighbors = torch.gather(neighbors, 2, knn_idx_expanded)  # (B, N, k, 3)

        # Relative positions
        center = points.unsqueeze(2).expand(-1, -1, self.k, -1)  # (B, N, k, 3)
        relative = neighbors - center  # (B, N, k, 3)

        # Edge features: [center_xyz, relative_xyz]
        edge_features = torch.cat([center, relative], dim=-1)  # (B, N, k, 6)
        edge_features = edge_features.permute(0, 3, 1, 2)  # (B, 6, N, k)

        # Edge convolutions
        edge_features = F.relu(self.edge_bn1(self.edge_conv1(edge_features)))
        edge_features = F.relu(self.edge_bn2(self.edge_conv2(edge_features)))  # (B, 128, N, k)

        # Max pool over neighbors
        local_features = edge_features.max(dim=3)[0]  # (B, 128, N)

        # Concatenate with point coordinates
        pts_t = points.transpose(1, 2)  # (B, 3, N)
        combined = torch.cat([local_features, pts_t], dim=1)  # (B, 131, N)

        # Point feature refinement
        point_features = F.relu(self.bn1(self.conv1(combined)))
        point_features = F.relu(self.bn2(self.conv2(point_features)))  # (B, feature_dim, N)

        # Global context
        global_feature = F.relu(self.bn_global(self.conv_global(point_features)))
        global_feature = global_feature.max(dim=2, keepdim=True)[0]  # (B, feature_dim, 1)
        global_feature = global_feature.expand(-1, -1, N)  # (B, feature_dim, N)

        # Combine local + global
        combined = torch.cat([point_features, global_feature], dim=1)  # (B, 2*feature_dim, N)

        # Predict displacements
        x = F.relu(self.bn3(self.conv3(combined)))
        x = F.relu(self.bn4(self.conv4(x)))
        displacements = self.conv5(x).transpose(1, 2)  # (B, N, 3)

        output_points = points + displacements

        return output_points, displacements
