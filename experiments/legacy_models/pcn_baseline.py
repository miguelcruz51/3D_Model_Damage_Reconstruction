"""
Standard PCN (Point Completion Network) Baseline

This is a faithful implementation of PCN [Yuan et al., 2018] adapted for
comparison with Hybrid PCN on the DMC dataset.

Key differences from Hybrid PCN:
- Single encoder (processes only damaged input)
- No nominal reference input
- No mask-guided blending
- Standard uniform Chamfer Distance loss

Reference:
Yuan, W., Khot, T., Held, D., Mertz, C., & Hebert, M. (2018).
PCN: Point completion network. In 3DV.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PCNEncoder(nn.Module):
    """
    PointNet-style encoder for PCN.
    Extracts global feature from partial input.
    """
    
    def __init__(self, output_dim=1024):
        super().__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv1d(3, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv1d(128, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv1d(256, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True)
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv1d(512, output_dim, 1),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        """
        Args:
            x: (B, N, 3) partial point cloud
        
        Returns:
            global_feat: (B, output_dim) global feature
        """
        B, N, _ = x.shape
        x = x.transpose(2, 1)  # (B, 3, N)
        
        x = self.conv1(x)      # (B, 128, N)
        x = self.conv2(x)      # (B, 256, N)
        x = self.conv3(x)      # (B, 512, N)
        x = self.conv4(x)      # (B, 1024, N)
        
        # Global max pooling
        global_feat = torch.max(x, dim=2)[0]  # (B, 1024)
        
        return global_feat


class PCN(nn.Module):
    """
    Standard Point Completion Network (PCN).
    
    Architecture:
    1. Encoder: Extract global feature from partial input
    2. Coarse decoder: Generate coarse complete point cloud
    3. Fine decoder: Refine to dense output using folding
    
    Args:
        num_dense: Number of output points (e.g., 4096, 8192, 16384)
        latent_dim: Dimension of latent feature (default: 1024)
        grid_size: Grid size for folding (default: 4, gives 16 points per coarse point)
    """
    
    def __init__(self, num_dense=4096, latent_dim=1024, grid_size=4):
        super().__init__()
        
        self.num_dense = num_dense
        self.latent_dim = latent_dim
        self.grid_size = grid_size
        self.num_coarse = num_dense // (grid_size ** 2)
        
        # Encoder
        self.encoder = PCNEncoder(output_dim=latent_dim)
        
        # Coarse decoder
        self.coarse_decoder = nn.Sequential(
            nn.Linear(latent_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 3 * self.num_coarse)
        )
        
        # Fine decoder (folding-based)
        self.fine_decoder = nn.Sequential(
            nn.Conv1d(latent_dim + 3 + 2, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 3, 1)
        )
        
        # Folding seed (2D grid)
        a = torch.linspace(-0.05, 0.05, steps=grid_size, dtype=torch.float).view(1, grid_size).expand(grid_size, grid_size).reshape(1, -1)
        b = torch.linspace(-0.05, 0.05, steps=grid_size, dtype=torch.float).view(grid_size, 1).expand(grid_size, grid_size).reshape(1, -1)
        seed = torch.cat([a, b], dim=0).view(1, 2, grid_size ** 2)
        self.register_buffer('folding_seed', seed)
    
    def forward(self, partial):
        """
        Args:
            partial: (B, N, 3) partial/damaged point cloud
        
        Returns:
            coarse: (B, num_coarse, 3) coarse completion
            fine: (B, num_dense, 3) fine completion
        """
        B = partial.shape[0]
        
        # 1. Encode partial input
        global_feat = self.encoder(partial)  # (B, latent_dim)
        
        # 2. Coarse completion
        coarse = self.coarse_decoder(global_feat)  # (B, 3 * num_coarse)
        coarse = coarse.reshape(B, self.num_coarse, 3)  # (B, num_coarse, 3)
        
        # 3. Fine completion (folding-based upsampling)
        # Expand coarse points
        point_feat = coarse.unsqueeze(2).expand(-1, -1, self.grid_size ** 2, -1)  # (B, num_coarse, S, 3)
        point_feat = point_feat.reshape(B, self.num_dense, 3).transpose(2, 1)  # (B, 3, num_dense)
        
        # Expand folding seed
        seed = self.folding_seed.unsqueeze(2).expand(B, -1, self.num_coarse, -1)  # (B, 2, num_coarse, S)
        seed = seed.reshape(B, -1, self.num_dense)  # (B, 2, num_dense)
        
        # Expand global features
        global_feat_exp = global_feat.unsqueeze(2).expand(-1, -1, self.num_dense)  # (B, latent_dim, num_dense)
        
        # Concatenate features
        feat = torch.cat([global_feat_exp, seed, point_feat], dim=1)  # (B, latent_dim+2+3, num_dense)
        
        # Decode fine points with residual connection
        fine = self.fine_decoder(feat) + point_feat  # (B, 3, num_dense)
        fine = fine.transpose(1, 2)  # (B, num_dense, 3)
        
        return coarse, fine


class PCNLoss(nn.Module):
    """
    Standard PCN loss function.
    
    Loss = α * CD_coarse + β * CD_fine
    
    Uses standard uniform Chamfer Distance (no damage-aware weighting).
    """
    
    def __init__(self, coarse_weight=1.0, fine_weight=1.0):
        super().__init__()
        self.coarse_weight = coarse_weight
        self.fine_weight = fine_weight
    
    def chamfer_distance_single_direction(self, pred, gt, chunk_size=256):
        """
        Compute single-direction Chamfer Distance (memory-efficient).
        """
        B, N, _ = pred.shape
        M = gt.shape[1]
        
        min_distances_list = []
        
        for i in range(0, N, chunk_size):
            end_i = min(i + chunk_size, N)
            pred_chunk = pred[:, i:end_i, :]  # (B, chunk, 3)
            
            pred_exp = pred_chunk.unsqueeze(2)  # (B, chunk, 1, 3)
            gt_exp = gt.unsqueeze(1)  # (B, 1, M, 3)
            
            distances = torch.norm(pred_exp - gt_exp, dim=3)  # (B, chunk, M)
            min_distances, _ = torch.min(distances, dim=2)  # (B, chunk)
            min_distances_list.append(min_distances)
        
        min_distances = torch.cat(min_distances_list, dim=1)  # (B, N)
        return min_distances
    
    def chamfer_distance_L1(self, pred, gt):
        """
        Compute bidirectional L1 Chamfer Distance.
        """
        cd_pred_to_gt = self.chamfer_distance_single_direction(pred, gt)
        cd_gt_to_pred = self.chamfer_distance_single_direction(gt, pred)
        
        cd_loss = cd_pred_to_gt.mean() + cd_gt_to_pred.mean()
        return cd_loss
    
    def forward(self, coarse_pred, fine_pred, gt):
        """
        Args:
            coarse_pred: (B, N_coarse, 3) coarse prediction
            fine_pred: (B, N_fine, 3) fine prediction
            gt: (B, N_fine, 3) ground truth
        
        Returns:
            total_loss: combined loss
            loss_dict: dictionary of individual losses
        """
        # Coarse loss
        loss_coarse = self.chamfer_distance_L1(coarse_pred, gt)
        
        # Fine loss
        loss_fine = self.chamfer_distance_L1(fine_pred, gt)
        
        # Total loss
        total_loss = self.coarse_weight * loss_coarse + self.fine_weight * loss_fine
        
        # Return individual losses for logging
        loss_dict = {
            'coarse': loss_coarse.item(),
            'fine': loss_fine.item(),
            'total': total_loss.item()
        }
        
        return total_loss, loss_dict


if __name__ == '__main__':
    # Test PCN model
    print("Testing PCN model...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    model = PCN(num_dense=4096, latent_dim=1024, grid_size=4).to(device)
    
    # Create dummy input
    batch_size = 4
    num_input = 2048
    partial = torch.randn(batch_size, num_input, 3).to(device)
    
    # Forward pass
    coarse, fine = model(partial)
    
    print(f"Input shape: {partial.shape}")
    print(f"Coarse output shape: {coarse.shape}")
    print(f"Fine output shape: {fine.shape}")
    
    # Test loss
    criterion = PCNLoss()
    gt = torch.randn(batch_size, 4096, 3).to(device)
    loss, loss_dict = criterion(coarse, fine, gt)
    
    print(f"\nLoss: {loss.item():.4f}")
    print(f"Loss dict: {loss_dict}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    print("\n✓ PCN model test passed!")
