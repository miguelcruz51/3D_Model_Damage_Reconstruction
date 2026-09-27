"""
Hybrid Damage-Aware PCN for Local Point Cloud Reconstruction

Key differences from standard PCN:
1. Dual encoder: processes damaged and nominal point clouds separately
2. Damage detection: identifies which regions need reconstruction
3. Local decoder: reconstructs only damaged patches
4. Blending: merges reconstructed patches with preserved undamaged regions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DamageDetector(nn.Module):
    """
    Detects damaged regions by comparing damaged vs nominal point clouds.
    
    Two modes:
    1. Distance-based (automatic): Points far from nominal are damaged
    2. Label-based (supervised): Use provided damage labels
    """
    
    def __init__(self, mode='distance', threshold=0.02):
        super().__init__()
        self.mode = mode
        self.threshold = threshold
        
        if mode == 'learned':
            # Learnable damage detection
            self.detector = nn.Sequential(
                nn.Conv1d(6, 64, 1),  # concat damaged + nominal features
                nn.BatchNorm1d(64),
                nn.ReLU(inplace=True),
                nn.Conv1d(64, 32, 1),
                nn.BatchNorm1d(32),
                nn.ReLU(inplace=True),
                nn.Conv1d(32, 1, 1),
                nn.Sigmoid()
            )
    
    def forward(self, damaged, nominal, labels=None):
        """
        Args:
            damaged: (B, N, 3) damaged point cloud
            nominal: (B, M, 3) nominal point cloud
            labels: (B, N) optional damage labels
        
        Returns:
            damage_mask: (B, N) binary mask (1=damaged, 0=intact)
        """
        if labels is not None and self.mode == 'labels':
            # Use provided labels
            return labels
        
        elif self.mode == 'distance':
            # Distance-based detection (memory-efficient version)
            B, N, _ = damaged.shape
            M = nominal.shape[1]
            
            # Process in smaller chunks to avoid OOM
            chunk_size = 256  # Process 256 points at a time
            min_distances_list = []
            
            for i in range(0, N, chunk_size):
                end_i = min(i + chunk_size, N)
                damaged_chunk = damaged[:, i:end_i, :]  # (B, chunk, 3)
                
                # Compute distances for this chunk
                damaged_exp = damaged_chunk.unsqueeze(2)  # (B, chunk, 1, 3)
                nominal_exp = nominal.unsqueeze(1)  # (B, 1, M, 3)
                
                # Compute distances: (B, chunk, M)
                distances_chunk = torch.norm(damaged_exp - nominal_exp, dim=3)
                
                # Minimum distance to nominal for each damaged point in chunk
                min_dist_chunk, _ = torch.min(distances_chunk, dim=2)  # (B, chunk)
                min_distances_list.append(min_dist_chunk)
            
            # Concatenate all chunks
            min_distances = torch.cat(min_distances_list, dim=1)  # (B, N)
            
            # Points far from nominal are damaged
            damage_mask = (min_distances > self.threshold).float()
            
            return damage_mask
        
        elif self.mode == 'learned':
            # Learned damage detection (future work)
            # For now, use distance-based as fallback
            return self.forward(damaged, nominal, labels)
        
        else:
            raise ValueError(f"Unknown damage detection mode: {self.mode}")


class PointNetEncoder(nn.Module):
    """
    PointNet-style encoder for feature extraction.
    Used for both damaged and nominal point clouds.
    """
    
    def __init__(self, input_channels=3, output_dim=512):
        super().__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv1d(input_channels, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv1d(128, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True)
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv1d(256, output_dim, 1),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        """
        Args:
            x: (B, N, 3) point cloud
        
        Returns:
            global_feat: (B, output_dim) global feature
            point_feat: (B, output_dim, N) per-point features
        """
        B, N, _ = x.shape
        x = x.transpose(2, 1)  # (B, 3, N)
        
        x = self.conv1(x)      # (B, 64, N)
        x = self.conv2(x)      # (B, 128, N)
        x = self.conv3(x)      # (B, 256, N)
        point_feat = self.conv4(x)  # (B, output_dim, N)
        
        # Global max pooling
        global_feat = torch.max(point_feat, dim=2)[0]  # (B, output_dim)
        
        return global_feat, point_feat


class HybridPCN(nn.Module):
    """
    Hybrid Damage-Aware Point Cloud Completion Network.
    
    Architecture:
    1. Dual encoder: Extract features from damaged and nominal
    2. Feature fusion: Combine damaged and nominal features
    3. Damage detection: Identify damaged regions
    4. Local decoder: Reconstruct only damaged patches
    5. Blending: Merge reconstructed with preserved regions
    """
    
    def __init__(self, num_dense=8192, latent_dim=1024, grid_size=4, 
                 damage_detection='distance', damage_threshold=0.02):
        super().__init__()
        
        self.num_dense = num_dense
        self.latent_dim = latent_dim
        self.grid_size = grid_size
        self.num_coarse = num_dense // (grid_size ** 2)
        
        # Dual encoders
        self.damaged_encoder = PointNetEncoder(input_channels=3, output_dim=512)
        self.nominal_encoder = PointNetEncoder(input_channels=3, output_dim=512)
        
        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Linear(1024, latent_dim),  # 512 + 512 = 1024
            nn.ReLU(inplace=True),
            nn.Linear(latent_dim, latent_dim)
        )
        
        # Damage detection
        self.damage_detector = DamageDetector(mode=damage_detection, threshold=damage_threshold)
        
        # Coarse decoder (similar to PCN)
        self.coarse_decoder = nn.Sequential(
            nn.Linear(latent_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 3 * self.num_coarse)
        )
        
        # Fine decoder (folding-based, similar to PCN)
        self.fine_decoder = nn.Sequential(
            nn.Conv1d(latent_dim + 3 + 2, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 3, 1)
        )
        
        # Folding seed (2D grid for fine reconstruction)
        a = torch.linspace(-0.05, 0.05, steps=grid_size, dtype=torch.float).view(1, grid_size).expand(grid_size, grid_size).reshape(1, -1)
        b = torch.linspace(-0.05, 0.05, steps=grid_size, dtype=torch.float).view(grid_size, 1).expand(grid_size, grid_size).reshape(1, -1)
        seed = torch.cat([a, b], dim=0).view(1, 2, grid_size ** 2)
        self.register_buffer('folding_seed', seed)
        
        # Blending weight predictor (learns how to blend reconstructed with nominal)
        self.blend_predictor = nn.Sequential(
            nn.Conv1d(latent_dim + 3, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 1, 1),
            nn.Sigmoid()
        )
    
    def forward(self, damaged, nominal, damage_labels=None):
        """
        Args:
            damaged: (B, N, 3) damaged point cloud
            nominal: (B, M, 3) nominal/reference point cloud
            damage_labels: (B, N) optional damage labels
        
        Returns:
            coarse: (B, num_coarse, 3) coarse reconstruction
            fine: (B, num_dense, 3) fine reconstruction
            damage_mask: (B, M) predicted damage mask for output
        """
        B = damaged.shape[0]
        
        # 1. Encode damaged and nominal
        damaged_global, damaged_point = self.damaged_encoder(damaged)  # (B, 512), (B, 512, N)
        nominal_global, nominal_point = self.nominal_encoder(nominal)  # (B, 512), (B, 512, M)
        
        # 2. Fuse features
        fused_global = torch.cat([damaged_global, nominal_global], dim=1)  # (B, 1024)
        fused_global = self.fusion(fused_global)  # (B, latent_dim)
        
        # 3. Detect damaged regions (on nominal space)
        # Resample damaged to match nominal size for damage detection
        if damaged.shape[1] != nominal.shape[1]:
            # Simple upsampling: repeat damaged points
            ratio = nominal.shape[1] // damaged.shape[1]
            damaged_upsampled = damaged.repeat_interleave(ratio, dim=1)[:, :nominal.shape[1], :]
        else:
            damaged_upsampled = damaged
        
        damage_mask = self.damage_detector(damaged_upsampled, nominal, damage_labels)  # (B, M)
        
        # 4. Coarse reconstruction
        coarse = self.coarse_decoder(fused_global)  # (B, 3 * num_coarse)
        coarse = coarse.reshape(B, self.num_coarse, 3)  # (B, num_coarse, 3)
        
        # 5. Fine reconstruction (folding-based)
        point_feat = coarse.unsqueeze(2).expand(-1, -1, self.grid_size ** 2, -1)  # (B, num_coarse, S, 3)
        point_feat = point_feat.reshape(B, self.num_dense, 3).transpose(2, 1)  # (B, 3, num_dense)
        
        seed = self.folding_seed.unsqueeze(2).expand(B, -1, self.num_coarse, -1)  # (B, 2, num_coarse, S)
        seed = seed.reshape(B, -1, self.num_dense)  # (B, 2, num_dense)
        
        fused_global_exp = fused_global.unsqueeze(2).expand(-1, -1, self.num_dense)  # (B, latent_dim, num_dense)
        feat = torch.cat([fused_global_exp, seed, point_feat], dim=1)  # (B, latent_dim+2+3, num_dense)
        
        fine = self.fine_decoder(feat) + point_feat  # (B, 3, num_dense)
        fine = fine.transpose(1, 2)  # (B, num_dense, 3)
        
        # 6. Blending (learn to blend reconstructed with nominal based on damage)
        # Predict blending weights
        blend_feat = torch.cat([fused_global_exp, fine.transpose(1, 2)], dim=1)  # (B, latent_dim+3, num_dense)
        blend_weights = self.blend_predictor(blend_feat).squeeze(1)  # (B, num_dense)
        
        # Blend: output = blend_weight * reconstructed + (1 - blend_weight) * nominal
        # For damaged regions: blend_weight ≈ 1 (use reconstruction)
        # For intact regions: blend_weight ≈ 0 (use nominal)
        if nominal.shape[1] == self.num_dense:
            # Align damage mask with blend weights
            damage_mask_aligned = damage_mask
            blend_weights = blend_weights * damage_mask_aligned + 0.1 * (1 - damage_mask_aligned)
            output = blend_weights.unsqueeze(2) * fine + (1 - blend_weights.unsqueeze(2)) * nominal
        else:
            # If sizes don't match, just use reconstruction
            output = fine
        
        return coarse, output, damage_mask
    
    def inference(self, damaged, nominal, damage_labels=None, preserve_threshold=0.5):
        """
        Inference mode with explicit preservation of undamaged regions.
        
        Args:
            damaged: (B, N, 3) damaged point cloud
            nominal: (B, M, 3) nominal point cloud
            damage_labels: (B, N) optional damage labels
            preserve_threshold: threshold for damage mask (0-1)
        
        Returns:
            output: (B, M, 3) reconstructed point cloud
            damage_mask: (B, M) damage mask
        """
        with torch.no_grad():
            coarse, fine, damage_mask = self.forward(damaged, nominal, damage_labels)
            
            # Hard blending based on threshold
            if nominal.shape[1] == fine.shape[1]:
                mask_binary = (damage_mask > preserve_threshold).unsqueeze(2).float()
                output = mask_binary * fine + (1 - mask_binary) * nominal
            else:
                output = fine
            
            return output, damage_mask
