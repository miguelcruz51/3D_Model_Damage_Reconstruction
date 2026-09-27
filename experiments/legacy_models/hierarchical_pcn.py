"""
Hierarchical Patch-Based Point Cloud Reconstructor

Key Features:
1. Global encoder: Understands overall geometry from all patches
2. Local reconstructor: Reconstructs damaged points in each patch
3. Mask-guided: Only reconstructs points with label=1, keeps label=0 untouched
4. No CAD at inference: Uses learned priors from training

Training:
    Input: damaged_patch + nominal_patch + mask
    Output: reconstructed_patch (label=0 preserved, label=1 reconstructed)

Inference:
    Input: damaged_patch + mask (no nominal needed!)
    Output: reconstructed_patch (label=0 preserved, label=1 reconstructed)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PointNetEncoder(nn.Module):
    """PointNet-style encoder for feature extraction."""
    
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
            x: (B, N, C) point cloud with C channels
        
        Returns:
            global_feat: (B, output_dim) global feature
        """
        B, N, C = x.shape
        x = x.transpose(2, 1)  # (B, C, N)
        
        x = self.conv1(x)      # (B, 64, N)
        x = self.conv2(x)      # (B, 128, N)
        x = self.conv3(x)      # (B, 256, N)
        x = self.conv4(x)      # (B, output_dim, N)
        
        # Global max pooling
        global_feat = torch.max(x, dim=2)[0]  # (B, output_dim)
        
        return global_feat


class HierarchicalPatchReconstructor(nn.Module):
    """
    Hierarchical architecture for patch-based reconstruction.
    
    Stage 1: Global encoder - understands overall geometry
    Stage 2: Local reconstructor - reconstructs damaged points only
    Stage 3: Mask-guided blending - keeps intact points, replaces damaged points
    """
    
    def __init__(self, patch_size=2048, global_feat_dim=512, patch_feat_dim=512):
        super().__init__()
        
        self.patch_size = patch_size
        
        # Stage 1: Global context encoder (uses xyz only)
        self.global_encoder = PointNetEncoder(
            input_channels=3,
            output_dim=global_feat_dim
        )
        
        # Stage 2: Patch encoder (uses xyz + mask)
        self.patch_encoder = PointNetEncoder(
            input_channels=4,  # xyz + mask
            output_dim=patch_feat_dim
        )
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(global_feat_dim + patch_feat_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1024)
        )
        
        # Coarse decoder
        num_coarse = patch_size // 16  # 2048 / 16 = 128
        self.coarse_decoder = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 3 * num_coarse)
        )
        
        # Fine decoder (folding-based)
        self.fine_decoder = nn.Sequential(
            nn.Conv1d(1024 + 3 + 2, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 3, 1)
        )
        
        # Folding grid (4×4 = 16 points per coarse point)
        grid_size = 4
        # Larger grid for better spreading (0.2×0.2 instead of 0.1×0.1)
        a = torch.linspace(-0.2, 0.2, steps=grid_size, dtype=torch.float).view(1, grid_size).expand(grid_size, grid_size).reshape(1, -1)
        b = torch.linspace(-0.2, 0.2, steps=grid_size, dtype=torch.float).view(grid_size, 1).expand(grid_size, grid_size).reshape(1, -1)
        seed = torch.cat([a, b], dim=0).view(1, 2, grid_size ** 2)
        self.register_buffer('folding_seed', seed)
        
        self.num_coarse = num_coarse
        self.grid_size = grid_size
    
    def reconstruct_patch(self, patch, mask, global_context):
        """
        Reconstruct damaged points in a patch.
        
        Args:
            patch: (B, patch_size, 3) damaged patch
            mask: (B, patch_size) damage mask (0=intact, 1=damaged)
            global_context: (B, global_feat_dim) global context
        
        Returns:
            reconstructed: (B, patch_size, 3) reconstructed patch
        """
        B = patch.shape[0]
        
        # Encode patch with mask
        patch_with_mask = torch.cat([patch, mask.unsqueeze(-1)], dim=-1)  # (B, patch_size, 4)
        patch_feat = self.patch_encoder(patch_with_mask)  # (B, patch_feat_dim)
        
        # Fuse with global context
        fused_feat = self.fusion(torch.cat([patch_feat, global_context], dim=1))  # (B, 1024)
        
        # Coarse reconstruction
        coarse = self.coarse_decoder(fused_feat)  # (B, 3 * num_coarse)
        coarse = coarse.reshape(B, self.num_coarse, 3)  # (B, num_coarse, 3)
        
        # Fine reconstruction (folding)
        point_feat = coarse.unsqueeze(2).expand(-1, -1, self.grid_size ** 2, -1)  # (B, num_coarse, 16, 3)
        point_feat = point_feat.reshape(B, self.patch_size, 3).transpose(2, 1)  # (B, 3, patch_size)
        
        seed = self.folding_seed.unsqueeze(2).expand(B, -1, self.num_coarse, -1)  # (B, 2, num_coarse, 16)
        seed = seed.reshape(B, -1, self.patch_size)  # (B, 2, patch_size)
        
        fused_global_exp = fused_feat.unsqueeze(2).expand(-1, -1, self.patch_size)  # (B, 1024, patch_size)
        feat = torch.cat([fused_global_exp, seed, point_feat], dim=1)  # (B, 1024+2+3, patch_size)
        
        fine = self.fine_decoder(feat) + point_feat  # (B, 3, patch_size)
        fine = fine.transpose(1, 2)  # (B, patch_size, 3)
        
        # Mask-guided blending: keep intact (mask=0), replace damaged (mask=1)
        mask_3d = mask.unsqueeze(-1).expand_as(fine)  # (B, patch_size, 3)
        output = mask_3d * fine + (1 - mask_3d) * patch
        
        return output
    
    def forward(self, damaged_patch, mask, nominal_patch=None, return_intermediate=False):
        """
        Forward pass.
        
        Training mode (nominal_patch provided):
            Uses nominal as global context to learn reconstruction
        
        Inference mode (nominal_patch=None):
            Uses damaged patch itself as global context
        
        Args:
            damaged_patch: (B, patch_size, 3)
            mask: (B, patch_size) - 0=intact, 1=damaged
            nominal_patch: (B, patch_size, 3) optional, only for training
            return_intermediate: If True, return (coarse, fine, reconstructed)
        
        Returns:
            If return_intermediate=False:
                reconstructed: (B, patch_size, 3)
            If return_intermediate=True:
                coarse: (B, num_coarse, 3)
                fine: (B, patch_size, 3) - before mask blending
                reconstructed: (B, patch_size, 3) - after mask blending
        """
        # Global context
        if self.training and nominal_patch is not None:
            # Training: use nominal as global context
            global_context = self.global_encoder(nominal_patch)
        else:
            # Inference: use damaged patch as global context
            global_context = self.global_encoder(damaged_patch)
        
        # Reconstruct with intermediate outputs
        if return_intermediate:
            B = damaged_patch.shape[0]
            
            # Encode patch with mask
            patch_with_mask = torch.cat([damaged_patch, mask.unsqueeze(-1)], dim=-1)
            patch_feat = self.patch_encoder(patch_with_mask)
            
            # Fuse with global context
            fused_feat = self.fusion(torch.cat([patch_feat, global_context], dim=1))
            
            # Coarse reconstruction
            coarse = self.coarse_decoder(fused_feat)
            coarse = coarse.reshape(B, self.num_coarse, 3)
            
            # Fine reconstruction (folding)
            point_feat = coarse.unsqueeze(2).expand(-1, -1, self.grid_size ** 2, -1)
            point_feat = point_feat.reshape(B, self.patch_size, 3).transpose(2, 1)
            
            seed = self.folding_seed.unsqueeze(2).expand(B, -1, self.num_coarse, -1)
            seed = seed.reshape(B, -1, self.patch_size)
            
            fused_global_exp = fused_feat.unsqueeze(2).expand(-1, -1, self.patch_size)
            feat = torch.cat([fused_global_exp, seed, point_feat], dim=1)
            
            fine = self.fine_decoder(feat) + point_feat
            fine = fine.transpose(1, 2)
            
            # Mask-guided blending
            mask_3d = mask.unsqueeze(-1).expand_as(fine)
            reconstructed = mask_3d * fine + (1 - mask_3d) * damaged_patch
            
            return coarse, fine, reconstructed
        else:
            # Standard forward (only final output)
            reconstructed = self.reconstruct_patch(damaged_patch, mask, global_context)
            return reconstructed


if __name__ == '__main__':
    # Test the model
    print("Testing HierarchicalPatchReconstructor...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Create model
    model = HierarchicalPatchReconstructor(patch_size=2048).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Test forward pass
    batch_size = 4
    damaged = torch.randn(batch_size, 2048, 3).to(device)
    mask = torch.randint(0, 2, (batch_size, 2048)).float().to(device)
    nominal = torch.randn(batch_size, 2048, 3).to(device)
    
    # Training mode
    model.train()
    output_train = model(damaged, mask, nominal)
    print(f"\nTraining mode:")
    print(f"  Input: {damaged.shape}")
    print(f"  Mask: {mask.shape}")
    print(f"  Nominal: {nominal.shape}")
    print(f"  Output: {output_train.shape}")
    
    # Inference mode
    model.eval()
    with torch.no_grad():
        output_infer = model(damaged, mask, nominal=None)
    print(f"\nInference mode:")
    print(f"  Input: {damaged.shape}")
    print(f"  Mask: {mask.shape}")
    print(f"  Output: {output_infer.shape}")
    
    # Verify mask-guided blending
    intact_mask = (mask == 0).unsqueeze(-1).expand_as(damaged)
    intact_preserved = torch.allclose(
        output_train[intact_mask],
        damaged[intact_mask],
        atol=1e-5
    )
    print(f"\nIntact points preserved: {intact_preserved}")
    
    print("\n✓ Model test passed!")
