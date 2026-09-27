"""
Training script for Label-Guided Point Completion Network

Trains with automatic preservation of intact points.
"""

import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from models.label_guided_pcn import LabelGuidedPCN
from dataset import DMCVolumetric
from metrics.metric import l1_cd


def train_epoch(model, train_loader, optimizer, device, epoch):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_cd_coarse = 0
    total_cd_fine = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    for batch_idx, batch_data in enumerate(pbar):
        # Unpack batch
        if len(batch_data) == 4:
            damaged, nominal, damaged_labels, nominal_labels = batch_data
        else:
            damaged, nominal = batch_data
            damaged_labels = torch.zeros(damaged.shape[0], damaged.shape[1])
            nominal_labels = torch.zeros(nominal.shape[0], nominal.shape[1])
        
        damaged = damaged.to(device)
        nominal = nominal.to(device)
        damaged_labels = damaged_labels.to(device)
        nominal_labels = nominal_labels.to(device)
        
        # Forward pass
        fine, coarse = model(damaged, damaged_labels, nominal_labels, training=True)
        
        # Compute losses
        cd_coarse = l1_cd(coarse, nominal)
        cd_fine = l1_cd(fine, nominal)
        
        # Total loss: weighted combination
        loss = cd_coarse + cd_fine
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Track metrics
        total_loss += loss.item()
        total_cd_coarse += cd_coarse.item()
        total_cd_fine += cd_fine.item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'cd_fine': f'{cd_fine.item():.4f}'
        })
    
    # Return average metrics
    n = len(train_loader)
    return {
        'loss': total_loss / n,
        'cd_coarse': total_cd_coarse / n,
        'cd_fine': total_cd_fine / n
    }


def validate(model, val_loader, device):
    """Validate the model."""
    model.eval()
    total_cd = 0
    
    with torch.no_grad():
        for batch_data in tqdm(val_loader, desc='Validating'):
            # Unpack batch
            if len(batch_data) == 4:
                damaged, nominal, damaged_labels, nominal_labels = batch_data
            else:
                damaged, nominal = batch_data
                damaged_labels = torch.zeros(damaged.shape[0], damaged.shape[1])
                nominal_labels = torch.zeros(nominal.shape[0], nominal.shape[1])
            
            damaged = damaged.to(device)
            nominal = nominal.to(device)
            damaged_labels = damaged_labels.to(device)
            
            # Forward pass
            output, _ = model(damaged, damaged_labels, training=False)
            
            # Compute CD
            cd = l1_cd(output, nominal)
            total_cd += cd.item()
    
    return total_cd / len(val_loader)


def main(args):
    # Set device
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Create datasets
    print("Loading datasets...")
    train_dataset = DMCVolumetric(
        split='train',
        num_input=args.num_input,
        num_output=args.num_output,
        extraction_method='bbox',
        use_high_density=args.use_high_density,
        high_density_points=args.high_density_points,
        augment=True,
        damage_aware=True
    )
    
    val_dataset = DMCVolumetric(
        split='valid',
        num_input=args.num_input,
        num_output=args.num_output,
        extraction_method='bbox',
        use_high_density=args.use_high_density,
        high_density_points=args.high_density_points,
        augment=False,
        damage_aware=True
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    
    # Create model
    print("Creating model...")
    model = LabelGuidedPCN(
        num_dense=args.num_output,
        latent_dim=args.latent_dim,
        grid_size=args.grid_size
    ).to(device)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")
    
    # Create optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=args.lr_decay_step,
        gamma=args.lr_decay_rate
    )
    
    # Training loop
    print("Starting training...")
    best_cd = float('inf')
    
    for epoch in range(1, args.epochs + 1):
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, device, epoch)
        
        # Validate
        val_cd = validate(model, val_loader, device)
        
        # Update learning rate
        scheduler.step()
        
        # Print metrics
        print(f"\nEpoch {epoch}/{args.epochs}")
        print(f"  Train Loss: {train_metrics['loss']:.4f}")
        print(f"  Train CD Fine: {train_metrics['cd_fine']:.4f}")
        print(f"  Val CD: {val_cd:.4f}")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save checkpoint
        if val_cd < best_cd:
            best_cd = val_cd
            checkpoint_path = os.path.join(args.log_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_cd': val_cd,
                'args': args
            }, checkpoint_path)
            print(f"  ✓ Saved best model (CD: {val_cd:.4f})")
        
        # Save periodic checkpoint
        if epoch % args.save_freq == 0:
            checkpoint_path = os.path.join(args.log_dir, f'model_epoch_{epoch}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_cd': val_cd,
                'args': args
            }, checkpoint_path)
    
    print(f"\nTraining complete! Best Val CD: {best_cd:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Label-Guided PCN')
    
    # Data parameters
    parser.add_argument('--num_input', type=int, default=2048,
                       help='Number of input points')
    parser.add_argument('--num_output', type=int, default=2048,
                       help='Number of output points')
    parser.add_argument('--use_high_density', action='store_true',
                       help='Use high-density nominal extraction')
    parser.add_argument('--high_density_points', type=int, default=100000,
                       help='Number of points for high-density extraction')
    
    # Model parameters
    parser.add_argument('--latent_dim', type=int, default=1024,
                       help='Latent dimension')
    parser.add_argument('--grid_size', type=int, default=4,
                       help='Grid size for coarse points')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.0001,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0001,
                       help='Weight decay')
    parser.add_argument('--lr_decay_step', type=int, default=50,
                       help='LR decay step')
    parser.add_argument('--lr_decay_rate', type=float, default=0.5,
                       help='LR decay rate')
    
    # System parameters
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU to use')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--log_dir', type=str, required=True,
                       help='Log directory')
    parser.add_argument('--save_freq', type=int, default=10,
                       help='Save frequency (epochs)')
    
    args = parser.parse_args()
    
    main(args)
