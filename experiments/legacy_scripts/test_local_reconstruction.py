"""
Test script for Local Reconstruction pipeline

Verifies:
1. Dataset can load partition pairs
2. Model can process samples
3. Loss computation works correctly
"""

import torch
from dataset.DMCLocalReconstruction import DMCLocalReconstruction
from models.local_reconstruction_pcn import LocalReconstructionPCN, LabelAwareLoss


def test_dataset():
    """Test dataset loading."""
    print("="*80)
    print("Testing Dataset")
    print("="*80)
    
    dataset = DMCLocalReconstruction(
        data_root='data/DMC',
        split='train',
        num_points=2048,
        augment=False,
        only_damaged=True
    )
    
    print(f"\nDataset size: {len(dataset)}")
    
    if len(dataset) == 0:
        print("\n❌ ERROR: No samples found!")
        print("Make sure you have:")
        print("  1. Partitions in data/DMC/*/partitions/")
        print("  2. Both damaged and GT files (e.g., 001001000201.npz and 001001000201_gt.npz)")
        print("  3. Partitions with damage flag '01'")
        return False
    
    # Test first sample
    damaged_pts, damaged_lbl, gt_pts, gt_lbl, scale_mm = dataset[0]
    
    print(f"\nSample 0:")
    print(f"  Damaged points: {damaged_pts.shape}")
    print(f"  Damaged labels: {damaged_lbl.shape}")
    print(f"    - Intact (0): {torch.sum(damaged_lbl == 0).item()} ({torch.sum(damaged_lbl == 0).item()/len(damaged_lbl)*100:.1f}%)")
    print(f"    - Damaged (1): {torch.sum(damaged_lbl == 1).item()} ({torch.sum(damaged_lbl == 1).item()/len(damaged_lbl)*100:.1f}%)")
    print(f"  GT points: {gt_pts.shape}")
    print(f"  GT labels: {gt_lbl.shape}")
    print(f"    - Intact (0): {torch.sum(gt_lbl == 0).item()} ({torch.sum(gt_lbl == 0).item()/len(gt_lbl)*100:.1f}%)")
    print(f"    - Reconstruction (1): {torch.sum(gt_lbl == 1).item()} ({torch.sum(gt_lbl == 1).item()/len(gt_lbl)*100:.1f}%)")
    
    # Verify labels make sense
    damaged_count = torch.sum(damaged_lbl == 1).item()
    recon_count = torch.sum(gt_lbl == 1).item()
    
    if damaged_count == 0:
        print("\n⚠️  WARNING: No damaged points in sample (should have damage flag '01')")
    
    if recon_count == 0:
        print("\n⚠️  WARNING: No reconstruction targets in GT")
    
    print("\n✓ Dataset test passed!")
    return True


def test_model():
    """Test model forward pass."""
    print("\n" + "="*80)
    print("Testing Model")
    print("="*80)
    
    # Create model
    model = LocalReconstructionPCN(
        num_points=2048,
        latent_dim=512,
        predict_ratio=0.5
    )
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters: {num_params:,}")
    
    # Create dummy batch
    batch_size = 4
    num_points = 2048
    
    damaged_points = torch.randn(batch_size, num_points, 3)
    damaged_labels = torch.zeros(batch_size, num_points)
    # Set 30% as damaged
    for b in range(batch_size):
        num_damaged = int(num_points * 0.3)
        damaged_indices = torch.randperm(num_points)[:num_damaged]
        damaged_labels[b, damaged_indices] = 1
    
    print(f"\nInput:")
    print(f"  Damaged points: {damaged_points.shape}")
    print(f"  Damaged labels: {damaged_labels.shape}")
    print(f"  Average damaged ratio: {damaged_labels.mean().item():.2%}")
    
    # Training mode
    print("\nTraining mode:")
    model.train()
    intact_points_list, reconstructed_points = model(damaged_points, damaged_labels, training=True)
    
    print(f"  Intact points (list): {len(intact_points_list)} samples")
    for i, pts in enumerate(intact_points_list):
        print(f"    Sample {i}: {pts.shape[0]} points")
    print(f"  Reconstructed points: {reconstructed_points.shape}")
    
    # Inference mode
    print("\nInference mode:")
    model.eval()
    with torch.no_grad():
        output = model(damaged_points, damaged_labels, training=False)
    print(f"  Output: {output.shape}")
    
    print("\n✓ Model test passed!")
    return True


def test_loss():
    """Test loss computation."""
    print("\n" + "="*80)
    print("Testing Loss")
    print("="*80)
    
    # Create model and loss
    model = LocalReconstructionPCN(num_points=2048, latent_dim=512, predict_ratio=0.5)
    criterion = LabelAwareLoss(coverage_weight=0.5)
    
    # Create dummy batch
    batch_size = 4
    num_points = 2048
    
    damaged_points = torch.randn(batch_size, num_points, 3)
    damaged_labels = torch.zeros(batch_size, num_points)
    gt_points = torch.randn(batch_size, num_points, 3)
    gt_labels = torch.zeros(batch_size, num_points)
    
    # Set 30% as damaged/reconstruction
    for b in range(batch_size):
        num_damaged = int(num_points * 0.3)
        damaged_indices = torch.randperm(num_points)[:num_damaged]
        damaged_labels[b, damaged_indices] = 1
        gt_labels[b, damaged_indices] = 1
    
    print(f"\nBatch:")
    print(f"  Batch size: {batch_size}")
    print(f"  Points per sample: {num_points}")
    print(f"  Average damaged ratio: {damaged_labels.mean().item():.2%}")
    print(f"  Average reconstruction ratio: {gt_labels.mean().item():.2%}")
    
    # Forward pass
    model.train()
    intact_points_list, reconstructed_points = model(damaged_points, damaged_labels, training=True)
    
    # Compute loss
    loss, metrics = criterion(intact_points_list, reconstructed_points, gt_points, gt_labels)
    
    print(f"\nLoss metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    
    # Verify loss is differentiable
    loss.backward()
    print(f"\n✓ Loss is differentiable")
    
    print("\n✓ Loss test passed!")
    return True


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("Local Reconstruction Pipeline Test")
    print("="*80)
    
    success = True
    
    # Test dataset
    if not test_dataset():
        success = False
        print("\n❌ Dataset test failed - fix dataset issues before proceeding")
        return
    
    # Test model
    if not test_model():
        success = False
    
    # Test loss
    if not test_loss():
        success = False
    
    # Summary
    print("\n" + "="*80)
    if success:
        print("✓ All tests passed! Ready to train.")
        print("\nTo start training, run:")
        print("  python train_local_reconstruction.py \\")
        print("    --data_root data/DMC \\")
        print("    --num_points 2048 \\")
        print("    --batch_size 16 \\")
        print("    --epochs 100 \\")
        print("    --lr 0.001 \\")
        print("    --log_dir ./log/local_reconstruction \\")
        print("    --gpu 0")
    else:
        print("❌ Some tests failed - please fix issues before training")
    print("="*80)


if __name__ == '__main__':
    main()
