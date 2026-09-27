"""
Inference for Seed Displacement Net — saves point clouds as .npz files.
Stores input, ground truth, and prediction for each sample.
"""

import argparse
import os
import torch
import numpy as np

from models.seed_displacement_net import SeedDisplacementNet
from dataset.DMCLocalReconstruction import DMCLocalReconstruction


def main(args):
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model_args = checkpoint['args']

    print(f"Seed Displacement - Epoch: {checkpoint['epoch']}, Val Loss: {checkpoint['val_loss']:.6f}")

    model = SeedDisplacementNet(
        latent_dim=model_args.latent_dim, feature_dim=model_args.feature_dim
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    dataset = DMCLocalReconstruction(
        data_root=args.data_root, split=args.split,
        num_points=model_args.num_points, augment=False, only_damaged=True
    )
    print(f"Dataset [{args.split}]: {len(dataset)} samples")

    os.makedirs(args.save_dir, exist_ok=True)

    with torch.no_grad():
        for i in range(min(args.num_samples, len(dataset))):
            damaged_pts, damaged_lbl, gt_pts, gt_lbl, _ = dataset[i]

            output, _, _ = model(
                damaged_pts.unsqueeze(0).to(device),
                damaged_lbl.unsqueeze(0).to(device)
            )
            pred_pts = output.squeeze(0).cpu().numpy()

            damaged_pts_np = damaged_pts.numpy()
            damaged_lbl_np = damaged_lbl.numpy()
            gt_pts_np = gt_pts.numpy()
            gt_lbl_np = gt_lbl.numpy()

            # Save all in one file
            save_path = os.path.join(args.save_dir, f'sample_{i:03d}.npz')
            np.savez(
                save_path,
                input_points=damaged_pts_np,
                input_labels=damaged_lbl_np,
                gt_points=gt_pts_np,
                gt_labels=gt_lbl_np,
                pred_points=pred_pts
            )
            
            n_damaged = int(damaged_lbl_np.sum())
            print(f"Saved sample {i}: {save_path} ({n_damaged} damaged points)")

    print(f"\nDone! {min(args.num_samples, len(dataset))} samples saved to {args.save_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--data_root', type=str, default='data/DMSurface')
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--num_samples', type=int, default=10)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--save_dir', type=str, required=True)
    args = parser.parse_args()
    main(args)
