"""
Interactive Open3D visualization of saved inference results.

Shows three views per sample:
1. Input (grey=intact, red=damaged)
2. Ground Truth (grey=intact, green=reconstructed)
3. Prediction (grey=intact, blue=corrected)

Press 'q' to close window and move to next view/sample.
Press Ctrl+C in terminal to exit.
"""

import argparse
import os
import glob
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree


def visualize_sample(data_path, sample_idx, total):
    data = np.load(data_path)
    input_pts = data['input_points']
    input_lbl = data['input_labels']
    gt_pts = data['gt_points']
    gt_lbl = data['gt_labels']
    pred_pts = data['pred_points']

    intact_mask = input_lbl == 0
    damaged_mask = input_lbl == 1
    gt_recon_mask = gt_lbl == 1

    n_intact = intact_mask.sum()
    n_damaged = damaged_mask.sum()

    # Compute metrics
    if damaged_mask.sum() > 0 and gt_recon_mask.sum() > 0:
        pred_damaged = pred_pts[damaged_mask]
        gt_recon = gt_pts[gt_recon_mask]
        tree_p = cKDTree(pred_damaged)
        tree_g = cKDTree(gt_recon)
        d1, _ = tree_p.query(gt_recon)
        d2, _ = tree_g.query(pred_damaged)
        cd_damaged = np.mean(d1) + np.mean(d2)
        
        min_len = min(len(pred_damaged), len(gt_recon))
        l2_err = np.sqrt(np.sum((pred_damaged[:min_len] - gt_recon[:min_len])**2, axis=1)).mean()
    else:
        cd_damaged = l2_err = 0

    print(f"\n{'='*70}")
    print(f"Sample {sample_idx}/{total}: {os.path.basename(data_path)}")
    print(f"{'='*70}")
    print(f"  Intact: {n_intact}, Damaged: {n_damaged}")
    print(f"  Damaged CD: {cd_damaged:.6f}, L2 Error: {l2_err:.6f}")

    # --- View 1: Input ---
    pcd_input = o3d.geometry.PointCloud()
    pcd_input.points = o3d.utility.Vector3dVector(input_pts)
    colors = np.zeros((len(input_pts), 3))
    colors[intact_mask] = [0.7, 0.7, 0.7]
    colors[damaged_mask] = [1.0, 0.0, 0.0]
    pcd_input.colors = o3d.utility.Vector3dVector(colors)

    print("\n  [1/3] Input (grey=intact, red=damaged) — press 'q' to continue")
    o3d.visualization.draw_geometries(
        [pcd_input],
        window_name=f"Sample {sample_idx} - Input ({n_damaged} damaged)",
        width=1200, height=800
    )

    # --- View 2: Ground Truth ---
    pcd_gt = o3d.geometry.PointCloud()
    pcd_gt.points = o3d.utility.Vector3dVector(gt_pts)
    colors_gt = np.zeros((len(gt_pts), 3))
    colors_gt[gt_lbl == 0] = [0.7, 0.7, 0.7]
    colors_gt[gt_lbl == 1] = [0.0, 1.0, 0.0]
    pcd_gt.colors = o3d.utility.Vector3dVector(colors_gt)

    print(f"  [2/3] Ground Truth (grey=intact, green=reconstructed) — press 'q' to continue")
    o3d.visualization.draw_geometries(
        [pcd_gt],
        window_name=f"Sample {sample_idx} - Ground Truth",
        width=1200, height=800
    )

    # --- View 3: Prediction ---
    pcd_pred = o3d.geometry.PointCloud()
    pcd_pred.points = o3d.utility.Vector3dVector(pred_pts)
    colors_pred = np.zeros((len(pred_pts), 3))
    colors_pred[intact_mask] = [0.7, 0.7, 0.7]
    colors_pred[damaged_mask] = [0.0, 0.4, 1.0]
    pcd_pred.colors = o3d.utility.Vector3dVector(colors_pred)

    print(f"  [3/3] Prediction (grey=intact, blue=corrected) — press 'q' to continue")
    o3d.visualization.draw_geometries(
        [pcd_pred],
        window_name=f"Sample {sample_idx} - Prediction (CD: {cd_damaged:.4f})",
        width=1200, height=800
    )


def main(args):
    files = sorted(glob.glob(os.path.join(args.pcd_dir, 'sample_*.npz')))

    if len(files) == 0:
        print(f"No .npz files found in {args.pcd_dir}")
        return

    print(f"Found {len(files)} samples in {args.pcd_dir}")

    try:
        for idx, f in enumerate(files, 1):
            visualize_sample(f, idx, len(files))
    except KeyboardInterrupt:
        print("\n\nVisualization interrupted.")

    print("\nDone!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize saved inference PCDs')
    parser.add_argument('--pcd_dir', type=str, required=True,
                       help='Directory containing saved .npz inference results')
    args = parser.parse_args()
    main(args)
