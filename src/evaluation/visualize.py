"""Qualitative reconstruction visualization.

Ports the 3-panel (input / ground truth / prediction) qualitative visualizer
from the original ``inference_seed_displacement.py``. Given a damaged input, its
ground truth, and a model prediction, :func:`visualize_prediction` renders a
figure and returns the full-cloud Chamfer distance, damaged-region Chamfer
distance, and mean per-point L2 error for the sample.

Visualization and metric behavior are preserved from the source implementation.
"""

import os
from typing import Optional, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.spatial import cKDTree


def visualize_prediction(damaged_pts, damaged_lbl, gt_pts, gt_lbl,
                         pred_pts, sample_idx, save_dir: Optional[str] = None) -> Tuple[float, float, float]:
    """Render a 3-panel qualitative figure for one reconstructed sample.

    Panels: damaged input (intact gray / damaged red), ground truth
    (intact gray / reconstructed green), and prediction (intact gray /
    corrected blue).

    Args:
        damaged_pts: Damaged input points, ``[N, 3]``.
        damaged_lbl: Damaged input labels (``1`` marks damaged points).
        gt_pts: Ground-truth points, ``[N, 3]``.
        gt_lbl: Ground-truth labels (``1`` marks reconstruction targets).
        pred_pts: Predicted points, ``[N, 3]``.
        sample_idx: Sample index used for the title/filename.
        save_dir: If given, the figure is saved as ``sample_{idx:03d}.png``.

    Returns:
        ``(cd_full, cd_damaged, mean_l2_error)`` for the sample.
    """
    if torch.is_tensor(damaged_pts):
        damaged_pts = damaged_pts.cpu().numpy()
        damaged_lbl = damaged_lbl.cpu().numpy()
        gt_pts = gt_pts.cpu().numpy()
        gt_lbl = gt_lbl.cpu().numpy()
        pred_pts = pred_pts.cpu().numpy()

    # Per-point error on damaged region
    damaged_mask = damaged_lbl == 1
    gt_recon_mask = gt_lbl == 1

    if damaged_mask.sum() > 0 and gt_recon_mask.sum() > 0:
        pred_damaged = pred_pts[damaged_mask]
        gt_recon = gt_pts[gt_recon_mask]
        min_len = min(len(pred_damaged), len(gt_recon))
        point_error = np.sqrt(np.sum((pred_damaged[:min_len] - gt_recon[:min_len]) ** 2, axis=1))
        mean_error = point_error.mean()

        tree_p = cKDTree(pred_damaged)
        tree_g = cKDTree(gt_recon)
        d1, _ = tree_p.query(gt_recon)
        d2, _ = tree_g.query(pred_damaged)
        cd_damaged = np.mean(d1) + np.mean(d2)
    else:
        mean_error = cd_damaged = 0.0

    # Full cloud CD
    tree_pred = cKDTree(pred_pts)
    tree_gt = cKDTree(gt_pts)
    d1, _ = tree_pred.query(gt_pts)
    d2, _ = tree_gt.query(pred_pts)
    cd_full = np.mean(d1) + np.mean(d2)

    print(f"\nSample {sample_idx}:")
    print(f"  Damaged: {damaged_mask.sum()}, Full CD: {cd_full:.6f}, "
          f"Damaged CD: {cd_damaged:.6f}, L2 Err: {mean_error:.6f}")

    fig = plt.figure(figsize=(20, 6))

    # Damaged input
    ax1 = fig.add_subplot(131, projection='3d')
    intact = damaged_lbl == 0
    damaged = damaged_lbl == 1
    if intact.any():
        ax1.scatter(damaged_pts[intact, 0], damaged_pts[intact, 1],
                    damaged_pts[intact, 2], c='gray', s=1, alpha=0.6, label='Intact')
    if damaged.any():
        ax1.scatter(damaged_pts[damaged, 0], damaged_pts[damaged, 1],
                    damaged_pts[damaged, 2], c='red', s=2, alpha=0.8, label='Damaged')
    ax1.set_title(f'Input\n{damaged.sum()} damaged')
    ax1.legend(markerscale=5)

    # GT
    ax2 = fig.add_subplot(132, projection='3d')
    intact_gt = gt_lbl == 0
    recon_gt = gt_lbl == 1
    if intact_gt.any():
        ax2.scatter(gt_pts[intact_gt, 0], gt_pts[intact_gt, 1],
                    gt_pts[intact_gt, 2], c='gray', s=1, alpha=0.6, label='Intact')
    if recon_gt.any():
        ax2.scatter(gt_pts[recon_gt, 0], gt_pts[recon_gt, 1],
                    gt_pts[recon_gt, 2], c='green', s=2, alpha=0.8, label='Reconstructed')
    ax2.set_title(f'Ground Truth\n{recon_gt.sum()} reconstructed')
    ax2.legend(markerscale=5)

    # Prediction
    ax3 = fig.add_subplot(133, projection='3d')
    if intact.any():
        ax3.scatter(pred_pts[intact, 0], pred_pts[intact, 1],
                    pred_pts[intact, 2], c='gray', s=1, alpha=0.6, label='Intact')
    if damaged.any():
        ax3.scatter(pred_pts[damaged, 0], pred_pts[damaged, 1],
                    pred_pts[damaged, 2], c='blue', s=2, alpha=0.8, label='Corrected')
    ax3.set_title(f'Prediction\nCD: {cd_damaged:.4f}, L2: {mean_error:.4f}')
    ax3.legend(markerscale=5)

    all_pts = np.vstack([damaged_pts, gt_pts, pred_pts])
    for ax in (ax1, ax2, ax3):
        ax.set_xlim(all_pts[:, 0].min(), all_pts[:, 0].max())
        ax.set_ylim(all_pts[:, 1].min(), all_pts[:, 1].max())
        ax.set_zlim(all_pts[:, 2].min(), all_pts[:, 2].max())
        ax.view_init(elev=20, azim=45)

    plt.suptitle(f'Seed Displacement - Sample {sample_idx}', fontsize=14)
    plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f'sample_{sample_idx:03d}.png'),
                    dpi=150, bbox_inches='tight')
    plt.close()
    return float(cd_full), float(cd_damaged), float(mean_error)
