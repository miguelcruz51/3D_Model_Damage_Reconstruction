"""Geometric reconstruction metrics.

This module provides :func:`compute_metrics`, which evaluates a single
reconstructed partition against its ground-truth nominal partition. The five
metrics are preserved verbatim from the original ``evaluate_all.py`` so that the
reorganized evaluation reports identical numbers:

- **Full-cloud Chamfer distance** (``cd_full``): bidirectional mean
  nearest-neighbor distance between the full predicted and ground-truth clouds.
- **Damaged-region Chamfer distance** (``cd_damaged``): the same bidirectional
  mean nearest-neighbor distance restricted to the damaged input points and the
  ground-truth reconstruction targets.
- **Per-point L2 error** (``l2_error``): mean nearest-neighbor distance from each
  predicted damaged point to the ground-truth reconstruction target
  (one direction, ``pred -> gt``). Using nearest-neighbor rather than index
  truncation keeps this metric valid regardless of output point ordering.
- **Hausdorff distance** (``hausdorff``): maximum of both directional
  nearest-neighbor distances on the damaged region.
- **Intact-preservation error** (``intact_error``): mean displacement of intact
  points relative to the *input* (not the ground truth); a method that perfectly
  preserves intact points scores ``0``. Returns ``-1.0`` when not applicable
  (for example, methods that regenerate all points).

All distances are multiplied by ``scale_mm`` to convert from unit-sphere-normalized
space back to physical millimeters.

For every metric here, a **lower** value indicates a better reconstruction.
"""

from typing import Dict

import numpy as np
import torch
from scipy.spatial import cKDTree


def compute_metrics(pred_pts, pred_labels, gt_pts, gt_labels, input_pts, input_labels, scale_mm=1.0) -> Dict:
    """Compute all reconstruction metrics for one sample.

    Args:
        pred_pts: Predicted points, ``[N, 3]`` tensor or array.
        pred_labels: Predicted per-point labels (unused; kept for signature
            compatibility with the original implementation).
        gt_pts: Ground-truth points, ``[N, 3]``.
        gt_labels: Ground-truth per-point labels (``1`` marks reconstruction
            targets).
        input_pts: Damaged input points, ``[N, 3]``.
        input_labels: Damaged input labels (``1`` marks damaged points).
        scale_mm: Scale factor (max normalization distance in mm) applied to all
            distances.

    Returns:
        Dict with ``n_damaged``, ``cd_full``, ``cd_damaged``, ``l2_error``,
        ``hausdorff``, and ``intact_error`` (all distances in millimeters).
    """
    pred_np = pred_pts.cpu().numpy() if torch.is_tensor(pred_pts) else pred_pts
    gt_np = gt_pts.cpu().numpy() if torch.is_tensor(gt_pts) else gt_pts
    input_np = input_pts.cpu().numpy() if torch.is_tensor(input_pts) else input_pts
    input_lbl = input_labels.cpu().numpy() if torch.is_tensor(input_labels) else input_labels
    gt_lbl = gt_labels.cpu().numpy() if torch.is_tensor(gt_labels) else gt_labels

    # Full cloud CD
    tree_p = cKDTree(pred_np)
    tree_g = cKDTree(gt_np)
    d1, _ = tree_p.query(gt_np)
    d2, _ = tree_g.query(pred_np)
    cd_full = float(np.mean(d1) + np.mean(d2))

    # Damaged region CD
    damaged_mask = input_lbl == 1
    gt_recon_mask = gt_lbl == 1
    n_damaged = int(damaged_mask.sum())

    if n_damaged > 0 and gt_recon_mask.sum() > 0:
        pred_dmg = pred_np[damaged_mask] if len(pred_np) == len(input_lbl) else pred_np
        gt_recon = gt_np[gt_recon_mask]

        tree_pd = cKDTree(pred_dmg)
        tree_gd = cKDTree(gt_recon)
        dd1, _ = tree_pd.query(gt_recon)
        dd2, _ = tree_gd.query(pred_dmg)
        cd_damaged = float(np.mean(dd1) + np.mean(dd2))

        # Per-point L2 error on damaged region (nearest-neighbour, one direction:
        # pred -> gt). Using NN instead of index-based truncation makes this valid
        # for all methods, regardless of point ordering in the output.
        # dd2 already contains distances from each pred_dmg point to its nearest
        # gt_recon point.
        l2_error = float(np.mean(dd2))

        # Hausdorff distance on damaged region
        hausdorff = float(max(np.max(dd1), np.max(dd2)))
    else:
        cd_damaged = l2_error = hausdorff = 0.0

    # Intact preservation error.
    # Measures how much the model moved the intact points relative to the input.
    # Comparing against input_np (not gt_np) is correct here: a method that
    # perfectly preserves intact points should score 0, regardless of any noise
    # offset between the input and the clean ground truth.
    intact_mask = input_lbl == 0
    if intact_mask.sum() > 0 and len(pred_np) == len(input_np):
        intact_error = float(np.sqrt(np.sum((pred_np[intact_mask] - input_np[intact_mask]) ** 2, axis=1)).mean())
    else:
        intact_error = -1.0  # N/A for methods that regenerate all points

    return {
        'n_damaged': n_damaged,
        'cd_full': cd_full * scale_mm,
        'cd_damaged': cd_damaged * scale_mm,
        'l2_error': l2_error * scale_mm,
        'hausdorff': hausdorff * scale_mm,
        'intact_error': intact_error * scale_mm if intact_error >= 0 else intact_error,
    }
