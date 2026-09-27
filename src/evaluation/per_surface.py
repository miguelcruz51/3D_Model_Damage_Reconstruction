"""Per-surface evaluation.

Breaks down reconstruction metrics by individual test surface to assess
generalization across surface geometries. Ported from the original
``evaluate_per_surface.py`` with clean-repo imports and missing-checkpoint skip
guards; numerical behavior is preserved.
"""

import csv
import os
from typing import Dict, List

import numpy as np
import torch
from scipy.spatial import cKDTree
from tqdm import tqdm

from src.data import DMCLocalReconstruction
from src.models import (
    SeedDisplacementNet,
    BlindDisplacementNet,
    FlowDisplacementNet,
    LocalReconstructionPCN,
    PurePCN,
)

METHOD_NAMES = {
    'm1': 'PCN',
    'm3': 'LR-PCN',
    'm4': 'PointCleanNet',
    'm5': 'FlowNet3D',
    'm7': 'SeedDispNet (CD L1)',
}


def compute_surface_metrics(pred_pts, d_pts, d_lbl, g_pts, g_lbl, scale_mm) -> Dict:
    """Compute CD/L2/Hausdorff metrics for one sample (source-equivalent)."""
    pred_np = pred_pts if isinstance(pred_pts, np.ndarray) else pred_pts.cpu().numpy()
    gt_np = g_pts.numpy() if torch.is_tensor(g_pts) else g_pts
    input_lbl = d_lbl.numpy() if torch.is_tensor(d_lbl) else d_lbl
    gt_lbl = g_lbl.numpy() if torch.is_tensor(g_lbl) else g_lbl

    tree_p = cKDTree(pred_np)
    tree_g = cKDTree(gt_np)
    d1, _ = tree_p.query(gt_np)
    d2, _ = tree_g.query(pred_np)
    cd_full = float(np.mean(d1) + np.mean(d2)) * scale_mm

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
        cd_damaged = float(np.mean(dd1) + np.mean(dd2)) * scale_mm
        l2_error = float(np.mean(dd2)) * scale_mm
        hausdorff = float(max(np.max(dd1), np.max(dd2))) * scale_mm
    else:
        cd_damaged = l2_error = hausdorff = 0.0

    return {
        'cd_full': cd_full,
        'cd_damaged': cd_damaged,
        'l2_error': l2_error,
        'hausdorff': hausdorff,
        'n_damaged': n_damaged,
    }


def predict(method_key, model, d_pts, d_lbl, device):
    """Run inference for one sample and return the predicted points as numpy."""
    d_pts_b = d_pts.unsqueeze(0).to(device)
    d_lbl_b = d_lbl.unsqueeze(0).to(device)
    if method_key == 'm1':
        return model(d_pts_b, training=False).squeeze(0).cpu().numpy()
    if method_key == 'm3':
        return model(d_pts_b, d_lbl_b, training=False).squeeze(0).cpu().numpy()
    if method_key in ('m4', 'm5'):
        pred, _ = model(d_pts_b, d_lbl_b)
        return pred.squeeze(0).cpu().numpy()
    if method_key == 'm7':
        pred, _, _ = model(d_pts_b, d_lbl_b)
        return pred.squeeze(0).cpu().numpy()
    raise ValueError(f"Unknown method: {method_key}")


@torch.no_grad()
def evaluate_method(method_key, model, dataset, device) -> List[Dict]:
    """Evaluate a model on all samples of a dataset."""
    results = []
    for i in tqdm(range(len(dataset)), desc=f'  {method_key}', leave=False):
        d_pts, d_lbl, g_pts, g_lbl, scale_mm = dataset[i]
        pred = predict(method_key, model, d_pts, d_lbl, device)
        results.append(compute_surface_metrics(pred, d_pts, d_lbl, g_pts, g_lbl, scale_mm.item()))
    return results


def load_models(checkpoints: Dict[str, str], device) -> Dict:
    """Load the available models, skipping any missing checkpoint.

    Args:
        checkpoints: Mapping of method key (``m1``..``m7``) to checkpoint path.
        device: Torch device.
    """
    models = {}

    def _load(path):
        return torch.load(path, map_location=device)

    if checkpoints.get('m1') and os.path.exists(checkpoints['m1']):
        ckpt = _load(checkpoints['m1'])
        a = ckpt['args']
        m = PurePCN(num_points=a.num_points, latent_dim=a.latent_dim).to(device)
        m.load_state_dict(ckpt['model_state_dict'])
        m.eval()
        models['m1'] = m

    if checkpoints.get('m3') and os.path.exists(checkpoints['m3']):
        ckpt = _load(checkpoints['m3'])
        a = ckpt['args']
        m = LocalReconstructionPCN(num_points=a.num_points, latent_dim=a.latent_dim,
                                   predict_ratio=a.predict_ratio).to(device)
        m.load_state_dict(ckpt['model_state_dict'])
        m.eval()
        models['m3'] = m

    if checkpoints.get('m4') and os.path.exists(checkpoints['m4']):
        ckpt = _load(checkpoints['m4'])
        a = ckpt['args']
        m = BlindDisplacementNet(feature_dim=a.feature_dim).to(device)
        m.load_state_dict(ckpt['model_state_dict'])
        m.eval()
        models['m4'] = m

    if checkpoints.get('m5') and os.path.exists(checkpoints['m5']):
        ckpt = _load(checkpoints['m5'])
        a = ckpt['args']
        m = FlowDisplacementNet(feature_dim=a.feature_dim).to(device)
        m.load_state_dict(ckpt['model_state_dict'])
        m.eval()
        models['m5'] = m

    if checkpoints.get('m7') and os.path.exists(checkpoints['m7']):
        ckpt = _load(checkpoints['m7'])
        a = ckpt['args']
        m = SeedDisplacementNet(latent_dim=a.latent_dim, feature_dim=a.feature_dim).to(device)
        m.load_state_dict(ckpt['model_state_dict'])
        m.eval()
        models['m7'] = m

    return models


def run_per_surface(data_root, surfaces, checkpoints, output_dir, device) -> str:
    """Run per-surface evaluation and write ``per_surface_metrics.csv``."""
    os.makedirs(output_dir, exist_ok=True)
    models = load_models(checkpoints, device)
    print(f"  Loaded: {list(models.keys())}")

    all_rows = []
    for surface in surfaces:
        print(f"\nSurface {surface}:")
        dataset = DMCLocalReconstruction(
            data_root=data_root, split='test', components=[surface],
            num_points=2048, augment=False, only_damaged=True,
        )
        if len(dataset) == 0:
            print("  No samples, skipping.")
            continue

        dataset_erase = DMCLocalReconstruction(
            data_root=data_root, split='test', components=[surface],
            num_points=2048, augment=False, only_damaged=True, erase_damaged=True,
        )

        for key, model in models.items():
            ds = dataset_erase if key == 'm1' else dataset
            results = evaluate_method(key, model, ds, device)

            cd_full = [r['cd_full'] for r in results]
            cd_dmg = [r['cd_damaged'] for r in results if r['cd_damaged'] > 0]
            l2 = [r['l2_error'] for r in results if r['l2_error'] > 0]
            haus = [r['hausdorff'] for r in results if r['hausdorff'] > 0]

            all_rows.append({
                'surface': surface,
                'method': METHOD_NAMES[key],
                'n_samples': len(results),
                'cd_full_mean': round(np.mean(cd_full), 4),
                'cd_full_std': round(np.std(cd_full), 4),
                'cd_damaged_mean': round(np.mean(cd_dmg), 4) if cd_dmg else 'N/A',
                'cd_damaged_std': round(np.std(cd_dmg), 4) if cd_dmg else 'N/A',
                'l2_mean': round(np.mean(l2), 4) if l2 else 'N/A',
                'l2_std': round(np.std(l2), 4) if l2 else 'N/A',
                'hausdorff_mean': round(np.mean(haus), 4) if haus else 'N/A',
                'hausdorff_std': round(np.std(haus), 4) if haus else 'N/A',
            })

    csv_path = os.path.join(output_dir, 'per_surface_metrics.csv')
    if all_rows:
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=all_rows[0].keys())
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"\nResults saved to: {csv_path}")
    return csv_path
