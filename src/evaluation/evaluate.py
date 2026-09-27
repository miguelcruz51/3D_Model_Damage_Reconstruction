"""Per-model evaluators and result writers.

Ports the per-model evaluation loops and CSV writers from the original
``evaluate_all.py``. Each evaluator loads a checkpoint, runs the matching model
over every sample in a dataset, and returns per-sample metric records together
with the parameter count and mean per-sample inference time.

The numerical behavior (model calls, metric computation, CSV columns) is
preserved from the source implementation.
"""

import csv
import os
import time
from typing import Dict, List, Tuple

import numpy as np
import torch

from src.models import (
    SeedDisplacementNet,
    SeedDisplacementNetNoLocal,
    BlindDisplacementNet,
    FlowDisplacementNet,
    LocalReconstructionPCN,
    PurePCN,
)

from .metrics import compute_metrics


def evaluate_seed_displacement(checkpoint_path, dataset, device) -> Tuple[List[Dict], int, float]:
    """Evaluate SeedDisplacementNet or its NoLocal ablation."""
    ckpt = torch.load(checkpoint_path, map_location=device)
    args = ckpt['args']

    if getattr(args, 'no_local', False):
        model = SeedDisplacementNetNoLocal(latent_dim=args.latent_dim, feature_dim=args.feature_dim).to(device)
    else:
        model = SeedDisplacementNet(latent_dim=args.latent_dim, feature_dim=args.feature_dim).to(device)

    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    results = []
    params = sum(p.numel() for p in model.parameters())
    total_time = 0.0

    with torch.no_grad():
        for i in range(len(dataset)):
            d_pts, d_lbl, g_pts, g_lbl, scale_mm = dataset[i]
            d_pts_b = d_pts.unsqueeze(0).to(device)
            d_lbl_b = d_lbl.unsqueeze(0).to(device)

            t0 = time.time()
            output, _, _ = model(d_pts_b, d_lbl_b)
            total_time += time.time() - t0

            pred = output.squeeze(0)
            metrics = compute_metrics(pred, d_lbl, g_pts, g_lbl, d_pts, d_lbl, scale_mm.item())
            metrics['sample_idx'] = i
            results.append(metrics)

    return results, params, total_time / len(dataset)


def evaluate_displacement(checkpoint_path, dataset, device) -> Tuple[List[Dict], int, float]:
    """Evaluate BlindDisplacementNet or FlowDisplacementNet."""
    ckpt = torch.load(checkpoint_path, map_location=device)
    args = ckpt['args']

    if args.model == 'blind':
        model = BlindDisplacementNet(feature_dim=args.feature_dim).to(device)
    elif args.model == 'flow':
        model = FlowDisplacementNet(feature_dim=args.feature_dim).to(device)
    else:
        raise ValueError(f"Unknown model: {args.model}")

    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    results = []
    params = sum(p.numel() for p in model.parameters())
    total_time = 0.0

    with torch.no_grad():
        for i in range(len(dataset)):
            d_pts, d_lbl, g_pts, g_lbl, scale_mm = dataset[i]
            d_pts_b = d_pts.unsqueeze(0).to(device)
            d_lbl_b = d_lbl.unsqueeze(0).to(device)

            t0 = time.time()
            output, _ = model(d_pts_b, d_lbl_b)
            total_time += time.time() - t0

            pred = output.squeeze(0)
            metrics = compute_metrics(pred, d_lbl, g_pts, g_lbl, d_pts, d_lbl, scale_mm.item())
            metrics['sample_idx'] = i
            results.append(metrics)

    return results, params, total_time / len(dataset)


def evaluate_lrpcn(checkpoint_path, dataset, device) -> Tuple[List[Dict], int, float]:
    """Evaluate LocalReconstructionPCN."""
    ckpt = torch.load(checkpoint_path, map_location=device)
    args = ckpt['args']
    model = LocalReconstructionPCN(
        num_points=args.num_points, latent_dim=args.latent_dim,
        predict_ratio=args.predict_ratio,
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    results = []
    params = sum(p.numel() for p in model.parameters())
    total_time = 0.0

    with torch.no_grad():
        for i in range(len(dataset)):
            d_pts, d_lbl, g_pts, g_lbl, scale_mm = dataset[i]
            d_pts_b = d_pts.unsqueeze(0).to(device)
            d_lbl_b = d_lbl.unsqueeze(0).to(device)

            t0 = time.time()
            output = model(d_pts_b, d_lbl_b, training=False)
            total_time += time.time() - t0

            pred = output.squeeze(0)
            metrics = compute_metrics(pred, d_lbl, g_pts, g_lbl, d_pts, d_lbl, scale_mm.item())

            # Override intact_error for LR-PCN: the model packs intact points at
            # indices 0..n_intact-1 in the output (same order as d_pts[intact_mask]),
            # so we compare those directly against the input intact points.
            d_pts_np = d_pts.numpy()
            pred_np = pred.cpu().numpy()
            d_lbl_np = d_lbl.numpy()
            intact_mask = d_lbl_np == 0
            n_intact = int(intact_mask.sum())
            if n_intact > 0 and len(pred_np) >= n_intact:
                intact_error = float(
                    np.sqrt(np.sum((pred_np[:n_intact] - d_pts_np[intact_mask]) ** 2, axis=1)).mean()
                )
                metrics['intact_error'] = intact_error * scale_mm.item()

            metrics['sample_idx'] = i
            results.append(metrics)

    return results, params, total_time / len(dataset)


def evaluate_pure_pcn(checkpoint_path, dataset_erase, device) -> Tuple[List[Dict], int, float]:
    """Evaluate PurePCN on the erase-damaged dataset variant."""
    ckpt = torch.load(checkpoint_path, map_location=device)
    args = ckpt['args']
    model = PurePCN(num_points=args.num_points, latent_dim=args.latent_dim).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    results = []
    params = sum(p.numel() for p in model.parameters())
    total_time = 0.0

    with torch.no_grad():
        for i in range(len(dataset_erase)):
            d_pts, d_lbl, g_pts, g_lbl, scale_mm = dataset_erase[i]
            d_pts_b = d_pts.unsqueeze(0).to(device)

            t0 = time.time()
            output = model(d_pts_b, training=False)
            total_time += time.time() - t0

            pred = output.squeeze(0)
            metrics = compute_metrics(pred, d_lbl, g_pts, g_lbl, d_pts, d_lbl, scale_mm.item())
            metrics['intact_error'] = -1.0  # N/A - regenerates all points
            metrics['sample_idx'] = i
            results.append(metrics)

    return results, params, total_time / len(dataset_erase)


def save_results(results, method_name, output_dir) -> str:
    """Save per-sample results to a CSV file (millimeter columns)."""
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, f'{method_name}_results.csv')

    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['sample_idx', 'n_damaged', 'cd_full_mm',
                                               'cd_damaged_mm', 'l2_error_mm', 'hausdorff_mm', 'intact_error_mm'])
        writer.writeheader()
        for r in results:
            writer.writerow({
                'sample_idx': r['sample_idx'],
                'n_damaged': r['n_damaged'],
                'cd_full_mm': r['cd_full'],
                'cd_damaged_mm': r['cd_damaged'],
                'l2_error_mm': r['l2_error'],
                'hausdorff_mm': r['hausdorff'],
                'intact_error_mm': r['intact_error'],
            })

    return csv_path


def summarize(results, method_name, params, inference_time) -> Dict:
    """Print and return a summary-statistics row for one method."""
    cd_full = [r['cd_full'] for r in results]
    cd_dmg = [r['cd_damaged'] for r in results if r['cd_damaged'] > 0]
    l2 = [r['l2_error'] for r in results if r['l2_error'] > 0]
    haus = [r['hausdorff'] for r in results if r['hausdorff'] > 0]
    intact = [r['intact_error'] for r in results if r['intact_error'] >= 0]

    print(f"\n{'=' * 70}")
    print(f"  {method_name}")
    print(f"{'=' * 70}")
    print(f"  Samples:          {len(results)}")
    print(f"  Parameters:       {params:,}")
    print(f"  Inference time:   {inference_time * 1000:.2f} ms/sample")
    print(f"  CD Full:          {np.mean(cd_full):.4f} +/- {np.std(cd_full):.4f} mm")
    if cd_dmg:
        print(f"  CD Damaged:       {np.mean(cd_dmg):.4f} +/- {np.std(cd_dmg):.4f} mm")
    if l2:
        print(f"  L2 Error:         {np.mean(l2):.4f} +/- {np.std(l2):.4f} mm")
    if haus:
        print(f"  Hausdorff:        {np.mean(haus):.4f} +/- {np.std(haus):.4f} mm")
    if intact:
        print(f"  Intact Error:     {np.mean(intact):.4f} +/- {np.std(intact):.4f} mm")
    else:
        print(f"  Intact Error:     N/A (regenerates all points)")

    return {
        'method': method_name,
        'params': params,
        'inference_ms': round(inference_time * 1000, 2),
        'cd_full_mean_mm': round(np.mean(cd_full), 4),
        'cd_full_std_mm': round(np.std(cd_full), 4),
        'cd_damaged_mean_mm': round(np.mean(cd_dmg), 4) if cd_dmg else 'N/A',
        'cd_damaged_std_mm': round(np.std(cd_dmg), 4) if cd_dmg else 'N/A',
        'l2_mean_mm': round(np.mean(l2), 4) if l2 else 'N/A',
        'l2_std_mm': round(np.std(l2), 4) if l2 else 'N/A',
        'hausdorff_mean_mm': round(np.mean(haus), 4) if haus else 'N/A',
        'hausdorff_std_mm': round(np.std(haus), 4) if haus else 'N/A',
        'intact_mean_mm': round(np.mean(intact), 4) if intact else 'N/A',
        'intact_std_mm': round(np.std(intact), 4) if intact else 'N/A',
    }
