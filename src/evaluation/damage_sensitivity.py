"""Damage-size sensitivity analysis.

Groups test samples into damage-percentage bins and evaluates each method per
bin, producing a CSV table and a bar plot. Ported from the original
``evaluate_damage_sensitivity.py`` with clean-repo imports and
missing-checkpoint skip guards; numerical behavior is preserved.
"""

import csv
import os
from typing import Dict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from src.data import DMCLocalReconstruction

from .per_surface import METHOD_NAMES, compute_surface_metrics, load_models, predict

BINS_ORDER = ['<5%', '5-10%', '10-20%', '20-50%', '>50%']


def get_bin_label(pct: float) -> str:
    """Assign a damage percentage to a bin label."""
    if pct < 5:
        return '<5%'
    if pct < 10:
        return '5-10%'
    if pct < 20:
        return '10-20%'
    if pct < 50:
        return '20-50%'
    return '>50%'


def run_damage_sensitivity(data_root, checkpoints, output_dir, device) -> str:
    """Run damage-sensitivity analysis; write CSV + PNG, return the CSV path."""
    os.makedirs(output_dir, exist_ok=True)
    models = load_models(checkpoints, device)
    print(f"  Loaded: {list(models.keys())}")

    dataset = DMCLocalReconstruction(
        data_root=data_root, split='test',
        num_points=2048, augment=False, only_damaged=True,
    )
    dataset_erase = DMCLocalReconstruction(
        data_root=data_root, split='test',
        num_points=2048, augment=False, only_damaged=True, erase_damaged=True,
    )

    all_results = {key: {b: [] for b in BINS_ORDER} for key in models.keys()}

    print(f"\nEvaluating {len(dataset)} samples...")
    with torch.no_grad():
        for i in tqdm(range(len(dataset))):
            d_pts, d_lbl, g_pts, g_lbl, scale_mm = dataset[i]
            sample_erase = dataset_erase[i]

            pct = float(d_lbl.sum()) / len(d_lbl) * 100
            bin_label = get_bin_label(pct)

            for key, model in models.items():
                if key == 'm1':
                    d_e, dl_e, _, _, _ = sample_erase
                    pred = predict(key, model, d_e, dl_e, device)
                else:
                    pred = predict(key, model, d_pts, d_lbl, device)
                metrics = compute_surface_metrics(pred, d_pts, d_lbl, g_pts, g_lbl, scale_mm.item())
                metrics['pct_damaged'] = pct
                all_results[key][bin_label].append(metrics)

    all_rows = []
    for key in models.keys():
        for bin_label in BINS_ORDER:
            results = all_results[key][bin_label]
            if not results:
                continue
            cd_dmg = [r['cd_damaged'] for r in results if r['cd_damaged'] > 0]
            l2 = [r['l2_error'] for r in results if r['l2_error'] > 0]
            haus = [r['hausdorff'] for r in results if r['hausdorff'] > 0]
            cd_full = [r['cd_full'] for r in results]

            all_rows.append({
                'method': METHOD_NAMES[key],
                'damage_range': bin_label,
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

    csv_path = os.path.join(output_dir, 'damage_sensitivity.csv')
    if all_rows:
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=all_rows[0].keys())
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"\nCSV saved to: {csv_path}")

    if models:
        _plot_damage_sensitivity(models, all_results, output_dir)

    return csv_path


def _plot_damage_sensitivity(models: Dict, all_results: Dict, output_dir: str) -> str:
    """Render the CD-damaged-per-bin grouped bar chart."""
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(BINS_ORDER))
    n_methods = len(models)
    width = 0.8 / n_methods
    colors = ['#FF6B6B', '#4ECDC4', '#FFA726', '#7E57C2', '#1E88E5']

    for idx, key in enumerate(models.keys()):
        means, stds = [], []
        for bin_label in BINS_ORDER:
            results = all_results[key][bin_label]
            cd_dmg = [r['cd_damaged'] for r in results if r['cd_damaged'] > 0]
            means.append(np.mean(cd_dmg) if cd_dmg else 0)
            stds.append(np.std(cd_dmg) if cd_dmg else 0)

        offset = (idx - n_methods / 2 + 0.5) * width
        ax.bar(x + offset, means, width, yerr=stds, capsize=2,
               label=METHOD_NAMES[key], color=colors[idx % len(colors)],
               edgecolor='black', linewidth=0.5, alpha=0.85)

    ax.set_xlabel('Damage Size (% of partition points)', fontsize=12)
    ax.set_ylabel('CD Damaged [mm]', fontsize=12)
    ax.set_title('Reconstruction Quality vs. Damage Extent', fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(BINS_ORDER)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_axisbelow(True)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'damage_sensitivity.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to: {plot_path}")
    return plot_path
