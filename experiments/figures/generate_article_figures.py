"""
Generate all article figures and analysis tables.

1. Per-surface-type analysis (breakdown by test surface 018, 019, 020)
2. Sensitivity to damage size (group by % damaged points)
3. Side-by-side qualitative comparison of all 5 methods on same samples
"""

import argparse
import os
import csv
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

from dataset.DMCLocalReconstruction import DMCLocalReconstruction


def load_model_and_predict(method, checkpoint_path, sample, device):
    """Load model and get prediction for one sample."""
    d_pts, d_lbl, g_pts, g_lbl, scale_mm = sample
    d_pts_b = d_pts.unsqueeze(0).to(device)
    d_lbl_b = d_lbl.unsqueeze(0).to(device)

    if method == 'm1':
        from models.pure_pcn import PurePCN
        ckpt = torch.load(checkpoint_path, map_location=device)
        args = ckpt['args']
        model = PurePCN(num_points=args.num_points, latent_dim=args.latent_dim).to(device)
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()
        with torch.no_grad():
            pred = model(d_pts_b, training=False).squeeze(0)
        return pred.cpu().numpy()

    elif method == 'm3':
        from models.local_reconstruction_pcn import LocalReconstructionPCN
        ckpt = torch.load(checkpoint_path, map_location=device)
        args = ckpt['args']
        model = LocalReconstructionPCN(
            num_points=args.num_points, latent_dim=args.latent_dim,
            predict_ratio=args.predict_ratio
        ).to(device)
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()
        with torch.no_grad():
            pred = model(d_pts_b, d_lbl_b, training=False).squeeze(0)
        return pred.cpu().numpy()

    elif method in ('m4', 'm5'):
        from models.displacement_net import BlindDisplacementNet, FlowDisplacementNet
        ckpt = torch.load(checkpoint_path, map_location=device)
        args = ckpt['args']
        if args.model == 'blind':
            model = BlindDisplacementNet(feature_dim=args.feature_dim).to(device)
        else:
            model = FlowDisplacementNet(feature_dim=args.feature_dim).to(device)
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()
        with torch.no_grad():
            pred, _ = model(d_pts_b, d_lbl_b)
        return pred.squeeze(0).cpu().numpy()

    elif method == 'm7':
        from models.seed_displacement_net import SeedDisplacementNet
        ckpt = torch.load(checkpoint_path, map_location=device)
        args = ckpt['args']
        model = SeedDisplacementNet(
            latent_dim=args.latent_dim, feature_dim=args.feature_dim
        ).to(device)
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()
        with torch.no_grad():
            pred, _, _ = model(d_pts_b, d_lbl_b)
        return pred.squeeze(0).cpu().numpy()


def compute_sample_metrics(pred_pts, d_pts, d_lbl, g_pts, g_lbl, scale_mm=1.0):
    """Compute metrics for one sample. scale_mm converts from normalized to physical units."""
    pred_np = pred_pts if isinstance(pred_pts, np.ndarray) else pred_pts.numpy()
    d_np = d_pts.numpy() if torch.is_tensor(d_pts) else d_pts
    g_np = g_pts.numpy() if torch.is_tensor(g_pts) else g_pts
    dl = d_lbl.numpy() if torch.is_tensor(d_lbl) else d_lbl
    gl = g_lbl.numpy() if torch.is_tensor(g_lbl) else g_lbl

    damaged_mask = dl == 1
    gt_recon_mask = gl == 1
    n_damaged = int(damaged_mask.sum())
    pct_damaged = n_damaged / len(dl) * 100

    # Full CD
    t1 = cKDTree(pred_np)
    t2 = cKDTree(g_np)
    d1, _ = t1.query(g_np)
    d2, _ = t2.query(pred_np)
    cd_full = float(np.mean(d1) + np.mean(d2)) * scale_mm

    # Damaged CD
    if n_damaged > 0 and gt_recon_mask.sum() > 0:
        pd = pred_np[damaged_mask] if len(pred_np) == len(dl) else pred_np
        gr = g_np[gt_recon_mask]
        t1d = cKDTree(pd)
        t2d = cKDTree(gr)
        dd1, _ = t1d.query(gr)
        dd2, _ = t2d.query(pd)
        cd_damaged = float(np.mean(dd1) + np.mean(dd2)) * scale_mm
        hausdorff = float(max(np.max(dd1), np.max(dd2))) * scale_mm
    else:
        cd_damaged = hausdorff = 0.0

    return {
        'n_damaged': n_damaged,
        'pct_damaged': pct_damaged,
        'cd_full': cd_full,
        'cd_damaged': cd_damaged,
        'hausdorff': hausdorff,
    }


# ============================================================
# 1. Per-surface-type analysis
# ============================================================
def generate_per_surface_analysis(checkpoints, data_root, output_dir, device):
    """Break down results by test surface (018, 019, 020)."""
    print("\n=== Per-Surface Analysis ===")
    os.makedirs(output_dir, exist_ok=True)

    methods = {
        'M7: SeedDispNet': ('m7', checkpoints['m7']),
        'M3: LR-PCN': ('m3', checkpoints['m3']),
        'M4: BlindDisp': ('m4', checkpoints['m4']),
    }

    surfaces = ['018', '019', '020']
    all_results = {m: {s: [] for s in surfaces} for m in methods}

    for surface in surfaces:
        print(f"\nSurface {surface}...")
        dataset = DMCLocalReconstruction(
            data_root=data_root, split='test',
            components=[surface], num_points=2048,
            augment=False, only_damaged=True
        )

        for i in range(len(dataset)):
            sample = dataset[i]
            d_pts, d_lbl, g_pts, g_lbl, scale_mm = sample

            for method_name, (method_key, ckpt_path) in methods.items():
                if ckpt_path is None:
                    continue
                pred = load_model_and_predict(method_key, ckpt_path, sample, device)
                metrics = compute_sample_metrics(pred, d_pts, d_lbl, g_pts, g_lbl, scale_mm.item())
                all_results[method_name][surface].append(metrics)

    # Generate table
    csv_path = os.path.join(output_dir, 'per_surface_analysis.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Method', 'Surface', 'N_samples', 'CD_Full_mean_mm', 'CD_Full_std_mm',
                         'CD_Damaged_mean_mm', 'CD_Damaged_std_mm', 'Hausdorff_mean_mm', 'Hausdorff_std_mm'])
        for method_name in methods:
            for surface in surfaces:
                results = all_results[method_name][surface]
                if not results:
                    continue
                cd_f = [r['cd_full'] for r in results]
                cd_d = [r['cd_damaged'] for r in results if r['cd_damaged'] > 0]
                haus = [r['hausdorff'] for r in results if r['hausdorff'] > 0]
                writer.writerow([
                    method_name, surface, len(results),
                    f"{np.mean(cd_f):.6f}", f"{np.std(cd_f):.6f}",
                    f"{np.mean(cd_d):.6f}" if cd_d else 'N/A',
                    f"{np.std(cd_d):.6f}" if cd_d else 'N/A',
                    f"{np.mean(haus):.6f}" if haus else 'N/A',
                    f"{np.std(haus):.6f}" if haus else 'N/A',
                ])

    # Generate plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Per-Surface CD Damaged', fontsize=14)

    for idx, surface in enumerate(surfaces):
        ax = axes[idx]
        method_names = []
        means = []
        stds = []
        for method_name in methods:
            results = all_results[method_name][surface]
            cd_d = [r['cd_damaged'] for r in results if r['cd_damaged'] > 0]
            if cd_d:
                method_names.append(method_name.split(':')[0])
                means.append(np.mean(cd_d))
                stds.append(np.std(cd_d))

        bars = ax.bar(method_names, means, yerr=stds, capsize=5,
                      color=['#4CAF50', '#2196F3', '#FF9800'], edgecolor='black', linewidth=0.5)
        ax.set_title(f'Surface {surface}')
        ax.set_ylabel('CD Damaged [mm]')
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'per_surface_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {csv_path}")


# ============================================================
# 2. Sensitivity to damage size
# ============================================================
def generate_damage_sensitivity(checkpoints, data_root, output_dir, device):
    """Group results by damage percentage."""
    print("\n=== Damage Size Sensitivity ===")
    os.makedirs(output_dir, exist_ok=True)

    methods = {
        'M7: SeedDispNet': ('m7', checkpoints['m7']),
        'M3: LR-PCN': ('m3', checkpoints['m3']),
        'M4: BlindDisp': ('m4', checkpoints['m4']),
    }

    bins = [(0, 5, '<5%'), (5, 10, '5-10%'), (10, 20, '10-20%'), (20, 50, '20-50%'), (50, 100, '>50%')]

    dataset = DMCLocalReconstruction(
        data_root=data_root, split='test', num_points=2048,
        augment=False, only_damaged=True
    )

    all_results = {m: {b[2]: [] for b in bins} for m in methods}

    for i in range(len(dataset)):
        sample = dataset[i]
        d_pts, d_lbl, g_pts, g_lbl, scale_mm = sample
        pct = float(d_lbl.sum()) / len(d_lbl) * 100

        bin_label = None
        for lo, hi, label in bins:
            if lo <= pct < hi:
                bin_label = label
                break
        if bin_label is None:
            continue

        for method_name, (method_key, ckpt_path) in methods.items():
            if ckpt_path is None:
                continue
            pred = load_model_and_predict(method_key, ckpt_path, sample, device)
            metrics = compute_sample_metrics(pred, d_pts, d_lbl, g_pts, g_lbl, scale_mm.item())
            all_results[method_name][bin_label].append(metrics)

    # Generate table
    csv_path = os.path.join(output_dir, 'damage_sensitivity.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Method', 'Damage_Range', 'N_samples', 'CD_Damaged_mean_mm', 'CD_Damaged_std_mm'])
        for method_name in methods:
            for _, _, label in bins:
                results = all_results[method_name][label]
                if not results:
                    continue
                cd_d = [r['cd_damaged'] for r in results if r['cd_damaged'] > 0]
                if cd_d:
                    writer.writerow([method_name, label, len(results),
                                     f"{np.mean(cd_d):.6f}", f"{np.std(cd_d):.6f}"])

    # Generate plot
    fig, ax = plt.subplots(figsize=(12, 6))
    bin_labels = [b[2] for b in bins]
    x = np.arange(len(bin_labels))
    width = 0.25
    colors = ['#4CAF50', '#2196F3', '#FF9800']

    for idx, method_name in enumerate(methods):
        means = []
        stds = []
        for _, _, label in bins:
            results = all_results[method_name][label]
            cd_d = [r['cd_damaged'] for r in results if r['cd_damaged'] > 0]
            means.append(np.mean(cd_d) if cd_d else 0)
            stds.append(np.std(cd_d) if cd_d else 0)

        ax.bar(x + idx * width, means, width, yerr=stds, capsize=3,
               label=method_name.split(':')[0], color=colors[idx], edgecolor='black', linewidth=0.5)

    ax.set_xlabel('Damage Size (% of partition)')
    ax.set_ylabel('CD Damaged')
    ax.set_title('Reconstruction Quality vs Damage Size')
    ax.set_xticks(x + width)
    ax.set_xticklabels(bin_labels)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'damage_sensitivity.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {csv_path}")


# ============================================================
# 3. Side-by-side qualitative comparison
# ============================================================
def generate_qualitative_comparison(checkpoints, data_root, output_dir, device, num_samples=3):
    """Side-by-side visualization of all methods on same samples."""
    print("\n=== Qualitative Comparison ===")
    os.makedirs(output_dir, exist_ok=True)

    methods_order = [
        ('Input', None, None),
        ('Ground Truth', None, None),
        ('M1: PCN', 'm1', checkpoints.get('m1')),
        ('M3: LR-PCN', 'm3', checkpoints.get('m3')),
        ('M4: BlindDisp', 'm4', checkpoints.get('m4')),
        ('M7: Ours', 'm7', checkpoints.get('m7')),
    ]

    dataset = DMCLocalReconstruction(
        data_root=data_root, split='test', num_points=2048,
        augment=False, only_damaged=True
    )

    # Pick samples with good damage visibility
    np.random.seed(42)
    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)

    for sample_idx, data_idx in enumerate(indices):
        sample = dataset[data_idx]
        d_pts, d_lbl, g_pts, g_lbl, scale_mm = sample
        d_np = d_pts.numpy()
        g_np = g_pts.numpy()
        dl = d_lbl.numpy()
        gl = g_lbl.numpy()

        n_cols = len(methods_order)
        fig = plt.figure(figsize=(4 * n_cols, 4))

        for col, (title, method_key, ckpt_path) in enumerate(methods_order):
            ax = fig.add_subplot(1, n_cols, col + 1, projection='3d')

            if title == 'Input':
                intact = dl == 0
                damaged = dl == 1
                if intact.any():
                    ax.scatter(d_np[intact, 0], d_np[intact, 1], d_np[intact, 2],
                              c='gray', s=0.5, alpha=0.5)
                if damaged.any():
                    ax.scatter(d_np[damaged, 0], d_np[damaged, 1], d_np[damaged, 2],
                              c='red', s=1, alpha=0.8)
                ax.set_title(f'Input\n({int(damaged.sum())} damaged)', fontsize=9)

            elif title == 'Ground Truth':
                intact = gl == 0
                recon = gl == 1
                if intact.any():
                    ax.scatter(g_np[intact, 0], g_np[intact, 1], g_np[intact, 2],
                              c='gray', s=0.5, alpha=0.5)
                if recon.any():
                    ax.scatter(g_np[recon, 0], g_np[recon, 1], g_np[recon, 2],
                              c='green', s=1, alpha=0.8)
                ax.set_title('Ground Truth', fontsize=9)

            else:
                if ckpt_path is None:
                    ax.set_title(f'{title}\n(N/A)', fontsize=9)
                    continue

                pred = load_model_and_predict(method_key, ckpt_path, sample, device)
                metrics = compute_sample_metrics(pred, d_pts, d_lbl, g_pts, g_lbl, scale_mm.item())

                intact = dl == 0
                damaged = dl == 1

                if len(pred) == len(dl):
                    if intact.any():
                        ax.scatter(pred[intact, 0], pred[intact, 1], pred[intact, 2],
                                  c='gray', s=0.5, alpha=0.5)
                    if damaged.any():
                        ax.scatter(pred[damaged, 0], pred[damaged, 1], pred[damaged, 2],
                                  c='blue', s=1, alpha=0.8)
                else:
                    ax.scatter(pred[:, 0], pred[:, 1], pred[:, 2],
                              c='blue', s=0.5, alpha=0.6)

                cd_d = metrics['cd_damaged']
                ax.set_title(f'{title}\nCD: {cd_d:.4f}', fontsize=9)

            # Consistent view
            all_pts = np.vstack([d_np, g_np])
            ax.set_xlim(all_pts[:, 0].min(), all_pts[:, 0].max())
            ax.set_ylim(all_pts[:, 1].min(), all_pts[:, 1].max())
            ax.set_zlim(all_pts[:, 2].min(), all_pts[:, 2].max())
            ax.view_init(elev=25, azim=45)
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_zticklabels([])

        plt.suptitle(f'Sample {sample_idx + 1}', fontsize=12)
        plt.tight_layout()
        save_path = os.path.join(output_dir, f'comparison_sample_{sample_idx + 1:02d}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {save_path}")


def main(args):
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    checkpoints = {
        'm1': args.m1_checkpoint,
        'm3': args.m3_checkpoint,
        'm4': args.m4_checkpoint,
        'm5': args.m5_checkpoint,
        'm7': args.m7_checkpoint,
    }

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    generate_per_surface_analysis(checkpoints, args.data_root, output_dir, device)
    generate_damage_sensitivity(checkpoints, args.data_root, output_dir, device)
    generate_qualitative_comparison(checkpoints, args.data_root, output_dir, device, args.num_samples)

    print(f"\n\nAll figures saved to: {output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate article figures')
    parser.add_argument('--data_root', type=str, default='data/DMSurface')
    parser.add_argument('--output_dir', type=str, default='log/article_final/figures')
    parser.add_argument('--num_samples', type=int, default=3)
    parser.add_argument('--gpu', type=int, default=0)

    parser.add_argument('--m1_checkpoint', type=str, default=None)
    parser.add_argument('--m3_checkpoint', type=str, default=None)
    parser.add_argument('--m4_checkpoint', type=str, default=None)
    parser.add_argument('--m5_checkpoint', type=str, default=None)
    parser.add_argument('--m7_checkpoint', type=str, default=None)

    args = parser.parse_args()
    main(args)
