"""
Generate qualitative comparison grid for the article.

Layout: 
- Columns: 3 partitions (one random partition per surface 018, 019, 020)
- Rows: Input, Ground Truth, PCN, LR-PCN, PointCleanNet, FlowNet3D, SeedDispNet

Colors:
- Input: grey (intact) + red (damaged)
- GT: grey (intact) + green (reconstruction targets)
- Predictions: grey (intact) + blue (reconstructed)

Generates 10 images with different random partitions so you can pick the best.
"""

import argparse
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

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
        from models.seed_displacement_net import SeedDisplacementNet, SeedDisplacementNetNoLocal
        ckpt = torch.load(checkpoint_path, map_location=device)
        args = ckpt['args']
        if hasattr(args, 'no_local') and args.no_local:
            model = SeedDisplacementNetNoLocal(
                latent_dim=args.latent_dim, feature_dim=args.feature_dim
            ).to(device)
        else:
            model = SeedDisplacementNet(
                latent_dim=args.latent_dim, feature_dim=args.feature_dim
            ).to(device)
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()
        with torch.no_grad():
            pred, _, _ = model(d_pts_b, d_lbl_b)
        return pred.squeeze(0).cpu().numpy()


def render_panel(ax, points, labels, mode, title=None):
    """
    Render a single 3D point cloud panel.
    
    mode: 'input', 'gt', or 'pred'
    """
    pts = points if isinstance(points, np.ndarray) else points.numpy()
    lbl = labels if isinstance(labels, np.ndarray) else labels.numpy()

    intact_mask = lbl == 0
    damaged_mask = lbl == 1

    if mode == 'input':
        color_intact = '#888888'
        color_highlight = '#CC0000'
    elif mode == 'gt':
        color_intact = '#888888'
        color_highlight = '#00AA00'
    else:  # pred
        color_intact = '#888888'
        color_highlight = '#0066CC'

    if intact_mask.any():
        ax.scatter(pts[intact_mask, 0], pts[intact_mask, 1], pts[intact_mask, 2],
                   c=color_intact, s=0.3, alpha=0.4, rasterized=True)
    if damaged_mask.any():
        ax.scatter(pts[damaged_mask, 0], pts[damaged_mask, 1], pts[damaged_mask, 2],
                   c=color_highlight, s=0.8, alpha=0.85, rasterized=True)

    if title:
        ax.set_title(title, fontsize=9, pad=2)

    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')
    ax.grid(False)


def generate_one_figure(surfaces, datasets, checkpoints, device, seed, output_path):
    """Generate one comparison grid figure with a given random seed."""
    np.random.seed(seed)

    methods = [
        ('Input', None, None),
        ('Ground Truth', None, None),
        ('PCN', 'm1', checkpoints.get('m1')),
        ('LR-PCN', 'm3', checkpoints.get('m3')),
        ('PointCleanNet', 'm4', checkpoints.get('m4')),
        ('FlowNet3D', 'm5', checkpoints.get('m5')),
        ('SeedDispNet (Ours)', 'm7', checkpoints.get('m7')),
    ]

    n_rows = len(methods)
    n_cols = len(surfaces)

    # Pick one random partition per surface
    samples = []
    for surf, ds in zip(surfaces, datasets):
        idx = np.random.randint(0, len(ds))
        samples.append(ds[idx])

    # Determine consistent view angles from the first sample
    fig = plt.figure(figsize=(3.5 * n_cols, 2.8 * n_rows))

    for row, (method_name, method_key, ckpt_path) in enumerate(methods):
        for col, (surf, sample) in enumerate(zip(surfaces, samples)):
            d_pts, d_lbl, g_pts, g_lbl, scale_mm = sample
            d_np = d_pts.numpy()
            g_np = g_pts.numpy()
            dl = d_lbl.numpy()
            gl = g_lbl.numpy()

            ax = fig.add_subplot(n_rows, n_cols, row * n_cols + col + 1, projection='3d')

            if method_name == 'Input':
                render_panel(ax, d_np, dl, 'input',
                             title=f'Surface {surf}' if row == 0 else None)
            elif method_name == 'Ground Truth':
                render_panel(ax, g_np, gl, 'gt')
            else:
                if ckpt_path is None:
                    ax.set_title('N/A', fontsize=9)
                    continue
                pred = load_model_and_predict(method_key, ckpt_path, sample, device)
                # For methods that output same size as input, use input labels
                if len(pred) == len(dl):
                    render_panel(ax, pred, dl, 'pred')
                else:
                    # For generative methods (PCN), show all as blue
                    fake_lbl = np.ones(len(pred))
                    render_panel(ax, pred, fake_lbl, 'pred')

            # Consistent bounds
            all_pts = np.vstack([d_np, g_np])
            margin = 0.05
            for axis_fn, dim in [(ax.set_xlim, 0), (ax.set_ylim, 1), (ax.set_zlim, 2)]:
                lo, hi = all_pts[:, dim].min(), all_pts[:, dim].max()
                axis_fn(lo - margin, hi + margin)

            ax.view_init(elev=25, azim=135)

            # Row labels on leftmost column
            if col == 0:
                ax.set_ylabel(method_name, fontsize=9, labelpad=15)

    # Legend
    legend_elements = [
        Patch(facecolor='#888888', label='Intact'),
        Patch(facecolor='#CC0000', label='Damaged (Input)'),
        Patch(facecolor='#00AA00', label='Reconstruction Target (GT)'),
        Patch(facecolor='#0066CC', label='Reconstructed (Prediction)'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=4,
               fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.01))

    plt.tight_layout(rect=[0, 0.02, 1, 1])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def main(args):
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    surfaces = args.surfaces
    checkpoints = {
        'm1': args.m1_checkpoint,
        'm3': args.m3_checkpoint,
        'm4': args.m4_checkpoint,
        'm5': args.m5_checkpoint,
        'm7': args.m7_checkpoint,
    }

    os.makedirs(args.output_dir, exist_ok=True)

    # Load datasets for each surface
    print("Loading datasets...")
    datasets = []
    for surf in surfaces:
        ds = DMCLocalReconstruction(
            data_root=args.data_root, split='test',
            components=[surf], num_points=2048,
            augment=False, only_damaged=True
        )
        datasets.append(ds)
        print(f"  Surface {surf}: {len(ds)} damaged partitions")

    # Generate N figures with different random seeds
    print(f"\nGenerating {args.num_figures} figures...")
    for i in range(args.num_figures):
        seed = args.base_seed + i
        output_path = os.path.join(args.output_dir, f'comparison_grid_{i+1:02d}.png')
        generate_one_figure(surfaces, datasets, checkpoints, device, seed, output_path)

    print(f"\nDone! {args.num_figures} figures saved to: {args.output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate comparison grid figures')
    parser.add_argument('--data_root', type=str, default='data/DMSurface')
    parser.add_argument('--surfaces', nargs='+', default=['018', '019', '020'],
                        help='Surface IDs to use as columns')
    parser.add_argument('--output_dir', type=str, default='log/article_final_corr/comparison_grids')
    parser.add_argument('--num_figures', type=int, default=10,
                        help='Number of figures to generate (pick the best)')
    parser.add_argument('--base_seed', type=int, default=100,
                        help='Base random seed (incremented per figure)')
    parser.add_argument('--gpu', type=int, default=0)

    parser.add_argument('--m1_checkpoint', type=str, default=None)
    parser.add_argument('--m3_checkpoint', type=str, default=None)
    parser.add_argument('--m4_checkpoint', type=str, default=None)
    parser.add_argument('--m5_checkpoint', type=str, default=None)
    parser.add_argument('--m7_checkpoint', type=str, default=None)

    args = parser.parse_args()
    main(args)
