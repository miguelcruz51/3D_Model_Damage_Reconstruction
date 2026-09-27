"""
Generate per-surface figures for article.

For each test surface (018, 019, 020):
  - Row 1-3: 3 random partitions, each showing [Input | GT | M1 | M3 | M4 | M7]
  - Row 4:   Full surface reconstructed by M7 (all partitions merged)

Usage:
    python inference_surface_figures.py \
        --data_root data/DMSurface \
        --log_dir log/article_final \
        --output_dir log/article_final/surface_figures
"""

import argparse
import os
import glob
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa

from dataset.DMCLocalReconstruction import DMCLocalReconstruction


# ── colour palette (same as existing figures) ──────────────────────────────
C_INTACT   = '#888888'   # gray
C_DAMAGED  = '#E53935'   # red
C_GT_RECON = '#43A047'   # green
C_PRED     = '#1E88E5'   # blue


# ── model loaders ───────────────────────────────────────────────────────────
def load_m1(ckpt_path, device):
    from models.pure_pcn import PurePCN
    ckpt = torch.load(ckpt_path, map_location=device)
    a = ckpt['args']
    m = PurePCN(num_points=a.num_points, latent_dim=a.latent_dim).to(device)
    m.load_state_dict(ckpt['model_state_dict'])
    m.eval()
    return m

def load_m3(ckpt_path, device):
    from models.local_reconstruction_pcn import LocalReconstructionPCN
    ckpt = torch.load(ckpt_path, map_location=device)
    a = ckpt['args']
    m = LocalReconstructionPCN(num_points=a.num_points, latent_dim=a.latent_dim,
                                predict_ratio=a.predict_ratio).to(device)
    m.load_state_dict(ckpt['model_state_dict'])
    m.eval()
    return m

def load_m4(ckpt_path, device):
    from models.displacement_net import BlindDisplacementNet, FlowDisplacementNet
    ckpt = torch.load(ckpt_path, map_location=device)
    a = ckpt['args']
    cls = BlindDisplacementNet if a.model == 'blind' else FlowDisplacementNet
    m = cls(feature_dim=a.feature_dim).to(device)
    m.load_state_dict(ckpt['model_state_dict'])
    m.eval()
    return m

def load_m5(ckpt_path, device):
    from models.displacement_net import FlowDisplacementNet
    ckpt = torch.load(ckpt_path, map_location=device)
    a = ckpt['args']
    m = FlowDisplacementNet(feature_dim=a.feature_dim).to(device)
    m.load_state_dict(ckpt['model_state_dict'])
    m.eval()
    return m

def load_m7(ckpt_path, device):
    from models.seed_displacement_net import SeedDisplacementNet
    ckpt = torch.load(ckpt_path, map_location=device)
    a = ckpt['args']
    m = SeedDisplacementNet(latent_dim=a.latent_dim, feature_dim=a.feature_dim).to(device)
    m.load_state_dict(ckpt['model_state_dict'])
    m.eval()
    return m


# ── single-sample inference ─────────────────────────────────────────────────
@torch.no_grad()
def predict(model_key, model, pts_t, lbl_t):
    """Returns predicted points as numpy (N,3)."""
    p = pts_t.unsqueeze(0)
    l = lbl_t.unsqueeze(0)
    if model_key == 'm1':
        out = model(p, training=False).squeeze(0)
    elif model_key == 'm3':
        out = model(p, l, training=False).squeeze(0)
    elif model_key in ('m4', 'm5'):
        out, _ = model(p, l)
        out = out.squeeze(0)    
    elif model_key == 'm7':
        out, _, _ = model(p, l)
        out = out.squeeze(0)
    return out.cpu().numpy()


# ── plot one 3-D panel — same style as comparison_sample figures ─────────────
def plot_panel(ax, pts, colors, sizes, alphas, title, ref_pts):
    for p, c, s, a in zip(pts, colors, sizes, alphas):
        if len(p) > 0:
            ax.scatter(p[:, 0], p[:, 1], p[:, 2], c=c, s=s, alpha=a)
    ax.set_title(title, fontsize=9)
    mn = ref_pts.min(axis=0)
    mx = ref_pts.max(axis=0)
    ax.set_xlim(mn[0], mx[0])
    ax.set_ylim(mn[1], mx[1])
    ax.set_zlim(mn[2], mx[2])
    ax.view_init(elev=25, azim=45)

    clean_3d_axis(ax)

    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])

def clean_3d_axis(ax):
    ax.grid(False)

    # Remove panes
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    # Remove pane edges
    ax.xaxis.pane.set_edgecolor('white')
    ax.yaxis.pane.set_edgecolor('white')
    ax.zaxis.pane.set_edgecolor('white')

    # Remove ticks
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    # Remove axis lines
    try:
        ax.xaxis.line.set_color((1,1,1,0))
        ax.yaxis.line.set_color((1,1,1,0))
        ax.zaxis.line.set_color((1,1,1,0))
    except:
        pass    

    


# ── build one partition row (6 panels) ──────────────────────────────────────
def draw_partition_row(fig, n_rows, subplot_start, n_cols, sample, models, device):
    d_pts, d_lbl, g_pts, g_lbl = sample[0], sample[1], sample[2], sample[3]
    d_np  = d_pts.numpy()
    g_np  = g_pts.numpy()
    dl    = d_lbl.numpy()
    gl    = g_lbl.numpy()
    ref   = np.vstack([d_np, g_np])

    intact_d  = d_np[dl == 0];  damaged_d = d_np[dl == 1]
    intact_g  = g_np[gl == 0];  recon_g   = g_np[gl == 1]
    n_dmg     = int((dl == 1).sum())

    col_titles = ['Input', 'Ground Truth', 'M1: PCN', 'M3: LR-PCN', 'M4: BlindDisp', 'M5: FlowDisp', 'M7: Ours']
    col_keys   = [None,    None,           'm1',      'm3',          'm4',            'm5',           'm7']

    for col, (title, key) in enumerate(zip(col_titles, col_keys)):
        ax = fig.add_subplot(n_rows, n_cols, subplot_start + col, projection='3d')

        if title == 'Input':
            plot_panel(ax,
                       [intact_d, damaged_d],
                       [C_INTACT, C_DAMAGED], [0.5, 1.0], [0.5, 0.8],
                       f'Input\n({n_dmg} dmg)', ref)

        elif title == 'Ground Truth':
            plot_panel(ax,
                       [intact_g, recon_g],
                       [C_INTACT, C_GT_RECON], [0.5, 1.0], [0.5, 0.8],
                       'Ground Truth', ref)

        else:
            model = models.get(key)
            if model is None:
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_zticklabels([])
                ax.set_title(f'{title}\n(N/A)', fontsize=9)
                continue
            pred = predict(key, model, d_pts.to(device), d_lbl.to(device))
            # colour: intact gray, predicted region blue
            if len(pred) == len(dl):
                pts_list   = [pred[dl == 0], pred[dl == 1]]
                col_list   = [C_INTACT, C_PRED]
                size_list  = [0.5, 1.0]
                alpha_list = [0.5, 0.8]
            else:
                pts_list   = [pred]
                col_list   = [C_PRED]
                size_list  = [0.5]
                alpha_list = [0.6]
            plot_panel(ax, pts_list, col_list, size_list, alpha_list, title, ref)


# ── full-surface row ─────────────────────────────────────────────────────────
def draw_surface_row(fig, n_rows, subplot_start, n_cols, surface_id, data_root, model_m7, device, num_points=2048):
    """
    Merge all partitions of a surface and show:
    [Damaged surface | GT surface | M7 reconstructed]
    """
    part_dir = os.path.join(data_root, surface_id, 'partitions')
    all_files = sorted(glob.glob(os.path.join(part_dir, '*.npz')))
    damaged_files = [f for f in all_files if '_gt' not in f]

    dmg_all, gt_all, pred_all = [], [], []

    for df in damaged_files:
        gf = df.replace('.npz', '_gt.npz')
        if not os.path.exists(gf):
            continue
        d = np.load(df)
        g = np.load(gf)
        dl = d['labels'].astype(np.float32)
        gl = g['labels'].astype(np.float32)
        if (dl == 0).sum() == 0 or (gl == 0).sum() == 0:
            continue

        d_pts_np = d['points'].astype(np.float32)
        g_pts_np = g['points'].astype(np.float32)

        # normalise (same as dataset)
        centroid = d_pts_np.mean(axis=0)
        d_c = d_pts_np - centroid
        max_d = np.max(np.sqrt((d_c ** 2).sum(axis=1)))
        if max_d < 1e-6:
            max_d = 1.0
        d_norm = d_c / max_d
        g_norm = (g_pts_np - centroid) / max_d

        # resample to num_points
        def resample(pts, lbl, n):
            idx = np.random.choice(len(pts), n, replace=(len(pts) < n))
            return pts[idx], lbl[idx]

        d_r, dl_r = resample(d_norm, dl, num_points)
        g_r, gl_r = resample(g_norm, gl, num_points)

        d_t = torch.from_numpy(d_r)
        dl_t = torch.from_numpy(dl_r)

        if model_m7 is not None:
            pred = predict('m7', model_m7, d_t.to(device), dl_t.to(device))
        else:
            pred = d_r.copy()

        # denormalise back to original space for surface assembly
        dmg_all.append(d_pts_np)
        gt_all.append(g_pts_np)
        pred_all.append(pred * max_d + centroid)

    if not dmg_all:
        return

    dmg_surf  = np.vstack(dmg_all)
    gt_surf   = np.vstack(gt_all)
    pred_surf = np.vstack(pred_all)

    # subsample for display (200k pts per partition × many partitions = too many)
    def subsample(pts, n=50000):
        if len(pts) > n:
            idx = np.random.choice(len(pts), n, replace=False)
            return pts[idx]
        return pts

    dmg_surf  = subsample(dmg_surf)
    gt_surf   = subsample(gt_surf)
    pred_surf = subsample(pred_surf)

    ref = np.vstack([dmg_surf, gt_surf])
    panels = [
        ('Damaged Surface', dmg_surf,  C_DAMAGED),
        ('GT Surface',      gt_surf,   C_GT_RECON),
        ('M7 Reconstructed', pred_surf, C_PRED),
    ]

    # centre the 3 panels in the row (cols 3,4,5 out of 7 → index offset 2)
    start_col = (n_cols - 3) // 2
    for i, (title, pts, color) in enumerate(panels):
        ax = fig.add_subplot(n_rows, n_cols, subplot_start + start_col + i, projection='3d')
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                   c=color, s=0.5, alpha=0.6)
        ax.set_title(title, fontsize=9)
        mn, mx = ref.min(axis=0), ref.max(axis=0)
        ax.set_xlim(mn[0], mx[0]); ax.set_ylim(mn[1], mx[1]); ax.set_zlim(mn[2], mx[2])
        ax.view_init(elev=25, azim=45)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])


# ── main figure per surface ──────────────────────────────────────────────────
def generate_surface_figure(surface_id, data_root, models, device, output_dir,
                             n_partitions=7, seed=42):
    np.random.seed(seed)

    dataset = DMCLocalReconstruction(
        data_root=data_root, split='test',
        components=[surface_id], num_points=2048,
        augment=False, only_damaged=True
    )

    if len(dataset) == 0:
        print(f"  No samples for surface {surface_id}, skipping.")
        return

    n_pick = min(n_partitions, len(dataset))
    indices = np.random.choice(len(dataset), n_pick, replace=False)

    surf_dir = os.path.join(output_dir, f'surface_{surface_id}')
    os.makedirs(surf_dir, exist_ok=True)

    n_cols = 7  # Input | GT | M1 | M3 | M4 | M5 | M7

    # ── one file per partition row ──
    for row_i, data_idx in enumerate(indices):
        sample = dataset[int(data_idx)]

        fig = plt.figure(figsize=(n_cols * 3.2, 3.5))
        fig.suptitle(f'Surface {surface_id} — Partition {row_i + 1}', fontsize=11)
        draw_partition_row(fig, 1, 1, n_cols, sample, models, device)
        plt.tight_layout()

        out_path = os.path.join(surf_dir, f'partition_{row_i + 1:02d}.png')
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {out_path}")

    # ── one file for the full surface row ──
    fig = plt.figure(figsize=(n_cols * 3.2, 3.5))
    fig.suptitle(f'Surface {surface_id} — Full Surface', fontsize=11)
    draw_surface_row(fig, 1, 1, n_cols, surface_id, data_root, models.get('m7'), device)
    plt.tight_layout()

    out_path = os.path.join(surf_dir, 'full_surface.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")


# ── entry point ──────────────────────────────────────────────────────────────
def main(args):
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # load models (skip if checkpoint not provided)
    models = {}
    loaders = {'m1': load_m1, 'm3': load_m3, 'm4': load_m4, 'm5': load_m5, 'm7': load_m7}
    ckpts   = {'m1': args.m1_checkpoint, 'm3': args.m3_checkpoint,
               'm4': args.m4_checkpoint, 'm5': args.m5_checkpoint, 'm7': args.m7_checkpoint}

    for key, loader in loaders.items():
        path = ckpts[key]
        if path and os.path.exists(path):
            print(f"Loading {key} from {path}")
            models[key] = loader(path, device)
        else:
            print(f"Skipping {key} (no checkpoint)")

    for surface_id in args.surfaces:
        print(f"\nGenerating figure for surface {surface_id}...")
        generate_surface_figure(
            surface_id, args.data_root, models, device,
            args.output_dir, n_partitions=args.n_partitions, seed=args.seed
        )

    print("\nDone.")


if __name__ == '__main__':
    BASE = 'log/article_final'
    parser = argparse.ArgumentParser(description='Per-surface article figures')
    parser.add_argument('--data_root',    type=str, default='data/DMC')
    parser.add_argument('--output_dir',   type=str, default=f'{BASE}/surface_figures')
    parser.add_argument('--surfaces',     type=str, nargs='+', default=['001', '002', '003', '004', '005','006', '007', '008', '009', '010', '011', '012', '013', '014', '015', '016'])
    parser.add_argument('--n_partitions', type=int, default=5)
    parser.add_argument('--seed',         type=int, default=42)
    parser.add_argument('--gpu',          type=int, default=0)
    parser.add_argument('--m1_checkpoint', type=str, default=f'{BASE}/m1_pure_pcn/best_model.pth')
    parser.add_argument('--m3_checkpoint', type=str, default=f'{BASE}/m3_lrpcn_label_guided/best_model.pth')
    parser.add_argument('--m4_checkpoint', type=str, default=f'{BASE}/m4_blind_displacement/best_model.pth')
    parser.add_argument('--m5_checkpoint', type=str, default=f'{BASE}/m5_flow_displacement/best_model.pth')
    parser.add_argument('--m7_checkpoint', type=str, default=f'{BASE}/m7_seed_displacement/best_model.pth')
    args = parser.parse_args()
    main(args)
