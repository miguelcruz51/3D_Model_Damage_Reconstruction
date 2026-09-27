
"""
Case Study Inference Script (No GT Required)

Output:
[Full | BBox | Input | Output (movement-colored)]
"""

import argparse
import os
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ── colors ────────────────────────────────────────────────────────────────
C_INTACT  = '#888888'
C_DAMAGED = '#E53935'
C_PRED    = '#1E88E5'
C_MOVED   = '#43A047'  # green


# ── model loader ──────────────────────────────────────────────────────────
def load_m7(ckpt_path, device):
    from models.seed_displacement_net import SeedDisplacementNet
    ckpt = torch.load(ckpt_path, map_location=device)
    a = ckpt['args']
    m = SeedDisplacementNet(latent_dim=a.latent_dim,
                            feature_dim=a.feature_dim).to(device)
    m.load_state_dict(ckpt['model_state_dict'])
    m.eval()
    return m


# ── inference ─────────────────────────────────────────────────────────────
@torch.no_grad()
def predict_m7(model, pts_t, lbl_t):
    p = pts_t.unsqueeze(0)
    l = lbl_t.unsqueeze(0)
    out, _, _ = model(p, l)
    return out.squeeze(0).cpu().numpy()


# ── clean 3D axis ─────────────────────────────────────────────────────────
def clean_3d_axis(ax):
    ax.grid(False)

    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    ax.xaxis.pane.set_edgecolor('white')
    ax.yaxis.pane.set_edgecolor('white')
    ax.zaxis.pane.set_edgecolor('white')

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    try:
        ax.xaxis.line.set_color((1,1,1,0))
        ax.yaxis.line.set_color((1,1,1,0))
        ax.zaxis.line.set_color((1,1,1,0))
    except:
        pass


# ── draw bbox ─────────────────────────────────────────────────────────────
def draw_bbox(ax, bb_min, bb_max, color='orange'):
    x = [bb_min[0], bb_max[0]]
    y = [bb_min[1], bb_max[1]]
    z = [bb_min[2], bb_max[2]]

    for xi in x:
        for yi in y:
            ax.plot([xi, xi], [yi, yi], z, c=color, linewidth=1.5)

    for xi in x:
        for zi in z:
            ax.plot([xi, xi], y, [zi, zi], c=color, linewidth=1.5)

    for yi in y:
        for zi in z:
            ax.plot(x, [yi, yi], [zi, zi], c=color, linewidth=1.5)


# ── plot row ──────────────────────────────────────────────────────────────
def draw_case_study_row(fig, full_pts, part_pts, part_lbl, pred_pts):

    intact = part_pts[part_lbl == 0]
    damaged = part_pts[part_lbl == 1]

    bb_min = part_pts.min(axis=0)
    bb_max = part_pts.max(axis=0)

    # ── Panel 1: Full ──
    ax = fig.add_subplot(1, 4, 1, projection='3d')
    ax.scatter(full_pts[:,0], full_pts[:,1], full_pts[:,2],
               c=C_INTACT, s=0.3, alpha=0.6)
    ax.set_title('Full Surface', fontsize=9)
    ax.view_init(elev=20, azim=210)
    clean_3d_axis(ax)

    # ── Panel 2: BBox ──
    ax = fig.add_subplot(1, 4, 2, projection='3d')
    ax.scatter(full_pts[:,0], full_pts[:,1], full_pts[:,2],
               c=C_INTACT, s=0.3, alpha=0.2)
    draw_bbox(ax, bb_min, bb_max)
    ax.scatter(part_pts[:,0], part_pts[:,1], part_pts[:,2],
               c='orange', s=1.0, alpha=0.9)
    ax.set_title('Partition Location', fontsize=9)
    ax.view_init(elev=20, azim=210)
    clean_3d_axis(ax)

    # ── Panel 3: Input ──
    ax = fig.add_subplot(1, 4, 3, projection='3d')
    ax.scatter(intact[:,0], intact[:,1], intact[:,2],
               c=C_INTACT, s=0.5, alpha=0.5)
    ax.scatter(damaged[:,0], damaged[:,1], damaged[:,2],
               c=C_DAMAGED, s=1.0, alpha=0.9)
    ax.set_title('Input Partition', fontsize=9)
    ax.view_init(elev=20, azim=210)
    clean_3d_axis(ax)

    # ── Panel 4: Output (movement-based coloring) ──
    ax = fig.add_subplot(1, 4, 4, projection='3d')

    disp = np.linalg.norm(pred_pts - part_pts, axis=1)

    # adaptive threshold (more robust)
    th = np.percentile(disp, 80)

    moved = disp > th
    static = ~moved

    ax.scatter(pred_pts[static,0], pred_pts[static,1], pred_pts[static,2],
               c=C_INTACT, s=0.5, alpha=0.3)

    ax.scatter(pred_pts[moved,0], pred_pts[moved,1], pred_pts[moved,2],
               c=C_MOVED, s=1.0, alpha=0.9)

    ax.set_title('M7 Reconstruction', fontsize=9)
    ax.view_init(elev=20, azim=210)
    clean_3d_axis(ax)


# ── main per surface ──────────────────────────────────────────────────────
def generate_case_study(surface_id, args, model, device):

    surf_dir = os.path.join(args.data_root, surface_id)

    # ── load full PCD ──
    candidates = [
        f for f in os.listdir(surf_dir)
        if f.endswith('.npz') and not f.startswith('partitions')
    ]

    if len(candidates) == 0:
        raise FileNotFoundError(f"No full PCD found in {surf_dir}")

    full_file = sorted(candidates, key=len)[0]
    print(f"Using full PCD: {full_file}")

    full_pts = np.load(os.path.join(surf_dir, full_file))['points']

    # ── partitions ──
    part_dir = os.path.join(surf_dir, 'partitions')
    all_files = sorted(os.listdir(part_dir))

    part_files = [
        f for f in all_files
        if f.endswith('01.npz') and f.startswith(surface_id)
    ]

    # select most damaged
    part_files = sorted(
        part_files,
        key=lambda f: np.load(os.path.join(part_dir, f))['labels'].sum(),
        reverse=True
    )

    selected = part_files[:args.n_partitions]

    out_dir = os.path.join(args.output_dir, surface_id)
    os.makedirs(out_dir, exist_ok=True)

    for i, fname in enumerate(selected):

        data = np.load(os.path.join(part_dir, fname))
        pts = data['points'].astype(np.float32)
        lbl = data['labels'].astype(np.float32)

        # normalize
        centroid = pts.mean(axis=0)
        pts_centered = pts - centroid

        scale = np.max(np.sqrt((pts_centered ** 2).sum(axis=1)))
        if scale < 1e-6:
            scale = 1.0

        pts_norm = pts_centered / scale

        # inference
        pts_t = torch.from_numpy(pts_norm)
        lbl_t = torch.from_numpy(lbl)

        pred = predict_m7(model, pts_t.to(device), lbl_t.to(device))

        # denormalize
        pts_world  = pts_norm * scale + centroid
        pred_world = pred * scale + centroid

        # plot
        fig = plt.figure(figsize=(12, 3.5))
        fig.suptitle(f"{surface_id} — Case Study {i+1}", fontsize=11)

        draw_case_study_row(fig, full_pts, pts_world, lbl, pred_world)

        plt.tight_layout()

        save_path = os.path.join(out_dir, f'case_{i+1:02d}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Saved: {save_path}")


# ── main ──────────────────────────────────────────────────────────────────
def main(args):
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print("Device:", device)

    model = load_m7(args.m7_checkpoint, device)

    for sid in args.surfaces:
        print(f"\nProcessing {sid}")
        generate_case_study(sid, args, model, device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='data/CaseStudy')
    parser.add_argument('--output_dir', type=str, default='log/case_study')
    parser.add_argument('--surfaces', nargs='+', default=['001'])
    parser.add_argument('--n_partitions', type=int, default=5)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--m7_checkpoint', type=str, required=True)

    args = parser.parse_args()
    main(args)

