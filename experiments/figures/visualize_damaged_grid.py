"""
Visualize DMSurface damaged point clouds (001-020) as a grid image.
Rendering style matches infer_interactive.py:
  blue  [0.2, 0.4, 0.9] = intact points  (label 0)
  red   [0.9, 0.1, 0.1] = damaged points (label 1)

Uses the first damaged version (001) of each surface by default.
Points are depth-shaded so the cloud reads as 3-D.

Usage:
    python visualize_damaged_grid.py
    python visualize_damaged_grid.py --cols 5 --cell_size 400 --dpi 120
    python visualize_damaged_grid.py --version 002 --no_labels
    python visualize_damaged_grid.py --max_pts 80000 --pt_size 0.5
"""

import argparse
import math
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATA_DIR    = "/home/miguelcruz/Documents/D-PCN2/data/DMSurface"
SURFACE_IDS = [f"{i:03d}" for i in range(1, 21)]

# ── same palette as infer_interactive.py ─────────────────────────────────────
COLOR_INTACT  = np.array([0.2, 0.4, 0.9])   # blue
COLOR_DAMAGED = np.array([0.9, 0.1, 0.1])   # red


def load_damaged(surface_id, version):
    path = os.path.join(DATA_DIR, surface_id, f"{surface_id}{version}.npz")
    d = np.load(path)
    return d['points'].astype(np.float32), d['labels'].astype(np.int32)


def depth_shade(pts_norm, base_colors, elev_deg, azim_deg, strength=0.35):
    """
    Modulate brightness by a simple dot-product with the camera direction,
    giving a depth-shading effect that makes the cloud read as 3-D.
    base_colors: (N, 3) float in [0, 1]
    Returns (N, 4) RGBA array.
    """
    elev = np.radians(elev_deg)
    azim = np.radians(azim_deg)
    # Camera look-direction (pointing toward origin)
    cam = np.array([
        np.cos(elev) * np.cos(azim),
        np.cos(elev) * np.sin(azim),
        np.sin(elev)
    ])
    # Approximate "depth" as projection onto camera axis
    proj = pts_norm @ cam                          # (N,)
    proj = (proj - proj.min()) / (np.ptp(proj) + 1e-6)   # normalise to [0,1]
    # Brighter = closer to camera
    brightness = 1.0 - strength + strength * proj  # [1-strength, 1]
    shaded = np.clip(base_colors * brightness[:, None], 0, 1)
    alpha  = np.full(len(pts_norm), 0.85)
    return np.hstack([shaded, alpha[:, None]])


def render_pcd_to_ax(ax, pts, labels, title, show_label, label_size,
                     elev, azim, pt_size):
    # ── normalise to unit sphere ──────────────────────────────────────────
    center = pts.mean(axis=0)
    pts    = pts - center
    scale  = np.abs(pts).max()
    if scale > 0:
        pts = pts / scale

    # ── base colors (infer_interactive palette) ───────────────────────────
    base = np.where(labels[:, None] == 0, COLOR_INTACT, COLOR_DAMAGED)

    # ── depth shading ─────────────────────────────────────────────────────
    rgba = depth_shade(pts, base, elev, azim)

    # ── sort back-to-front so closer points paint on top ─────────────────
    elev_r = np.radians(elev)
    azim_r = np.radians(azim)
    cam    = np.array([np.cos(elev_r)*np.cos(azim_r),
                       np.cos(elev_r)*np.sin(azim_r),
                       np.sin(elev_r)])
    depth  = pts @ cam
    order  = np.argsort(depth)          # back first
    pts, rgba = pts[order], rgba[order]

    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
               c=rgba, s=pt_size, linewidths=0, depthshade=False)

    ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(-1, 1)
    ax.set_axis_off()
    if show_label:
        n_dmg = int((labels == 1).sum())
        pct   = 100 * n_dmg / max(len(labels), 1)
        ax.set_title(f"{title}\n{n_dmg} dmg ({pct:.1f}%)",
                     fontsize=label_size, pad=2)
    ax.view_init(elev=elev, azim=azim)


def main():
    p = argparse.ArgumentParser(
        description="DMSurface damaged PCD grid — infer_interactive colour scheme"
    )
    p.add_argument('--cols',       type=int,   default=5,
                   help='Grid columns (default: 5)')
    p.add_argument('--cell_size',  type=int,   default=350,
                   help='Cell size in pixels (default: 350)')
    p.add_argument('--dpi',        type=int,   default=100,
                   help='Output DPI (default: 100)')
    p.add_argument('--version',    type=str,   default='001',
                   help='Damage version to load (default: 001)')
    p.add_argument('--max_pts',    type=int,   default=50000,
                   help='Max points per surface (default: 50000)')
    p.add_argument('--pt_size',    type=float, default=0.4,
                   help='Scatter point size (default: 0.4)')
    p.add_argument('--no_labels',  action='store_true',
                   help='Hide titles')
    p.add_argument('--label_size', type=int,   default=7,
                   help='Title font size (default: 7)')
    p.add_argument('--elev',       type=float, default=25,
                   help='Camera elevation (default: 25)')
    p.add_argument('--azim',       type=float, default=-60,
                   help='Camera azimuth (default: -60)')
    args = p.parse_args()

    version     = args.version.zfill(3)
    show_labels = not args.no_labels
    rng         = np.random.default_rng(42)

    n    = len(SURFACE_IDS)
    rows = math.ceil(n / args.cols)
    fw   = args.cols * args.cell_size / args.dpi
    fh   = rows      * args.cell_size / args.dpi
    print(f"Grid: {args.cols}x{rows} | "
          f"{int(fw*args.dpi)}x{int(fh*args.dpi)} px @ {args.dpi} dpi  "
          f"| {args.max_pts} pts/surface")

    fig = plt.figure(figsize=(fw, fh), dpi=args.dpi)
    fig.patch.set_facecolor('white')

    for idx, sid in enumerate(SURFACE_IDS):
        print(f"  [{idx+1:02d}/20] {sid} v{version} ...", end=' ', flush=True)
        ax = fig.add_subplot(rows, args.cols, idx + 1, projection='3d')
        try:
            pts, labels = load_damaged(sid, version)

            # stratified subsample: keep proportions of intact / damaged
            if len(pts) > args.max_pts:
                intact_idx  = np.where(labels == 0)[0]
                damaged_idx = np.where(labels == 1)[0]
                ratio       = len(damaged_idx) / max(len(pts), 1)
                n_dmg  = min(len(damaged_idx),
                             int(args.max_pts * ratio) + 1)
                n_int  = min(len(intact_idx),
                             args.max_pts - n_dmg)
                sel = np.concatenate([
                    rng.choice(intact_idx,  n_int,  replace=False),
                    rng.choice(damaged_idx, n_dmg,  replace=False),
                ])
                pts, labels = pts[sel], labels[sel]

            print(f"{len(pts)} pts  "
                  f"(dmg={int((labels==1).sum())})")

            render_pcd_to_ax(ax, pts, labels,
                             title=f"Surface {sid}",
                             show_label=show_labels,
                             label_size=args.label_size,
                             elev=args.elev,
                             azim=args.azim,
                             pt_size=args.pt_size)
        except Exception as e:
            print(f"ERROR: {e}")
            ax.set_axis_off()
            if show_labels:
                ax.set_title(f"{sid} (err)", fontsize=args.label_size)

    plt.tight_layout(pad=0.3)
    out = (f"dmsurface_damaged_grid_v{version}_{args.cols}x{rows}"
           f"_{int(fw*args.dpi)}x{int(fh*args.dpi)}.png")
    fig.savefig(out, dpi=args.dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"\nSaved: {out}")


if __name__ == '__main__':
    main()
