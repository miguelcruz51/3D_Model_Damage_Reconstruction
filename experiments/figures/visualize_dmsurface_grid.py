"""
Visualize DMSurface dataset as a grid image, with optional partition mode.

GRID MODE (default):
    python visualize_dmsurface_grid.py
    python visualize_dmsurface_grid.py --cols 10 --cell_size 200 --dpi 150
    python visualize_dmsurface_grid.py --no_labels --mesh_scale 0.9

PARTITION MODE:
    python visualize_dmsurface_grid.py --partition --surface 001 --version 001
    python visualize_dmsurface_grid.py --partition --surface 005 --version 003 --num_partitions 4
    python visualize_dmsurface_grid.py --partition --surface 001 --version 001 --open3d

Partition mode produces an image showing:
  [0] Full damaged surface (grey=intact, red=damaged)
  [1] Grey surface + black wireframe bboxes for all partitions
  [2..N] Individual partitions standalone (grey=intact, red=damaged)

With --open3d: opens interactive Open3D windows for each panel.
"""

import argparse
import os
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import trimesh

DATA_DIR = "/home/miguelcruz/Documents/D-PCN2/data/DMSurface"
SURFACE_IDS = [f"{i:03d}" for i in range(1, 21)]

# ─── Shared rendering helpers ────────────────────────────────────────────────

def normalize_verts(verts):
    center = verts.mean(axis=0)
    verts = verts - center
    scale = np.abs(verts).max()
    return verts / scale if scale > 0 else verts, center, scale


def shade_mesh(mesh, cmap_name='Blues', base=0.3, spread=0.5):
    """Return per-face RGBA colors using a simple directional light."""
    light = np.array([0.5, 0.5, 1.0])
    light /= np.linalg.norm(light)
    intensity = np.clip(mesh.face_normals @ light, 0.1, 1.0)
    return plt.get_cmap(cmap_name)(base + spread * intensity)


def render_stl_to_ax(ax, mesh, title, label_size, show_label, elev=25, azim=-60):
    verts = mesh.vertices.copy()
    verts, _, _ = normalize_verts(verts)
    poly = Poly3DCollection(verts[mesh.faces], linewidth=0, edgecolor=None, alpha=0.88)
    poly.set_facecolor(shade_mesh(mesh))
    ax.add_collection3d(poly)
    ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(-1, 1)
    ax.set_axis_off()
    if show_label:
        ax.set_title(title, fontsize=label_size, pad=2)
    ax.view_init(elev=elev, azim=azim)


def scatter_pcd(ax, pts, colors, size=1.0, alpha=0.6):
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
               c=colors, s=size, alpha=alpha, linewidths=0)


def set_ax_limits(ax, pts, margin=0.05):
    mn, mx = pts.min(axis=0), pts.max(axis=0)
    rng = (mx - mn).max() / 2 + margin
    mid = (mn + mx) / 2
    ax.set_xlim(mid[0]-rng, mid[0]+rng)
    ax.set_ylim(mid[1]-rng, mid[1]+rng)
    ax.set_zlim(mid[2]-rng, mid[2]+rng)


# ─── Grid mode ───────────────────────────────────────────────────────────────

def load_stl(surface_id):
    path = os.path.join(DATA_DIR, surface_id, f"{surface_id}_nom.STL")
    return trimesh.load(path, force='mesh')


def render_grid(cols, cell_px, dpi, show_labels, label_size, mesh_scale, elev, azim):
    n = len(SURFACE_IDS)
    rows = math.ceil(n / cols)
    fw = cols * cell_px / dpi
    fh = rows * cell_px / dpi
    print(f"Grid: {cols}x{rows} | {int(fw*dpi)}x{int(fh*dpi)} px @ {dpi} dpi")

    fig = plt.figure(figsize=(fw, fh), dpi=dpi)
    fig.patch.set_facecolor('white')

    for idx, sid in enumerate(SURFACE_IDS):
        print(f"  Loading {sid}...")
        try:
            mesh = load_stl(sid)
            # Apply mesh_scale: shrink/grow the mesh inside the cell
            mesh.vertices *= mesh_scale
        except Exception as e:
            print(f"    ERROR: {e}")
            mesh = None

        ax = fig.add_subplot(rows, cols, idx + 1, projection='3d')
        if mesh is not None:
            render_stl_to_ax(ax, mesh, f"Surface {sid}", label_size,
                             show_labels, elev=elev, azim=azim)
        else:
            ax.set_axis_off()
            if show_labels:
                ax.set_title(f"{sid} (err)", fontsize=label_size)

    plt.tight_layout(pad=0.3)
    out = f"dmsurface_grid_{cols}x{rows}_{int(fw*dpi)}x{int(fh*dpi)}.png"
    fig.savefig(out, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved: {out}")


# ─── Partition mode ──────────────────────────────────────────────────────────

def parse_partition_filename(fname):
    """
    Parse: {surf3}{version3}{partition4}{flag2}.npz
    e.g.  001001000001.npz -> surf=001, ver=001, part=0000, flag=01
    """
    base = fname.replace('_gt.npz', '').replace('.npz', '')
    if len(base) != 12:
        return None
    return {
        'surf':    base[0:3],
        'version': base[3:6],
        'part':    base[6:10],
        'flag':    base[10:12],
    }


def load_damaged_surface(surface_id, version):
    """Load the full damaged surface npz (200k points)."""
    path = os.path.join(DATA_DIR, surface_id, f"{surface_id}{version}.npz")
    d = np.load(path)
    return d['points'], d['labels']   # (N,3), (N,)


def load_partitions_for_version(surface_id, version):
    """
    Return list of dicts with keys: part, flag, points, labels, gt_points, gt_labels
    Only returns partitions that have a _gt counterpart (flag='01').
    """
    part_dir = os.path.join(DATA_DIR, surface_id, 'partitions')
    files = sorted(os.listdir(part_dir))
    partitions = []
    seen = set()

    for fname in files:
        if '_gt' in fname or not fname.endswith('.npz'):
            continue
        info = parse_partition_filename(fname)
        if info is None:
            continue
        if info['surf'] != surface_id or info['version'] != version:
            continue
        if info['flag'] != '01':   # only damaged partitions
            continue
        key = info['part']
        if key in seen:
            continue
        seen.add(key)

        gt_fname = fname.replace('.npz', '_gt.npz')
        gt_path = os.path.join(part_dir, gt_fname)
        if not os.path.exists(gt_path):
            continue

        d  = np.load(os.path.join(part_dir, fname))
        dg = np.load(gt_path)
        partitions.append({
            'part':      key,
            'flag':      info['flag'],
            'points':    d['points'],
            'labels':    d['labels'],
            'gt_points': dg['points'],
            'gt_labels': dg['labels'],
        })

    return partitions


def render_partition_mode(surface_id, version, num_partitions, cell_px, dpi,
                          show_labels, label_size, elev, azim, use_open3d=False):
    print(f"\nPartition mode: surface={surface_id}, version={version}")

    # ── Load data ──────────────────────────────────────────────────────────
    print("  Loading full damaged surface...")
    try:
        dmg_pts, dmg_lbl = load_damaged_surface(surface_id, version)
    except FileNotFoundError as e:
        print(f"  ERROR: {e}")
        return

    print("  Loading partitions...")
    partitions = load_partitions_for_version(surface_id, version)
    if not partitions:
        print("  No damaged partitions found for this version.")
        return
    print(f"  Found {len(partitions)} damaged partitions")

    # Pick a random subset to highlight
    rng = np.random.default_rng(42)
    highlight = partitions[:num_partitions] if len(partitions) >= num_partitions \
                else partitions

    # ── Figure layout: 1 row, (2 + num_partitions) cols ───────────────────
    total_panels = 2 + len(highlight)
    fw = total_panels * cell_px / dpi
    fh = cell_px / dpi
    print(f"  Output: {int(fw*dpi)}x{int(fh*dpi)} px @ {dpi} dpi")

    fig = plt.figure(figsize=(fw, fh), dpi=dpi)
    fig.patch.set_facecolor('white')

    # Subsample full surface for speed (max 30k pts)
    max_pts = 30_000
    if len(dmg_pts) > max_pts:
        idx = rng.choice(len(dmg_pts), max_pts, replace=False)
        dmg_pts_vis = dmg_pts[idx]
        dmg_lbl_vis = dmg_lbl[idx]
    else:
        dmg_pts_vis, dmg_lbl_vis = dmg_pts, dmg_lbl

    # Normalize full surface
    center = dmg_pts_vis.mean(axis=0)
    scale  = np.abs(dmg_pts_vis - center).max()
    scale  = scale if scale > 0 else 1.0
    dmg_norm = (dmg_pts_vis - center) / scale

    def norm_part(pts):
        return (pts - center) / scale

    # Color map: grey=intact(0), red=damaged(1)
    def label_colors(labels):
        colors = np.where(labels[:, None] == 0,
                          np.array([[0.55, 0.55, 0.55, 0.7]]),
                          np.array([[0.9,  0.2,  0.2,  0.9]]))
        return colors

    # ── Panel 0: Full damaged surface ─────────────────────────────────────
    ax0 = fig.add_subplot(1, total_panels, 1, projection='3d')
    colors0 = label_colors(dmg_lbl_vis)
    scatter_pcd(ax0, dmg_norm, colors0, size=0.3, alpha=0.5)
    set_ax_limits(ax0, dmg_norm)
    ax0.set_axis_off()
    ax0.view_init(elev=elev, azim=azim)
    if show_labels:
        ax0.set_title(f"Damaged\n{surface_id} v{version}", fontsize=label_size, pad=2)

    # ── Panel 1: Damaged surface + partition bboxes ───────────────────────
    ax1 = fig.add_subplot(1, total_panels, 2, projection='3d')
    # Draw full surface in grey
    scatter_pcd(ax1, dmg_norm, [[0.55, 0.55, 0.55, 0.4]], size=0.3, alpha=0.4)
    # Draw a wireframe bbox for each partition — all black
    for i, p in enumerate(partitions):
        pn = norm_part(p['points'])
        mn, mx = pn.min(axis=0), pn.max(axis=0)
        # 12 edges of the bounding box
        corners = np.array([[mn[0], mn[1], mn[2]],
                             [mx[0], mn[1], mn[2]],
                             [mx[0], mx[1], mn[2]],
                             [mn[0], mx[1], mn[2]],
                             [mn[0], mn[1], mx[2]],
                             [mx[0], mn[1], mx[2]],
                             [mx[0], mx[1], mx[2]],
                             [mn[0], mx[1], mx[2]]])
        edges = [(0,1),(1,2),(2,3),(3,0),  # bottom face
                 (4,5),(5,6),(6,7),(7,4),  # top face
                 (0,4),(1,5),(2,6),(3,7)]  # verticals
        for a, b in edges:
            ax1.plot([corners[a,0], corners[b,0]],
                     [corners[a,1], corners[b,1]],
                     [corners[a,2], corners[b,2]],
                     color='black', linewidth=0.8, alpha=0.85)
    set_ax_limits(ax1, dmg_norm)
    ax1.set_axis_off()
    ax1.view_init(elev=elev, azim=azim)
    if show_labels:
        ax1.set_title(f"Partition grid\n({len(partitions)} damaged)", fontsize=label_size, pad=2)

    # ── Panels 2..N: Individual partitions (standalone) ──────────────────
    for i, p in enumerate(highlight):
        ax = fig.add_subplot(1, total_panels, 3 + i, projection='3d')

        # Only the partition itself: blue=intact, red=damaged
        pn = norm_part(p['points'])
        pc = label_colors(p['labels'])
        scatter_pcd(ax, pn, pc, size=6.0, alpha=0.95)

        set_ax_limits(ax, pn, margin=0.1)
        ax.set_axis_off()
        ax.view_init(elev=elev, azim=azim)
        if show_labels:
            n_dmg = int((p['labels'] == 1).sum())
            ax.set_title(f"Part {p['part']}\n{n_dmg} dmg pts", fontsize=label_size, pad=2)

    plt.tight_layout(pad=0.3)
    out = f"dmsurface_partition_{surface_id}_v{version}_{total_panels}panels.png"
    fig.savefig(out, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved: {out}")

    # ── Open3D interactive viewers ─────────────────────────────────────────
    if use_open3d:
        _open3d_partition_viewers(
            surface_id, version, dmg_pts, dmg_lbl, partitions, highlight
        )


def _make_o3d_pcd(points, labels):
    """Build an Open3D PointCloud: grey=intact(0), red=damaged(1)."""
    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    colors = np.where(labels[:, None] == 0,
                      np.array([[0.55, 0.55, 0.55]]),
                      np.array([[0.9,  0.2,  0.2]]))
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def _make_o3d_bbox(points, color=(0, 0, 0)):
    """Build an Open3D AxisAlignedBoundingBox from a point array."""
    import open3d as o3d
    mn = points.min(axis=0)
    mx = points.max(axis=0)
    bbox = o3d.geometry.AxisAlignedBoundingBox(
        min_bound=mn.astype(np.float64),
        max_bound=mx.astype(np.float64)
    )
    bbox.color = color
    return bbox


def _open3d_partition_viewers(surface_id, version, dmg_pts, dmg_lbl,
                               partitions, highlight):
    try:
        import open3d as o3d
    except ImportError:
        print("  open3d not installed — skipping interactive viewers.")
        print("  Install with: pip install open3d")
        return

    rng = np.random.default_rng(42)

    # Subsample full surface for Open3D (max 50k)
    max_pts = 50_000
    if len(dmg_pts) > max_pts:
        idx = rng.choice(len(dmg_pts), max_pts, replace=False)
        vis_pts = dmg_pts[idx]
        vis_lbl = dmg_lbl[idx]
    else:
        vis_pts, vis_lbl = dmg_pts, dmg_lbl

    # ── Window 1: Full damaged surface ────────────────────────────────────
    print("  [Open3D] Window 1/3: Full damaged surface (close to continue)")
    pcd_full = _make_o3d_pcd(vis_pts, vis_lbl)
    o3d.visualization.draw_geometries(
        [pcd_full],
        window_name=f"Damaged surface {surface_id} v{version}",
        width=900, height=700
    )

    # ── Window 2: Surface + all partition bboxes ──────────────────────────
    print("  [Open3D] Window 2/3: Surface + partition bboxes (close to continue)")
    geoms = [pcd_full]
    for p in partitions:
        geoms.append(_make_o3d_bbox(p['points'], color=(0, 0, 0)))
    o3d.visualization.draw_geometries(
        geoms,
        window_name=f"Partition grid {surface_id} v{version} ({len(partitions)} partitions)",
        width=900, height=700
    )

    # ── Windows 3..N: Individual partitions ───────────────────────────────
    for i, p in enumerate(highlight):
        print(f"  [Open3D] Partition {i+1}/{len(highlight)}: part {p['part']} (close to continue)")
        pcd_part = _make_o3d_pcd(p['points'], p['labels'])
        n_dmg = int((p['labels'] == 1).sum())
        o3d.visualization.draw_geometries(
            [pcd_part],
            window_name=f"Partition {p['part']} — {n_dmg} damaged pts",
            width=700, height=600
        )


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Visualize DMSurface dataset (grid or partition mode)",
        formatter_class=argparse.RawTextHelpFormatter
    )

    # ── Grid options ──────────────────────────────────────────────────────
    p.add_argument('--cols',       type=int,   default=5,
                   help='Columns in the grid (default: 5)')
    p.add_argument('--cell_size',  type=int,   default=300,
                   help='Cell size in pixels (default: 300)')
    p.add_argument('--dpi',        type=int,   default=100,
                   help='Output DPI (default: 100)')
    p.add_argument('--no_labels',  action='store_true',
                   help='Hide surface labels')
    p.add_argument('--label_size', type=int,   default=8,
                   help='Label font size (default: 8)')
    p.add_argument('--mesh_scale', type=float, default=1.0,
                   help='Scale mesh inside cell, <1 adds padding (default: 1.0)')
    p.add_argument('--elev',       type=float, default=25,
                   help='Camera elevation angle (default: 25)')
    p.add_argument('--azim',       type=float, default=-60,
                   help='Camera azimuth angle (default: -60)')

    # ── Partition mode ────────────────────────────────────────────────────
    p.add_argument('--partition',       action='store_true',
                   help='Enable partition visualization mode')
    p.add_argument('--surface',         type=str, default='001',
                   help='Surface ID for partition mode (default: 001)')
    p.add_argument('--version',         type=str, default='001',
                   help='Damage version for partition mode (default: 001)')
    p.add_argument('--num_partitions',  type=int, default=4,
                   help='Number of individual partitions to show (default: 4)')
    p.add_argument('--open3d',          action='store_true',
                   help='Open interactive Open3D viewers after saving the image')

    args = p.parse_args()
    show_labels = not args.no_labels

    if args.partition:
        render_partition_mode(
            surface_id=args.surface.zfill(3),
            version=args.version.zfill(3),
            num_partitions=args.num_partitions,
            cell_px=args.cell_size,
            dpi=args.dpi,
            show_labels=show_labels,
            label_size=args.label_size,
            elev=args.elev,
            azim=args.azim,
            use_open3d=args.open3d,
        )
    else:
        render_grid(
            cols=args.cols,
            cell_px=args.cell_size,
            dpi=args.dpi,
            show_labels=show_labels,
            label_size=args.label_size,
            mesh_scale=args.mesh_scale,
            elev=args.elev,
            azim=args.azim,
        )


if __name__ == '__main__':
    main()
