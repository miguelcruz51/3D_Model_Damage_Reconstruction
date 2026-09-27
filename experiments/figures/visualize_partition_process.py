"""
Interactive Open3D visualization of the DMSurface partitioning process.

Steps shown (close each window to advance):
  1. Full damaged surface  — grey=intact, red=damaged
  2. Same surface + black wireframe bboxes for every partition in the chosen version
  3. Each selected partition individually — grey=intact, red=damaged

Usage:
    python visualize_partition_process.py
    python visualize_partition_process.py --surface 003 --version 002 --num_partitions 6
    python visualize_partition_process.py --max_pts 80000
"""

import argparse
import os
import numpy as np
import open3d as o3d

DATA_DIR = "/home/miguelcruz/Documents/D-PCN2/data/DMSurface"


# ── data helpers ─────────────────────────────────────────────────────────────

def list_surfaces():
    return sorted(
        d for d in os.listdir(DATA_DIR)
        if os.path.isdir(os.path.join(DATA_DIR, d)) and d.isdigit()
    )


def list_versions(surface_id):
    surf_dir = os.path.join(DATA_DIR, surface_id)
    versions = set()
    for f in os.listdir(surf_dir):
        if f.endswith('.npz') and not f.endswith('_gt.npz') and '_' not in f:
            # e.g. 001001.npz → version = f[3:6]
            stem = f[:-4]
            if len(stem) == 6 and stem[:3] == surface_id:
                versions.add(stem[3:6])
    return sorted(versions)


def load_surface(surface_id, version):
    path = os.path.join(DATA_DIR, surface_id, f"{surface_id}{version}.npz")
    d = np.load(path)
    return d['points'].astype(np.float64), d['labels'].astype(np.int32)


def parse_partition_stem(stem):
    """001001000001 → (surf, ver, part, flag)"""
    if len(stem) != 12:
        return None
    return stem[0:3], stem[3:6], stem[6:10], stem[10:12]


def load_partitions(surface_id, version):
    """Return list of dicts for all damaged (flag=01) partitions of this version."""
    part_dir = os.path.join(DATA_DIR, surface_id, 'partitions')
    if not os.path.exists(part_dir):
        return []
    partitions = []
    seen = set()
    for fname in sorted(os.listdir(part_dir)):
        if '_gt' in fname or not fname.endswith('.npz'):
            continue
        stem = fname[:-4]
        parsed = parse_partition_stem(stem)
        if parsed is None:
            continue
        surf, ver, part, flag = parsed
        if surf != surface_id or ver != version:
            continue
        if flag != '01':
            continue
        if part in seen:
            continue
        seen.add(part)
        gt_path = os.path.join(part_dir, fname.replace('.npz', '_gt.npz'))
        if not os.path.exists(gt_path):
            continue
        d = np.load(os.path.join(part_dir, fname))
        partitions.append({
            'part':   part,
            'points': d['points'].astype(np.float64),
            'labels': d['labels'].astype(np.int32),
        })
    return partitions


# ── Open3D helpers ────────────────────────────────────────────────────────────

def make_pcd(points, labels):
    """grey=intact(0), red=damaged(1)"""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    colors = np.where(labels[:, None] == 0,
                      np.array([[0.55, 0.55, 0.55]]),
                      np.array([[0.9,  0.1,  0.1]]))
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def make_bbox(points, color=(0.0, 0.0, 0.0)):
    mn = points.min(axis=0)
    mx = points.max(axis=0)
    bbox = o3d.geometry.AxisAlignedBoundingBox(
        min_bound=mn, max_bound=mx)
    bbox.color = color
    return bbox


def show(geoms, title):
    print(f"\n  [Open3D] {title}  — close window to continue")
    o3d.visualization.draw_geometries(geoms, window_name=title,
                                      width=1000, height=750)


# ── prompt helpers ────────────────────────────────────────────────────────────

def prompt_choice(prompt, options):
    print(f"\n{prompt}")
    for i, o in enumerate(options):
        print(f"  [{i+1}] {o}")
    while True:
        raw = input("Enter number: ").strip()
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return options[int(raw) - 1]
        print(f"  Enter a number between 1 and {len(options)}.")


# ── main flow ─────────────────────────────────────────────────────────────────

def run(surface_id, version, num_partitions, max_pts):
    rng = np.random.default_rng(42)

    # ── load full surface ─────────────────────────────────────────────────
    print(f"\nLoading surface {surface_id} v{version} ...")
    try:
        pts, labels = load_surface(surface_id, version)
    except FileNotFoundError as e:
        print(f"  ERROR: {e}")
        return

    # subsample for display
    if len(pts) > max_pts:
        intact_idx  = np.where(labels == 0)[0]
        damaged_idx = np.where(labels == 1)[0]
        ratio  = len(damaged_idx) / max(len(pts), 1)
        n_dmg  = min(len(damaged_idx), int(max_pts * ratio) + 1)
        n_int  = min(len(intact_idx),  max_pts - n_dmg)
        sel    = np.concatenate([
            rng.choice(intact_idx,  n_int, replace=False),
            rng.choice(damaged_idx, n_dmg, replace=False),
        ])
        pts_vis, lbl_vis = pts[sel], labels[sel]
    else:
        pts_vis, lbl_vis = pts, labels

    n_dmg_total = int((labels == 1).sum())
    print(f"  {len(pts)} pts total  |  {n_dmg_total} damaged  "
          f"|  displaying {len(pts_vis)} pts")

    # ── load partitions ───────────────────────────────────────────────────
    print("  Loading partitions ...")
    partitions = load_partitions(surface_id, version)
    if not partitions:
        print("  No damaged partitions found for this version.")
        return
    print(f"  Found {len(partitions)} damaged partitions")

    highlight = partitions[:num_partitions]

    # ═══════════════════════════════════════════════════════════════════════
    # Window 1 — full damaged surface
    # ═══════════════════════════════════════════════════════════════════════
    pcd_full = make_pcd(pts_vis, lbl_vis)
    show([pcd_full],
         f"Surface {surface_id} v{version} — grey=intact  red=damaged")

    # ═══════════════════════════════════════════════════════════════════════
    # Window 2 — surface + partition bboxes
    # ═══════════════════════════════════════════════════════════════════════
    geoms = [pcd_full]
    for p in partitions:
        geoms.append(make_bbox(p['points'], color=(0.0, 0.0, 0.0)))
    show(geoms,
         f"Surface {surface_id} v{version} — partition grid "
         f"({len(partitions)} damaged partitions)")

    # ═══════════════════════════════════════════════════════════════════════
    # Windows 3..N — individual partitions
    # ═══════════════════════════════════════════════════════════════════════
    for i, p in enumerate(highlight):
        n_d = int((p['labels'] == 1).sum())
        pcd_part = make_pcd(p['points'], p['labels'])
        show([pcd_part],
             f"Partition {p['part']}  ({i+1}/{len(highlight)})  "
             f"— {n_d} damaged pts")

    print("\nDone.")


def main():
    p = argparse.ArgumentParser(
        description="Interactive Open3D viewer for the DMSurface partitioning process"
    )
    p.add_argument('--surface',         type=str, default=None,
                   help='Surface ID (e.g. 001). Prompted if omitted.')
    p.add_argument('--version',         type=str, default=None,
                   help='Damage version (e.g. 001). Prompted if omitted.')
    p.add_argument('--num_partitions',  type=int, default=4,
                   help='Number of individual partitions to show (default: 4)')
    p.add_argument('--max_pts',         type=int, default=50000,
                   help='Max pts for full-surface display (default: 50000)')
    args = p.parse_args()

    # ── interactive prompts if not provided via CLI ───────────────────────
    surfaces = list_surfaces()
    if not surfaces:
        print(f"No surfaces found in {DATA_DIR}")
        return

    surface_id = args.surface.zfill(3) if args.surface else \
                 prompt_choice("Select surface:", surfaces)

    versions = list_versions(surface_id)
    if not versions:
        print(f"No versions found for surface {surface_id}")
        return

    version = args.version.zfill(3) if args.version else \
              prompt_choice(f"Select version for surface {surface_id}:", versions)

    run(surface_id, version, args.num_partitions, args.max_pts)


if __name__ == '__main__':
    main()
