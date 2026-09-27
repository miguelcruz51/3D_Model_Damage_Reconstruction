"""
Interactive inference script.
Two modes:
  - partition : pick component / version / split / single partition → infer + visualize
  - component : pick component / version → infer ALL partitions, join and visualize full component

Filename format: CCCVVVPPPPPLL.npz
  CCC   = component (e.g. 001)
  VVV   = version   (e.g. 002)
  PPPPP = partition index (e.g. 00123)
  LL    = label: 00 = no damage, 01 = has damage
"""

import os
import sys
import json
import importlib
import numpy as np
import torch
import open3d as o3d
from collections import defaultdict

# ── paths ────────────────────────────────────────────────────────────────────
DATA_ROOT   = "/home/miguelcruz/Documents/3DL-Based/data/damaged_components"
SPLIT_DIR   = os.path.join(DATA_ROOT, "train_test_split")
MODEL_PATH  = "/home/miguelcruz/Documents/3DL-Based/model_experiments/PointNet2_Results/model_experiments/PointNet2_Results/pointnet2_boundary_aware/checkpoints/best_model.pth"
MODEL_NAME  = "models.pointnet2_dmg_seg_ssg"

# ── inference hyper-params ───────────────────────────────────────────────────
NPOINTS        = 1024
NUM_SUBSAMPLES = 6
POINT_THR      = 0.5
PATCH_THR      = 0.05

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── helpers ──────────────────────────────────────────────────────────────────
def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    return pc / (m + 1e-6)


def load_split(split: str):
    """Return set of relative paths like '001/001002003400.npz' for the split."""
    fname = {
        "train": "shuffled_train_file_list.json",
        "val":   "shuffled_val_file_list.json",
        "test":  "shuffled_test_file_list.json",
    }[split]
    with open(os.path.join(SPLIT_DIR, fname)) as f:
        return set(json.load(f))


def parse_filename(stem: str):
    """stem = '001002003400'  →  (comp='001', ver='002', part='03400', label='00')"""
    comp  = stem[0:3]
    ver   = stem[3:6]
    part  = stem[6:11]
    label = stem[11:13]
    return comp, ver, part, label


def list_components():
    return sorted(
        d for d in os.listdir(DATA_ROOT)
        if os.path.isdir(os.path.join(DATA_ROOT, d)) and d.isdigit()
    )


def list_versions(comp: str):
    comp_dir = os.path.join(DATA_ROOT, comp)
    versions = set()
    for f in os.listdir(comp_dir):
        if f.endswith(".npz"):
            stem = f[:-4]
            versions.add(stem[3:6])
    return sorted(versions)


def filter_partitions(comp: str, ver: str, split_files: set):
    """Return list of (filename, part_idx, label) matching comp+ver in the split."""
    prefix = f"{comp}/{comp}{ver}"
    matches = []
    for rel in split_files:
        if rel.startswith(prefix):
            fname = os.path.basename(rel)
            stem  = fname[:-4]
            _, _, part, label = parse_filename(stem)
            matches.append((fname, part, label))
    return sorted(matches, key=lambda x: x[1])


def all_partitions_for_version(comp: str, ver: str):
    """Return ALL (filename, part_idx, label) for comp+ver regardless of split."""
    comp_dir = os.path.join(DATA_ROOT, comp)
    prefix   = f"{comp}{ver}"
    matches  = []
    for f in sorted(os.listdir(comp_dir)):
        if f.endswith(".npz") and f.startswith(prefix):
            stem = f[:-4]
            _, _, part, label = parse_filename(stem)
            matches.append((f, part, label))
    return sorted(matches, key=lambda x: x[1])


def prompt_choice(prompt: str, options: list):
    print(f"\n{prompt}")
    for i, o in enumerate(options):
        print(f"  [{i+1}] {o}")
    while True:
        raw = input("Enter number: ").strip()
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return options[int(raw) - 1]
        print(f"  Please enter a number between 1 and {len(options)}.")


# ── model ────────────────────────────────────────────────────────────────────
def load_model():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    mod = importlib.import_module(MODEL_NAME)
    net = mod.get_model(normal_channel=True).to(DEVICE)
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
    net.load_state_dict(ckpt["model_state_dict"])
    net.eval()
    print(f"[INFO] Model loaded  ({DEVICE})")
    return net


# ── visualization ────────────────────────────────────────────────────────────
def visualize_gt(pts: np.ndarray, labels: np.ndarray, title: str):
    """Blue = intact (0), Red = damaged (1)."""
    colors = np.zeros((len(pts), 3))
    colors[labels == 0] = [0.2, 0.4, 0.9]   # blue
    colors[labels == 1] = [0.9, 0.1, 0.1]   # red

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    print(f"\n[VIZ] {title}  (blue=intact, red=damaged)  — close window to continue")
    o3d.visualization.draw_geometries([pcd], window_name=title)


def visualize_pred(pts: np.ndarray, point_probs: np.ndarray, title: str):
    """Blue = intact, Red = predicted damaged (thresholded)."""
    pred_labels = (point_probs > POINT_THR).astype(int)
    colors = np.zeros((len(pts), 3))
    colors[pred_labels == 0] = [0.2, 0.4, 0.9]   # blue
    colors[pred_labels == 1] = [0.9, 0.1, 0.1]   # red

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    print(f"\n[VIZ] {title}  (blue=intact, red=predicted damage)  — close window to continue")
    o3d.visualization.draw_geometries([pcd], window_name=title)


def visualize_component(all_pts, all_labels, all_probs, title_gt, title_pred):
    """Assemble all partitions and show GT + prediction for the full component."""
    pts    = np.vstack(all_pts)
    labels = np.concatenate(all_labels)
    probs  = np.concatenate(all_probs)
    visualize_gt(pts, labels, title_gt)
    visualize_pred(pts, probs, title_pred)


def visualize_component_with_partition(comp: str, ver: str, chosen_fname: str, partition_pts: np.ndarray, partition_labels: np.ndarray, title: str):
    """Full component in grey (excluding selected partition); selected partition colored."""
    all_partitions = all_partitions_for_version(comp, ver)

    bg_pts = []
    for fname, _, _ in all_partitions:
        if fname == chosen_fname:
            continue   # skip the selected partition from the grey background
        d = np.load(os.path.join(DATA_ROOT, comp, fname))
        bg_pts.append(d["points"].astype(np.float64))

    geoms = []

    # grey background (rest of component)
    if bg_pts:
        bg_pcd = o3d.geometry.PointCloud()
        bg_pcd.points = o3d.utility.Vector3dVector(np.vstack(bg_pts))
        bg_pcd.paint_uniform_color([0.6, 0.6, 0.6])
        geoms.append(bg_pcd)

    # colored partition
    part_colors = np.zeros((len(partition_pts), 3))
    part_colors[partition_labels == 0] = [0.2, 0.4, 0.9]   # blue = intact
    part_colors[partition_labels == 1] = [0.9, 0.1, 0.1]   # red  = damaged

    part_pcd = o3d.geometry.PointCloud()
    part_pcd.points = o3d.utility.Vector3dVector(partition_pts)
    part_pcd.colors = o3d.utility.Vector3dVector(part_colors)
    geoms.append(part_pcd)

    print(f"\n[VIZ] {title}  (grey=rest of component, blue=intact, green=damaged)  — close window to continue")
    o3d.visualization.draw_geometries(geoms, window_name=title)


# ── core inference (single npz) ──────────────────────────────────────────────
def infer_partition(model, npz_path: str):
    """Run inference on one npz. Returns (pts, labels, point_probs)."""
    data    = np.load(npz_path)
    pts     = data["points"].astype(np.float64)
    normals = data["normals"].astype(np.float64)
    labels  = data["labels"].astype(np.int32)

    features        = np.hstack([pts, normals])
    patch_probs     = []
    point_probs_acc = defaultdict(list)

    with torch.no_grad():
        for _ in range(NUM_SUBSAMPLES):
            feat = features.copy()
            feat[:, :3] = pc_normalize(feat[:, :3])
            choice = np.random.choice(len(feat), NPOINTS, replace=True)
            inp    = feat[choice]

            batch   = torch.tensor(inp, dtype=torch.float32).T.unsqueeze(0).to(DEVICE)
            one_hot = torch.zeros((1, 1), device=DEVICE)

            logits, _ = model(batch, one_hot)
            probs = torch.softmax(logits, dim=-1)[0, :, 1].cpu().numpy()

            patch_probs.append(probs)
            for i, idx in enumerate(choice):
                point_probs_acc[idx].append(probs[i])

    point_probs = np.zeros(len(pts))
    for idx, vals in point_probs_acc.items():
        point_probs[idx] = np.mean(vals)

    return pts, labels, point_probs


def print_metrics(pts, labels, point_probs, fname=""):
    all_probs   = point_probs
    patch_score = (all_probs > POINT_THR).mean()
    prediction  = int(patch_score >= PATCH_THR)
    preds       = (point_probs > POINT_THR).astype(np.int32)

    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())

    precision = tp / (tp + fp + 1e-6)
    recall    = tp / (tp + fn + 1e-6)
    f1        = 2 * precision * recall / (precision + recall + 1e-6)
    iou       = tp / (tp + fp + fn + 1e-6)
    accuracy  = (tp + tn) / len(labels)
    gt_ratio  = (labels == 1).mean()
    gt_label  = "DAMAGED" if labels.max() > 0 else "INTACT"

    print("\n" + "=" * 60)
    if fname:
        print(f"  File          : {fname}")
    print(f"  Points        : {len(pts)}")
    print(f"  Ground truth  : {gt_label}  (damaged={labels.sum()}, ratio={gt_ratio:.4f})")
    print(f"  Patch pred    : {'🔴 DAMAGED' if prediction else '🔵 INTACT'}  (score={patch_score:.4f})")
    print()
    print("  ── Point-level metrics (damaged class) ──")
    print(f"  Accuracy      : {accuracy:.4f}")
    print(f"  Precision     : {precision:.4f}")
    print(f"  Recall        : {recall:.4f}")
    print(f"  F1            : {f1:.4f}")
    print(f"  IoU (damaged) : {iou:.4f}")
    print(f"  Confusion     : TP={tp}  FP={fp}  FN={fn}  TN={tn}")
    print("=" * 60)


# ── mode: single partition ────────────────────────────────────────────────────
def mode_partition(model):
    components = list_components()
    comp = prompt_choice("Select component:", components)

    versions = list_versions(comp)
    if not versions:
        print(f"No versions found for component {comp}.")
        return
    ver = prompt_choice(f"Select version for component {comp}:", versions)

    split = prompt_choice("Select split:", ["train", "val", "test"])
    split_files = load_split(split)

    partitions = filter_partitions(comp, ver, split_files)
    if not partitions:
        print(f"\nNo partitions found for component={comp}, version={ver}, split={split}.")
        return

    options = [
        f"{fname}  (partition={part}, {'HAS damage' if lbl == '01' else 'no damage'})"
        for fname, part, lbl in partitions
    ]
    chosen       = prompt_choice(f"Found {len(partitions)} partition(s) — select one:", options)
    chosen_fname = chosen.split("  ")[0]
    npz_path     = os.path.join(DATA_ROOT, comp, chosen_fname)

    pts, labels, point_probs = infer_partition(model, npz_path)

    # 1. full component context with partition highlighted
    visualize_component_with_partition(comp, ver, chosen_fname, pts, labels, f"Component {comp} v{ver} — partition location")

    # 2. partition ground truth
    visualize_gt(pts, labels, f"Ground Truth — {chosen_fname}")

    # 3. partition prediction
    visualize_pred(pts, point_probs, f"Prediction — {chosen_fname}")

    print_metrics(pts, labels, point_probs, fname=chosen_fname)


# ── mode: full component ──────────────────────────────────────────────────────
def mode_component(model):
    components = list_components()
    comp = prompt_choice("Select component:", components)

    versions = list_versions(comp)
    if not versions:
        print(f"No versions found for component {comp}.")
        return
    ver = prompt_choice(f"Select version for component {comp}:", versions)

    partitions = all_partitions_for_version(comp, ver)
    if not partitions:
        print(f"\nNo partitions found for component={comp}, version={ver}.")
        return

    print(f"\n[INFO] Running inference on {len(partitions)} partitions for {comp} v{ver} ...")

    all_pts, all_labels, all_probs = [], [], []

    for i, (fname, part, lbl) in enumerate(partitions):
        npz_path = os.path.join(DATA_ROOT, comp, fname)
        pts, labels, point_probs = infer_partition(model, npz_path)
        all_pts.append(pts)
        all_labels.append(labels)
        all_probs.append(point_probs)
        status = "HAS damage" if lbl == "01" else "no damage"
        print(f"  [{i+1}/{len(partitions)}] {fname}  ({status})")

    tag = f"Component {comp} v{ver}"
    visualize_component(all_pts, all_labels, all_probs,
                        title_gt=f"Ground Truth — {tag}",
                        title_pred=f"Prediction — {tag}")

    # aggregate metrics over the full component
    pts_all    = np.vstack(all_pts)
    labels_all = np.concatenate(all_labels)
    probs_all  = np.concatenate(all_probs)
    print(f"\n  ── Aggregated metrics for {tag} ──")
    print_metrics(pts_all, labels_all, probs_all)


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Damage Segmentation — Interactive Inference")
    print("=" * 60)

    mode  = prompt_choice("Select inference mode:", ["partition", "component"])
    model = load_model()

    if mode == "partition":
        mode_partition(model)
    else:
        mode_component(model)


if __name__ == "__main__":
    main()
