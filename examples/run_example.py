#!/usr/bin/env python
"""Minimal end-to-end SeedDispNet example on a single partition.

This script demonstrates the full inference path using only ``src/`` modules:

1. Load one partition from the paired dataset (or a synthetic partition if no
   dataset is available).
2. Build :class:`SeedDisplacementNet` and load weights from a checkpoint if one
   is provided, otherwise use random weights.
3. Run inference to reconstruct the damaged region.
4. Compute and print the five reconstruction metrics.

Example::

    python examples/run_example.py \\
        --data_root data/DMSurface \\
        --checkpoint log/m7_seed_displacement_cd/best_model.pth

Both arguments are optional: with no dataset a synthetic partition is generated,
and with no checkpoint random weights are used (metrics will not be meaningful
but the pipeline still runs).
"""

import argparse
import os
import sys

import numpy as np
import torch

# Make ``src.*`` importable when run directly from anywhere.
_CLEAN_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _CLEAN_REPO_ROOT not in sys.path:
    sys.path.insert(0, _CLEAN_REPO_ROOT)

from src.evaluation.metrics import compute_metrics  # noqa: E402
from src.models import SeedDisplacementNet  # noqa: E402


def _synthetic_partition(num_points: int = 2048, seed: int = 0):
    """Build a synthetic paired partition when no dataset is available.

    Returns damaged points/labels, ground-truth points/labels, and a scale (mm),
    matching the dataset's ``__getitem__`` output shapes.
    """
    rng = np.random.default_rng(seed)
    # A flat-ish nominal patch in the unit sphere.
    gt = rng.uniform(-0.7, 0.7, size=(num_points, 3)).astype(np.float32)
    gt[:, 2] *= 0.05  # roughly planar
    labels = np.zeros(num_points, dtype=np.float32)
    # Mark ~10% of points as a "dent" and push them inward along z.
    n_damaged = max(1, num_points // 10)
    idx = rng.choice(num_points, size=n_damaged, replace=False)
    labels[idx] = 1.0
    damaged = gt.copy()
    damaged[idx, 2] -= 0.2  # dent depression
    return (
        torch.from_numpy(damaged),
        torch.from_numpy(labels),
        torch.from_numpy(gt),
        torch.from_numpy(labels.copy()),
        1.0,
    )


def _load_partition(data_root: str, num_points: int):
    """Load the first damaged partition from the dataset, if available."""
    from src.data import DMCLocalReconstruction

    dataset = DMCLocalReconstruction(
        data_root=data_root, split="test",
        num_points=num_points, augment=False, only_damaged=True,
    )
    if len(dataset) == 0:
        raise RuntimeError("dataset is empty")
    return dataset[0]


def main(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_points = 2048
    if args.data_root and os.path.isdir(args.data_root):
        try:
            damaged, d_labels, gt, gt_labels, scale_mm = _load_partition(
                args.data_root, num_points
            )
            print(f"Loaded one partition from {args.data_root}")
        except Exception as exc:  # dataset missing/empty -> fall back
            print(f"Could not load dataset ({exc}); using a synthetic partition.")
            damaged, d_labels, gt, gt_labels, scale_mm = _synthetic_partition(num_points)
    else:
        print("No dataset provided; using a synthetic partition.")
        damaged, d_labels, gt, gt_labels, scale_mm = _synthetic_partition(num_points)

    # Build the model and (optionally) load a checkpoint.
    if args.checkpoint and os.path.exists(args.checkpoint):
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model_args = checkpoint["args"]
        model = SeedDisplacementNet(
            latent_dim=model_args.latent_dim, feature_dim=model_args.feature_dim,
        ).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded checkpoint: {args.checkpoint}")
    else:
        model = SeedDisplacementNet().to(device)
        print("No checkpoint provided; using random weights (metrics not meaningful).")

    model.eval()

    # Run inference on the single partition.
    with torch.no_grad():
        output, _, _ = model(
            damaged.unsqueeze(0).to(device),
            d_labels.unsqueeze(0).to(device),
        )
    pred = output.squeeze(0).cpu()

    # Compute and print the five reconstruction metrics.
    metrics = compute_metrics(
        pred, d_labels, gt, gt_labels, damaged, d_labels, scale_mm=float(scale_mm),
    )

    print("\nReconstruction metrics (mm; lower is better):")
    print(f"  n_damaged    : {metrics['n_damaged']}")
    print(f"  cd_full      : {metrics['cd_full']:.6f}")
    print(f"  cd_damaged   : {metrics['cd_damaged']:.6f}")
    print(f"  l2_error     : {metrics['l2_error']:.6f}")
    print(f"  hausdorff    : {metrics['hausdorff']:.6f}")
    print(f"  intact_error : {metrics['intact_error']:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Minimal SeedDispNet end-to-end example")
    parser.add_argument("--data_root", type=str, default=None,
                        help="Dataset root; if absent, a synthetic partition is used.")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="SeedDispNet checkpoint; if absent, random weights are used.")
    main(parser.parse_args())
