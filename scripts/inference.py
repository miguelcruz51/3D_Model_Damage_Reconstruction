#!/usr/bin/env python
"""Inference entry point.

Loads a SeedDisplacementNet checkpoint and writes qualitative 3-panel
reconstruction figures (input / ground truth / prediction) for the first
``--num_samples`` samples of the selected split to the output directory.
Example::

    python scripts/inference.py --checkpoint log/m7_seed_displacement_cd/best_model.pth \\
        --data_root data/DMSurface --split test --num_samples 5 \\
        --save_dir results/qualitative

Requirements: 7.3
"""

import argparse
import os
import sys

import numpy as np
import torch

_CLEAN_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _CLEAN_REPO_ROOT not in sys.path:
    sys.path.insert(0, _CLEAN_REPO_ROOT)

from src.data import DMCLocalReconstruction  # noqa: E402
from src.evaluation.visualize import visualize_prediction  # noqa: E402
from src.models import SeedDisplacementNet  # noqa: E402


def main(args: argparse.Namespace) -> None:
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model_args = checkpoint['args']

    print(f"Seed Displacement - Epoch: {checkpoint['epoch']}, Val Loss: {checkpoint['val_loss']:.6f}")

    model = SeedDisplacementNet(
        latent_dim=model_args.latent_dim, feature_dim=model_args.feature_dim,
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    dataset = DMCLocalReconstruction(
        data_root=args.data_root, split=args.split,
        num_points=model_args.num_points, augment=False, only_damaged=True,
    )
    print(f"Dataset [{args.split}]: {len(dataset)}")

    cd_full_list, cd_dmg_list, err_list = [], [], []

    with torch.no_grad():
        for i in range(min(args.num_samples, len(dataset))):
            damaged_pts, damaged_lbl, gt_pts, gt_lbl, _ = dataset[i]
            output, _, _ = model(
                damaged_pts.unsqueeze(0).to(device),
                damaged_lbl.unsqueeze(0).to(device),
            )
            cd_f, cd_d, err = visualize_prediction(
                damaged_pts, damaged_lbl, gt_pts, gt_lbl,
                output.squeeze(0), i, args.save_dir,
            )
            cd_full_list.append(cd_f)
            cd_dmg_list.append(cd_d)
            err_list.append(err)

    if cd_full_list:
        print("\nSUMMARY:")
        print(f"  Full CD:    {np.mean(cd_full_list):.6f} +/- {np.std(cd_full_list):.6f}")
        print(f"  Damaged CD: {np.mean(cd_dmg_list):.6f} +/- {np.std(cd_dmg_list):.6f}")
        print(f"  L2 Error:   {np.mean(err_list):.6f} +/- {np.std(err_list):.6f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SeedDispNet qualitative inference')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--data_root', type=str, default='data/DMSurface')
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--num_samples', type=int, default=5)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--save_dir', type=str, default='results/qualitative')
    main(parser.parse_args())
