#!/usr/bin/env python
"""Damage-size sensitivity analysis entry point.

Groups the test set into damage-percentage bins, evaluates the available
methods per bin, and writes a CSV table and bar plot. Missing checkpoints are
skipped. Example::

    python scripts/evaluate_damage_sensitivity.py --data_root data/DMSurface \\
        --output_dir results/metrics/damage_sensitivity \\
        --m7_checkpoint log/m7_seed_displacement_cd/best_model.pth

Requirements: 7.2, 7.4
"""

import argparse
import os
import sys

import torch

_CLEAN_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _CLEAN_REPO_ROOT not in sys.path:
    sys.path.insert(0, _CLEAN_REPO_ROOT)

from src.evaluation.damage_sensitivity import run_damage_sensitivity  # noqa: E402


def main(args: argparse.Namespace) -> None:
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    checkpoints = {
        'm1': args.m1_checkpoint,
        'm3': args.m3_checkpoint,
        'm4': args.m4_checkpoint,
        'm5': args.m5_checkpoint,
        'm7': args.m7_checkpoint,
    }
    run_damage_sensitivity(args.data_root, checkpoints, args.output_dir, device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Damage size sensitivity analysis')
    parser.add_argument('--data_root', type=str, default='data/DMSurface')
    parser.add_argument('--output_dir', type=str, default='results/metrics/damage_sensitivity')
    parser.add_argument('--gpu', type=int, default=0)

    parser.add_argument('--m1_checkpoint', type=str, default=None)
    parser.add_argument('--m3_checkpoint', type=str, default=None)
    parser.add_argument('--m4_checkpoint', type=str, default=None)
    parser.add_argument('--m5_checkpoint', type=str, default=None)
    parser.add_argument('--m7_checkpoint', type=str, default=None)

    main(parser.parse_args())
