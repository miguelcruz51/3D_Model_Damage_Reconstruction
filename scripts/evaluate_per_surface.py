#!/usr/bin/env python
"""Per-surface evaluation entry point.

Evaluates the available methods on individual test surfaces and writes a
per-surface metrics CSV. Missing checkpoints are skipped. Example::

    python scripts/evaluate_per_surface.py --data_root data/DMSurface \\
        --surfaces 018 019 020 --output_dir results/metrics \\
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

from src.evaluation.per_surface import run_per_surface  # noqa: E402


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
    run_per_surface(args.data_root, args.surfaces, checkpoints, args.output_dir, device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Per-surface evaluation')
    parser.add_argument('--data_root', type=str, default='data/DMSurface')
    parser.add_argument('--surfaces', nargs='+', default=['018', '019', '020'])
    parser.add_argument('--output_dir', type=str, default='results/metrics/per_surface')
    parser.add_argument('--gpu', type=int, default=0)

    parser.add_argument('--m1_checkpoint', type=str, default=None)
    parser.add_argument('--m3_checkpoint', type=str, default=None)
    parser.add_argument('--m4_checkpoint', type=str, default=None)
    parser.add_argument('--m5_checkpoint', type=str, default=None)
    parser.add_argument('--m7_checkpoint', type=str, default=None)

    main(parser.parse_args())
