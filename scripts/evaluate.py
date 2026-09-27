#!/usr/bin/env python
"""Evaluation entry point.

Loads one or more checkpoints, runs each method over the test set, writes
per-sample metric CSVs and a summary table to the output directory. Methods
whose checkpoint is not provided (or missing on disk) are skipped. Example::

    python scripts/evaluate.py --data_root data/DMSurface \\
        --output_dir results/metrics \\
        --m7_checkpoint log/m7_seed_displacement_cd/best_model.pth

Requirements: 7.2, 7.4
"""

import argparse
import csv
import os
import sys

import torch

_CLEAN_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _CLEAN_REPO_ROOT not in sys.path:
    sys.path.insert(0, _CLEAN_REPO_ROOT)

from src.data import DMCLocalReconstruction  # noqa: E402
from src.evaluation.evaluate import (  # noqa: E402
    evaluate_displacement,
    evaluate_lrpcn,
    evaluate_pure_pcn,
    evaluate_seed_displacement,
    save_results,
    summarize,
)


def main(args: argparse.Namespace) -> None:
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    print("Loading test dataset...")
    dataset = DMCLocalReconstruction(
        data_root=args.data_root, split='test',
        num_points=2048, augment=False, only_damaged=True,
    )
    dataset_erase = DMCLocalReconstruction(
        data_root=args.data_root, split='test',
        num_points=2048, augment=False, only_damaged=True, erase_damaged=True,
    )

    summary_rows = []

    def _exists(path):
        return path and os.path.exists(path)

    if _exists(args.m1_checkpoint):
        print("\nEvaluating M1: Pure PCN...")
        results, params, t = evaluate_pure_pcn(args.m1_checkpoint, dataset_erase, device)
        save_results(results, 'm1_pure_pcn', output_dir)
        summary_rows.append(summarize(results, 'M1: Pure PCN', params, t))

    if _exists(args.m3_checkpoint):
        print("\nEvaluating M3: LR-PCN Label-Guided...")
        results, params, t = evaluate_lrpcn(args.m3_checkpoint, dataset, device)
        save_results(results, 'm3_lrpcn_label_guided', output_dir)
        summary_rows.append(summarize(results, 'M3: LR-PCN Label-Guided', params, t))

    if _exists(args.m4_checkpoint):
        print("\nEvaluating M4: BlindDisplacementNet...")
        results, params, t = evaluate_displacement(args.m4_checkpoint, dataset, device)
        save_results(results, 'm4_blind_displacement', output_dir)
        summary_rows.append(summarize(results, 'M4: BlindDisplacement (PointCleanNet)', params, t))

    if _exists(args.m5_checkpoint):
        print("\nEvaluating M5: FlowDisplacementNet...")
        results, params, t = evaluate_displacement(args.m5_checkpoint, dataset, device)
        save_results(results, 'm5_flow_displacement', output_dir)
        summary_rows.append(summarize(results, 'M5: FlowDisplacement (FlowNet3D)', params, t))

    if _exists(args.m7_checkpoint):
        print("\nEvaluating M7: SeedDisplacementNet...")
        results, params, t = evaluate_seed_displacement(args.m7_checkpoint, dataset, device)
        save_results(results, 'm7_seed_displacement', output_dir)
        summary_rows.append(summarize(results, 'M7: SeedDisplacement (Ours)', params, t))

    if _exists(args.m7_ablation_checkpoint):
        print("\nEvaluating M7 Ablation: No local features...")
        results, params, t = evaluate_seed_displacement(args.m7_ablation_checkpoint, dataset, device)
        save_results(results, 'm7_ablation_no_local', output_dir)
        summary_rows.append(summarize(results, 'M7 Ablation: No Local Features', params, t))

    if summary_rows:
        summary_path = os.path.join(output_dir, 'summary_table.csv')
        with open(summary_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"\n\nSummary table saved to: {summary_path}")

    print("\nEvaluation complete!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate reconstruction methods on the test set')
    parser.add_argument('--data_root', type=str, default='data/DMSurface')
    parser.add_argument('--output_dir', type=str, default='results/metrics')
    parser.add_argument('--gpu', type=int, default=0)

    parser.add_argument('--m1_checkpoint', type=str, default=None)
    parser.add_argument('--m3_checkpoint', type=str, default=None)
    parser.add_argument('--m4_checkpoint', type=str, default=None)
    parser.add_argument('--m5_checkpoint', type=str, default=None)
    parser.add_argument('--m7_checkpoint', type=str, default=None)
    parser.add_argument('--m7_ablation_checkpoint', type=str, default=None)

    main(parser.parse_args())
