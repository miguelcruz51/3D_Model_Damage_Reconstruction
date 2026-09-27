#!/usr/bin/env python
"""Training entry point.

Selects the model and hyperparameters from a YAML config file and runs the
shared trainer. Example::

    python scripts/train.py --config configs/seeddispnet.yaml

The config's ``model`` key chooses SeedDispNet (proposed) or one of the
baselines; all hyperparameters, dataset path, loss flags, seed, and log
directory come from the config file (Req 6.1, 6.3, 7.1).
"""

import argparse
import os
import sys

# Ensure the clean_repo root is importable so ``src.*`` resolves when this
# script is run directly.
_CLEAN_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _CLEAN_REPO_ROOT not in sys.path:
    sys.path.insert(0, _CLEAN_REPO_ROOT)

from src.training import load_config, train  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a reconstruction model from a YAML config")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to a YAML config file (see configs/).",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    checkpoint_path = train(config)
    print(f"Best checkpoint written to: {checkpoint_path}")


if __name__ == "__main__":
    main()
