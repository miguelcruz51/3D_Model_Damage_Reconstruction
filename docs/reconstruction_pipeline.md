# Reconstruction Pipeline and Reproducibility

This document describes the end-to-end reconstruction pipeline, the dataset
layout and label conventions, and the reproducibility details (seeds, dataset
preparation, training config, checkpoint requirements, and known limitations).
It also records the behavioral changes made during reorganization and the known
broken legacy code.

## Pipeline

```
Nominal CAD surface (STL)
  → uniform point sampling + normal estimation
  → irregular dent generation (Gaussian depth profile, surface roughness,
    irregular angular boundary)
  → damaged full cloud + per-point labels {0 = intact, 1 = damaged}
  → spatial grid partitioning; each cell resampled to num_points (2048)
    producing paired (damaged partition, GT partition) with shared indexing
  → unit-sphere normalization
      center = centroid(damaged points)
      scale  = max distance of damaged points from center
      both damaged and GT transformed by the same (center, scale)
      scale_mm returned to convert metrics back to millimeters
  → reconstruction network
      SeedDispNet (M7): displace each damaged seed; intact preserved
      baselines: completion (M1, M3) or displacement (M4, M5)
  → reconstructed partition
  → geometric evaluation: cd_full, cd_damaged, l2_error, hausdorff, intact_error
  → metrics CSV + qualitative 3-panel figures
```

## Dataset Layout and Labels

On disk, each partition is stored as an `.npz` file with arrays:

- `points` `[N, 3]`, `normals` `[N, 3]`, `labels` `[N]`.
- The GT counterpart is the same filename suffixed with `_gt.npz`.

Filenames encode `{component}{version}{partition}{flag}`, where the flag `01`
marks a damaged partition and `00` marks a partition with no damage. Only
damaged partitions (`01`) are loaded for training/evaluation, because a
non-damaged partition has no distinct ground truth.

Damage-label meaning (identical in both damaged and GT clouds):

- `0` = intact point (context; preserved by the reconstruction).
- `1` = damaged point in the input, and the corresponding reconstruction target
  in the ground truth.

Split lists (`train.list`, `valid.list`, `test.list`) select which components
belong to each split.

## Dataset Preparation

Regenerate the DMSurface dataset from nominal surfaces:

```bash
python scripts/generate_dataset.py --base_path data/DMSurface
```

The generator (`src/data/generate_dataset.py`) exposes all user parameters —
`base_path`, number of `versions`, `points_per_cloud`, and the dent/partition
parameters — through `GenerationConfig` and the CLI. The dataset is large and is
excluded from version control via `.gitignore`.

## Random Seeds

Reproducibility is controlled by the `seed` key in each config. The training
config loader (`src/training/config.py`) sets the NumPy and PyTorch random seeds
from this value before training starts. The default seed in the shipped configs
is `42`. Note that full determinism on GPU additionally depends on CUDA/cuDNN
determinism settings and hardware, which are not forced here.

## Training Configuration

Each training run is fully specified by a YAML config in `configs/`. Required
keys (validated at load time): `model`, `data_root`, `num_points`, `batch_size`,
`lr`, `epochs`, `lr_decay_step`, `lr_decay_rate`, `weight_decay`, `seed`,
`log_dir`. Model/loss-specific keys include `latent_dim`, `feature_dim`,
`use_cd`, `use_squared_cd`, `no_local` (SeedDispNet), `coarse_weight` (PurePCN),
and `predict_ratio`/`coverage_weight`/`imbalance_aware` (LR-PCN). See
`configs/README.md` for the full list.

The trainer uses Adam with StepLR decay, gradient clipping (`max_norm=1.0`), CSV
logging, best-checkpoint saving, and training-curve plots.

## Checkpoint Requirements

A checkpoint (`best_model.pth`) stores `{epoch, model_state_dict, val_loss,
args}`, where `args` carries the hyperparameters (e.g. `latent_dim`,
`feature_dim`, `num_points`) needed to rebuild the model. Evaluation and
inference load `args` from the checkpoint to reconstruct the exact architecture
before loading `model_state_dict`. Evaluation and inference therefore require a
checkpoint produced by the trainer; missing checkpoints are skipped rather than
raising (evaluation) or are required arguments (inference).

## Known Reproducibility Limitations

- **Datasets and checkpoints are not committed.** The raw datasets under `data/`
  and trained checkpoints (`*.pth`) are excluded via `.gitignore` due to size.
  Reproduction requires regenerating the dataset (above) and retraining, or
  supplying externally stored checkpoints.
- **GPU non-determinism.** Only the NumPy/PyTorch seeds are set. Exact numerical
  reproduction across different GPUs/driver versions is not guaranteed.
- **Trained metric values.** The `results/` directory contains a representative
  subset of the original metric CSVs and figures with their numeric values
  preserved; it is not a full re-run of every experiment.

## Behavioral Changes During Reorganization

Per scientific-integrity requirements, code behavior is preserved; the following
changes were intentional and are non-numerical:

- **Hard-coded absolute path removed.** The original `DMSurface_gen.py` embedded
  an absolute machine path (`/home/<user>/Documents/...`) as its `base_path`.
  In `src/data/generate_dataset.py` this is replaced by a required `base_path`
  argument/config value with no default, making generation portable. The
  generation computation itself is unchanged.
- **Config validation.** The YAML loader raises a clear error when required keys
  are missing. This is a safeguard that runs before training and does not affect
  numerical results.

## Known Broken / Legacy Code

The following original artifacts do not run against the active tree and were
shelved under `experiments/` rather than included in the clean pipeline:

- **`train_label_guided.py`** (now `experiments/legacy_scripts/`) imports
  `dataset` as a callable and `metrics.metric`, neither of which exists in the
  active codebase; the script cannot run as-is.
- **`label_guided_pcn.py`** (now `experiments/legacy_models/`) is the model that
  `train_label_guided.py` targets and depends on the same missing
  `dataset.DMCVolumetric` / `metrics` symbols.

These are documented rather than deleted so no methodology is silently lost.
