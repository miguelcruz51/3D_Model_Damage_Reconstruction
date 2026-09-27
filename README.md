# SeedDisplacementNet: Nominal Geometry Reconstruction from Damaged Point Clouds

Deep-learning reconstruction of the original (nominal) 3D geometry of damaged
surfaces, from labeled point-cloud partitions. The proposed method,
**SeedDisplacementNet (M7)**, treats each damaged point as a *seed* and predicts
a per-seed displacement that moves it to its reconstructed nominal position,
while intact points are preserved unchanged.

## Overview

Physical components (for example metallic parts) accumulate localized surface
damage such as dents and cavities. Recovering the *nominal* (undamaged) geometry
of the damaged region is a prerequisite for automated repair planning. This
project frames that recovery as a point-cloud reconstruction problem and
compares a family of learning-based methods, culminating in SeedDisplacementNet.

## Problem

- **Input:** a damaged partition of `N` points, each with `(x, y, z)`
  coordinates and a binary `Damage_Label` (`0` = intact, `1` = damaged).
- **Output:** a reconstructed partition of `N` points in which the damaged
  points are moved to their nominal positions and the intact points are
  preserved.
- **Ground truth (GT):** the nominal partition, index-aligned with the input,
  where a label of `1` marks the reconstruction target.

Input and GT share point indexing: point `i` in the damaged cloud corresponds to
point `i` in the GT cloud (`Point_To_Point_Correspondence`), which makes the
per-point objective well defined.

## Reconstruction Pipeline

```
Nominal CAD surface (STL)
  → uniform point sampling + normals
  → irregular dent generation (Gaussian profile + roughness + boundary noise)
  → damaged full cloud + labels {0,1}
  → spatial grid partitioning, resample to 2048 points  (paired damaged / GT)
  → unit-sphere normalization (center on damaged centroid, scale by max distance)
  → reconstruction network (SeedDispNet or a baseline)
  → reconstructed partition (intact preserved, damaged displaced)
  → geometric evaluation (5 metrics) + qualitative figures
```

See `docs/reconstruction_pipeline.md` for the detailed pipeline and
`docs/methodology.md` for the full method evolution (M1 → M7).

## Method: SeedDisplacementNet (M7)

- **Input representation:** the full partition plus labels, `(B, N, 4)` =
  `[x, y, z, label]`.
- **Output representation:** the full partition, `(B, N, 3)`; intact points are
  copied through unchanged and each damaged seed is replaced by
  `seed + predicted_displacement`.
- **Process:**
  1. A global PointNet encoder produces a 512-dim global feature for the cloud.
  2. A local per-point PointNet encoder produces a 256-dim feature per point.
  3. For each damaged point (seed), the concatenation
     `[seed_xyz(3) + local_feat(256) + global_feat(512)] = 771` is fed to an MLP
     `[771 → 512 → 256 → 128 → 3]` predicting a displacement `(dx, dy, dz)`.
  4. The reconstructed position is `seed + displacement`.

This produces exactly `N_damaged` reconstructed points (no fixed output-size
waste), keeps a 1-to-1 correspondence with the GT targets, and guarantees intact
preservation. An ablation, `SeedDisplacementNetNoLocal`, drops the local
per-point features.

## Dataset

The primary dataset is **DMSurface** (surfaces `001`–`020`): synthetic surfaces
with generated dents, partitioned and resampled to 2048 points per partition.
Only damaged partitions (filename flag `01`) are used for training, since
non-damaged partitions have no distinct ground truth.

The dataset itself is large and is not committed (see `.gitignore`). Regenerate
it with:

```bash
python scripts/generate_dataset.py --base_path data/DMSurface
```

See `docs/reconstruction_pipeline.md` for the dataset layout and label meaning.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`torch` runs on CPU by default and falls back to CPU when CUDA is unavailable.
For GPU acceleration, install a CUDA-enabled `torch` build matching your toolkit
(see comments in `requirements.txt`). `open3d` is only needed for dataset
generation.

## Usage

Train (the config's `model` key selects SeedDispNet or a baseline):

```bash
python scripts/train.py --config configs/seeddispnet.yaml
```

Evaluate a checkpoint on the test set (writes per-sample and summary metric
CSVs):

```bash
python scripts/evaluate.py --data_root data/DMSurface \
    --output_dir results/metrics \
    --m7_checkpoint log/m7_seed_displacement_cd/best_model.pth
```

Run qualitative inference (writes 3-panel input/GT/prediction figures):

```bash
python scripts/inference.py --checkpoint log/m7_seed_displacement_cd/best_model.pth \
    --data_root data/DMSurface --split test --num_samples 5 \
    --save_dir results/qualitative
```

A minimal end-to-end walkthrough on a single partition is in `examples/`.

## Results

Representative results are stored under `results/`:

- `results/metrics/` — summary and per-sample metric CSVs.
- `results/figures/` — pipeline and training-curve figures.
- `results/qualitative/` — representative reconstruction images.

All five reconstruction metrics (full-cloud Chamfer distance, damaged-region
Chamfer distance, per-point L2 error, Hausdorff distance, intact-preservation
error) are reported in millimeters, and for every one of them a **lower** value
indicates a better reconstruction. See `docs/methodology.md` for the method
comparison and reported loss values.

## Repository Structure

```
clean_repo/
├── README.md, LICENSE, requirements.txt, .gitignore
├── configs/        # one YAML per training run (model + hyperparameters)
├── src/
│   ├── data/       # paired dataset loader + DMSurface generation
│   ├── models/     # SeedDispNet (+ NoLocal ablation) and baselines
│   ├── losses/     # Chamfer (L1 / squared) + per-model losses
│   ├── training/   # shared trainer + YAML config loader
│   ├── evaluation/ # 5 metrics, per-model evaluators, visualization
│   └── utils/      # point-cloud helpers
├── scripts/        # train.py, evaluate.py, inference.py, generate_dataset.py, ...
├── experiments/    # legacy/alternative models, scripts, figures, article drafts
├── results/        # figures / metrics / qualitative
├── docs/           # methodology, methods comparison, pipeline
├── examples/       # minimal end-to-end example
└── tests/          # unit + property-based tests
```

## Research Context

This reconstruction module is one stage of a broader automated repair pipeline
for **metallic components**: a damaged part is scanned to a point cloud, damage
is localized, the *nominal* geometry of each damaged region is reconstructed
(this project), and the difference between damaged and nominal geometry drives
downstream repair (for example additive material deposition and machining).
Accurate, intact-preserving reconstruction of only the damaged region is the
value this method adds to that pipeline.

## Files to Inspect First

1. `src/models/seeddispnet.py` — the proposed SeedDisplacementNet.
2. `src/losses/seed_displacement_loss.py` — its training objective (L2 / CD).
3. `src/data/dataset.py` — paired loader, correspondence, normalization.
4. `src/evaluation/metrics.py` — the five reconstruction metrics.
5. `docs/methodology.md` — the M1 → M7 method evolution and motivation.

## Reproducibility

Random seeds, dataset preparation, training configuration, checkpoint
requirements, and known reproducibility limitations are documented in
`docs/reconstruction_pipeline.md`.

## Citation

If you use this work, please cite the associated thesis/article on nominal
geometry reconstruction for metallic-component repair (see
`experiments/article/`).

## License

Released under the MIT License. See `LICENSE`.
