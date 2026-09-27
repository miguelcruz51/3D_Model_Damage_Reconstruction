# Training Configs

Each YAML file fully specifies one training run: which model to build, the
dataset location, all hyperparameters, loss flags, the random seed, and the
output log directory. Run a config with:

```bash
python scripts/train.py --config configs/seeddispnet.yaml
```

## Available configs

| Config | Model | Description |
|---|---|---|
| `seeddispnet.yaml` | SeedDisplacementNet (M7) | Proposed per-seed displacement method |
| `pure_pcn.yaml` | PurePCN (M1) | Full-reconstruction completion baseline |
| `lrpcn.yaml` | LocalReconstructionPCN (M3) | Label-guided local reconstruction |
| `blind_displacement.yaml` | BlindDisplacementNet (M4) | PointCleanNet-style displacement baseline |
| `flow_displacement.yaml` | FlowDisplacementNet (M5) | FlowNet3D-style displacement baseline |

## Required keys

Every config must define these keys (validated at load time):

- `model` — one of `seeddispnet`, `pure_pcn`, `lrpcn`, `blind_displacement`, `flow_displacement`
- `data_root` — path to the dataset root (e.g. `data/DMSurface`)
- `num_points` — points per partition (2048 in the paper)
- `batch_size`, `lr`, `epochs`
- `lr_decay_step`, `lr_decay_rate`, `weight_decay`
- `seed` — random seed for numpy/torch (reproducibility)
- `log_dir` — output directory for the checkpoint, CSV log, and curves

## Optional / model-specific keys

- `latent_dim`, `feature_dim` — model capacity
- `use_cd`, `use_squared_cd`, `no_local` — SeedDispNet loss/ablation flags
  (default per-point L2: `use_cd: false`)
- `coarse_weight` — PurePCN loss
- `predict_ratio`, `coverage_weight`, `imbalance_aware` — LR-PCN
- `gpu`, `num_workers` — system settings
