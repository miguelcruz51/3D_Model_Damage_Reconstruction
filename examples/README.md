# Minimal End-to-End Example

`run_example.py` demonstrates the full SeedDisplacementNet inference path using
only modules from `src/`:

1. Load one partition via the dataset (`src.data.DMCLocalReconstruction`), or
   fall back to a synthetic partition if no dataset is available.
2. Build `SeedDisplacementNet` (`src.models`) and load weights from a checkpoint
   if one is provided, otherwise use random weights.
3. Run inference to reconstruct the damaged region.
4. Compute and print the five reconstruction metrics via
   `src.evaluation.metrics.compute_metrics`.

## Run

With a dataset and a trained checkpoint (meaningful metrics):

```bash
python examples/run_example.py \
    --data_root data/DMSurface \
    --checkpoint log/m7_seed_displacement_cd/best_model.pth
```

Without any data or checkpoint (pipeline smoke run on a synthetic partition,
random weights — the metrics are not meaningful but the flow runs end to end):

```bash
python examples/run_example.py
```

## Output

The script prints the five metrics in millimeters (lower is better):

```
Reconstruction metrics (mm; lower is better):
  n_damaged    : ...
  cd_full      : ...
  cd_damaged   : ...
  l2_error     : ...
  hausdorff    : ...
  intact_error : ...
```
