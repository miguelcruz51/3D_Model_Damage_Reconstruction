# Results

This directory contains a representative subset of the research results produced
by the reconstruction pipeline. Numeric values are copied verbatim from the
source repository and are **not** altered (see Requirements 8.3, 11.1).

All files below are copied from the corrected final experiment run
`log/article_final_corr/` in the source repository.

## metrics/

Per-sample and summary reconstruction metrics (values in millimeters).

| Clean-repo file | Source file |
|---|---|
| `summary_table.csv` | `log/article_final_corr/results_mm/summary_table.csv` |
| `m1_pure_pcn_results.csv` | `log/article_final_corr/results_mm/m1_pure_pcn_results.csv` |
| `m3_lrpcn_label_guided_results.csv` | `log/article_final_corr/results_mm/m3_lrpcn_label_guided_results.csv` |
| `m4_blind_displacement_results.csv` | `log/article_final_corr/results_mm/m4_blind_displacement_results.csv` |
| `m5_flow_displacement_results.csv` | `log/article_final_corr/results_mm/m5_flow_displacement_results.csv` |
| `m7_seed_displacement_results.csv` | `log/article_final_corr/results_mm/m7_seed_displacement_results.csv` |
| `m7_ablation_no_local_results.csv` | `log/article_final_corr/results_mm/m7_ablation_no_local_results.csv` |
| `per_surface_metrics.csv` | `log/article_final_corr/per_surface_eval/per_surface_metrics.csv` |
| `damage_sensitivity.csv` | `log/article_final_corr/damage_sensitivity/damage_sensitivity.csv` |

Metric direction: for `cd_full`, `cd_damaged`, `l2_error`, `hausdorff`, and
`intact_error`, a **lower** value indicates a better reconstruction.

## figures/

Training and dataset figures.

| Clean-repo file | Source file |
|---|---|
| `m7_seed_displacement_training_curves.png` | `log/article_final_corr/m7_seed_cd/training_curves.png` |
| `m7_seed_displacement_loss_curve.png` | `log/article_final_corr/m7_seed_cd/loss_curve.png` |
| `damage_sensitivity.png` | `log/article_final_corr/damage_sensitivity/damage_sensitivity.png` |
| `dmsurface_dataset_grid.png` | `dmsurface_grid_4x5_2000x2500.png` |

## qualitative/

Representative reconstruction outputs.

| Clean-repo file | Source file |
|---|---|
| `comparison_grid_01.png` | `log/article_final_corr/comparison_grids/comparison_grid_01.png` |
| `comparison_grid_02.png` | `log/article_final_corr/comparison_grids/comparison_grid_02.png` |
| `surface_018_full_reconstruction.png` | `log/article_final_corr/surface_figures/surface_018/full_surface.png` |
| `surface_018_partition_01_reconstruction.png` | `log/article_final_corr/surface_figures/surface_018/partition_01.png` |
