#!/bin/bash
# ============================================================
# LEGACY ORCHESTRATION SCRIPT (historical — not the main pipeline)
# ------------------------------------------------------------
# Invokes the original repository's root-level entry points. In the clean repo,
# use the config-driven entry points instead:
#   Train:     python scripts/train.py --config configs/<name>.yaml
#   Evaluate:  python scripts/evaluate.py --data_root ... --output_dir ...
# Kept to document the exact original "correspondence fix" retraining run.
# ============================================================
# Retrain after correspondence fix (dataset resampling bug)
#
# Only retrains methods affected by the fix:
# - M4, M5: use per-point L2 loss (index-dependent)
# - M7 variants with L2 loss (M7b, M7d)
# - M7 full ablation matrix for completeness
#
# M1 and M3 use CD loss and are NOT affected — reuse old checkpoints.
# ============================================================

DATA=data/DMSurface
BASE_LOG=log/article_final_corr

EPOCHS=500
BATCH=24
POINTS=2048

DISP_LR=0.001
DISP_DECAY_STEP=80
DISP_DECAY_RATE=0.7
FEAT_DIM=256

echo "============================================================"
echo "RETRAIN (correspondence fix)"
echo "Dataset:    $DATA"
echo "Output:     $BASE_LOG"
echo "============================================================"

# ============================================================
# M1: Pure PCN
# ============================================================

echo ""
echo "=== M1: Pure PCN ==="
python train_pure_pcn.py \
    --data_root $DATA \
    --log_dir ${BASE_LOG}/m1_pure_pcn \
    --epochs $EPOCHS --batch_size $BATCH \
    --lr 0.0005 --lr_decay_step 80 --lr_decay_rate 0.7 \
    --num_points $POINTS

# ============================================================
# M3: LR-PCN Label-Guided
# ============================================================

echo ""
echo "=== M3: LR-PCN Label-Guided ==="
python train_local_reconstruction.py \
    --data_root $DATA \
    --log_dir ${BASE_LOG}/m3_lrpcn_label_guided \
    --epochs $EPOCHS --batch_size $BATCH \
    --lr 0.0005 --lr_decay_step 80 --lr_decay_rate 0.7 \
    --coverage_weight 0.5 --imbalance_aware \
    --num_points $POINTS

# ============================================================
# BASELINES (affected by fix)
# ============================================================

echo ""
echo "=== M4: BlindDisplacementNet ==="
python train_displacement.py \
    --model blind \
    --data_root $DATA \
    --log_dir ${BASE_LOG}/m4_blind_displacement \
    --epochs $EPOCHS --batch_size $BATCH \
    --lr $DISP_LR --lr_decay_step $DISP_DECAY_STEP --lr_decay_rate $DISP_DECAY_RATE \
    --feature_dim $FEAT_DIM --num_points $POINTS

echo ""
echo "=== M5: FlowDisplacementNet ==="
python train_displacement.py \
    --model flow \
    --data_root $DATA \
    --log_dir ${BASE_LOG}/m5_flow_displacement \
    --epochs $EPOCHS --batch_size $BATCH \
    --lr $DISP_LR --lr_decay_step $DISP_DECAY_STEP --lr_decay_rate $DISP_DECAY_RATE \
    --feature_dim $FEAT_DIM --num_points $POINTS

# ============================================================
# M7: SeedDisplacement — 3 losses × 2 architectures = 6 variants
# ============================================================

echo ""
echo "=== M7: SeedDisplacement (CD L1) ==="
python train_seed_displacement.py \
    --data_root $DATA \
    --log_dir ${BASE_LOG}/m7_seed_cd \
    --epochs $EPOCHS --batch_size $BATCH \
    --lr $DISP_LR --lr_decay_step $DISP_DECAY_STEP --lr_decay_rate $DISP_DECAY_RATE \
    --latent_dim 512 --feature_dim $FEAT_DIM --num_points $POINTS \
    --use_cd

echo ""
echo "=== M7a: SeedDisplacement (CD L1, No Local) ==="
python train_seed_displacement.py \
    --data_root $DATA \
    --log_dir ${BASE_LOG}/m7a_seed_cd_no_local \
    --epochs $EPOCHS --batch_size $BATCH \
    --lr $DISP_LR --lr_decay_step $DISP_DECAY_STEP --lr_decay_rate $DISP_DECAY_RATE \
    --latent_dim 512 --feature_dim $FEAT_DIM --num_points $POINTS \
    --use_cd --no_local

echo ""
echo "=== M7b: SeedDisplacement (L2) ==="
python train_seed_displacement.py \
    --data_root $DATA \
    --log_dir ${BASE_LOG}/m7b_seed_l2 \
    --epochs $EPOCHS --batch_size $BATCH \
    --lr $DISP_LR --lr_decay_step $DISP_DECAY_STEP --lr_decay_rate $DISP_DECAY_RATE \
    --latent_dim 512 --feature_dim $FEAT_DIM --num_points $POINTS

echo ""
echo "=== M7c: SeedDisplacement (Squared CD) ==="
python train_seed_displacement.py \
    --data_root $DATA \
    --log_dir ${BASE_LOG}/m7c_seed_cd_squared \
    --epochs $EPOCHS --batch_size $BATCH \
    --lr $DISP_LR --lr_decay_step $DISP_DECAY_STEP --lr_decay_rate $DISP_DECAY_RATE \
    --latent_dim 512 --feature_dim $FEAT_DIM --num_points $POINTS \
    --use_cd --use_squared_cd

echo ""
echo "=== M7d: SeedDisplacement (L2, No Local) ==="
python train_seed_displacement.py \
    --data_root $DATA \
    --log_dir ${BASE_LOG}/m7d_seed_l2_no_local \
    --epochs $EPOCHS --batch_size $BATCH \
    --lr $DISP_LR --lr_decay_step $DISP_DECAY_STEP --lr_decay_rate $DISP_DECAY_RATE \
    --latent_dim 512 --feature_dim $FEAT_DIM --num_points $POINTS \
    --no_local

echo ""
echo "=== M7e: SeedDisplacement (Squared CD, No Local) ==="
python train_seed_displacement.py \
    --data_root $DATA \
    --log_dir ${BASE_LOG}/m7e_seed_cd_squared_no_local \
    --epochs $EPOCHS --batch_size $BATCH \
    --lr $DISP_LR --lr_decay_step $DISP_DECAY_STEP --lr_decay_rate $DISP_DECAY_RATE \
    --latent_dim 512 --feature_dim $FEAT_DIM --num_points $POINTS \
    --use_cd --use_squared_cd --no_local

# ============================================================
# EVALUATION (reuse M1/M3 from previous run)
# ============================================================
echo ""
echo "=== Full evaluation ==="

python evaluate_all.py \
    --data_root $DATA \
    --output_dir ${BASE_LOG}/results_mm \
    --m1_checkpoint ${BASE_LOG}/m1_pure_pcn/best_model.pth \
    --m3_checkpoint ${BASE_LOG}/m3_lrpcn_label_guided/best_model.pth \
    --m4_checkpoint ${BASE_LOG}/m4_blind_displacement/best_model.pth \
    --m5_checkpoint ${BASE_LOG}/m5_flow_displacement/best_model.pth \
    --m7_checkpoint ${BASE_LOG}/m7_seed_cd/best_model.pth \
    --m7_ablation_checkpoint ${BASE_LOG}/m7a_seed_cd_no_local/best_model.pth \
    --m7b_checkpoint ${BASE_LOG}/m7b_seed_l2/best_model.pth \
    --m7c_checkpoint ${BASE_LOG}/m7c_seed_cd_squared/best_model.pth \
    --m7d_checkpoint ${BASE_LOG}/m7d_seed_l2_no_local/best_model.pth \
    --m7e_checkpoint ${BASE_LOG}/m7e_seed_cd_squared_no_local/best_model.pth

echo ""
echo "============================================================"
echo "DONE — Results: ${BASE_LOG}/results_mm/summary_table.csv"
echo "============================================================"
