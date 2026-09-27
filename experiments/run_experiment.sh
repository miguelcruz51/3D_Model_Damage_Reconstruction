#!/bin/bash
# ============================================================
# LEGACY ORCHESTRATION SCRIPT (historical — not the main pipeline)
# ------------------------------------------------------------
# Invokes the original repository's root-level entry points. In the clean repo,
# use the config-driven entry points instead:
#   Train:     python scripts/train.py --config configs/<name>.yaml
#   Evaluate:  python scripts/evaluate.py --data_root ... --output_dir ...
#   Inference: python scripts/inference.py --checkpoint ... --split ...
# Kept to document the exact original parameterized experiment invocation.
# ============================================================
# Parameterized Experiment: Run full comparison on any dataset
# ============================================================
#
# Usage:
#   ./run_experiment.sh data/DMSurface log/results_dmsurface
#   ./run_experiment.sh data/DMC log/results_dmc
#
# ============================================================

if [ $# -lt 2 ]; then
    echo "Usage: $0 <data_root> <output_dir>"
    echo "  Example: $0 data/DMSurface log/results_dmsurface"
    echo "  Example: $0 data/DMC log/results_dmc"
    exit 1
fi

DATA=$1
BASE_LOG=$2

echo "============================================================"
echo "Dataset:    $DATA"
echo "Output:     $BASE_LOG"
echo "============================================================"

EPOCHS=500
BATCH=24
POINTS=2048

PCN_LR=0.0005
PCN_DECAY_STEP=80
PCN_DECAY_RATE=0.7

DISP_LR=0.001
DISP_DECAY_STEP=80
DISP_DECAY_RATE=0.7
FEAT_DIM=256

# ============================================================
# TRAINING
# ============================================================

echo ""
echo "=== M1: Pure PCN ==="
python train_pure_pcn.py \
    --data_root $DATA \
    --log_dir ${BASE_LOG}/m1_pure_pcn \
    --epochs $EPOCHS --batch_size $BATCH \
    --lr $PCN_LR --lr_decay_step $PCN_DECAY_STEP --lr_decay_rate $PCN_DECAY_RATE \
    --num_points $POINTS

echo ""
echo "=== M3: LR-PCN Label-Guided ==="
python train_local_reconstruction.py \
    --data_root $DATA \
    --log_dir ${BASE_LOG}/m3_lrpcn_label_guided \
    --epochs $EPOCHS --batch_size $BATCH \
    --lr $PCN_LR --lr_decay_step $PCN_DECAY_STEP --lr_decay_rate $PCN_DECAY_RATE \
    --coverage_weight 0.5 --imbalance_aware \
    --num_points $POINTS

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

echo ""
echo "=== M7: SeedDisplacementNet (OURS) ==="
python train_seed_displacement.py \
    --data_root $DATA \
    --log_dir ${BASE_LOG}/m7_seed_displacement \
    --epochs $EPOCHS --batch_size $BATCH \
    --lr $DISP_LR --lr_decay_step $DISP_DECAY_STEP --lr_decay_rate $DISP_DECAY_RATE \
    --latent_dim 512 --feature_dim $FEAT_DIM --num_points $POINTS \
    --use_cd

echo ""
echo "=== M7a: SeedDisplacementNet Ablation (no local) ==="
python train_seed_displacement.py \
    --data_root $DATA \
    --log_dir ${BASE_LOG}/m7a_seed_no_local \
    --epochs $EPOCHS --batch_size $BATCH \
    --lr $DISP_LR --lr_decay_step $DISP_DECAY_STEP --lr_decay_rate $DISP_DECAY_RATE \
    --latent_dim 512 --feature_dim $FEAT_DIM --num_points $POINTS \
    --use_cd --no_local

echo ""
echo "=== M7b: SeedDisplacementNet (L2 Loss) ==="
python train_seed_displacement.py \
    --data_root $DATA \
    --log_dir ${BASE_LOG}/m7b_seed_l2 \
    --epochs $EPOCHS --batch_size $BATCH \
    --lr $DISP_LR --lr_decay_step $DISP_DECAY_STEP --lr_decay_rate $DISP_DECAY_RATE \
    --latent_dim 512 --feature_dim $FEAT_DIM --num_points $POINTS

echo ""
echo "=== M7c: SeedDisplacementNet (Squared CD) ==="
python train_seed_displacement.py \
    --data_root $DATA \
    --log_dir ${BASE_LOG}/m7c_seed_cd_squared \
    --epochs $EPOCHS --batch_size $BATCH \
    --lr $DISP_LR --lr_decay_step $DISP_DECAY_STEP --lr_decay_rate $DISP_DECAY_RATE \
    --latent_dim 512 --feature_dim $FEAT_DIM --num_points $POINTS \
    --use_cd --use_squared_cd

echo ""
echo "=== M7d: SeedDisplacementNet (L2 Loss, No Local) ==="
python train_seed_displacement.py \
    --data_root $DATA \
    --log_dir ${BASE_LOG}/m7d_seed_l2_no_local \
    --epochs $EPOCHS --batch_size $BATCH \
    --lr $DISP_LR --lr_decay_step $DISP_DECAY_STEP --lr_decay_rate $DISP_DECAY_RATE \
    --latent_dim 512 --feature_dim $FEAT_DIM --num_points $POINTS \
    --no_local

echo ""
echo "=== M7e: SeedDisplacementNet (Squared CD, No Local) ==="
python train_seed_displacement.py \
    --data_root $DATA \
    --log_dir ${BASE_LOG}/m7e_seed_cd_squared_no_local \
    --epochs $EPOCHS --batch_size $BATCH \
    --lr $DISP_LR --lr_decay_step $DISP_DECAY_STEP --lr_decay_rate $DISP_DECAY_RATE \
    --latent_dim 512 --feature_dim $FEAT_DIM --num_points $POINTS \
    --use_cd --use_squared_cd --no_local

# ============================================================
# INFERENCE IMAGES
# ============================================================
echo ""
echo "=== Generating inference images ==="

for SPLIT in train valid test; do
    python inference_pure_pcn.py \
        --checkpoint ${BASE_LOG}/m1_pure_pcn/best_model.pth \
        --data_root $DATA --split $SPLIT --num_samples 3 \
        --save_dir ${BASE_LOG}/m1_pure_pcn/inference_${SPLIT}
done

for SPLIT in train valid test; do
    python inference_local_reconstruction.py \
        --checkpoint ${BASE_LOG}/m3_lrpcn_label_guided/best_model.pth \
        --data_root $DATA --split $SPLIT --num_samples 3 \
        --save_dir ${BASE_LOG}/m3_lrpcn_label_guided/inference_${SPLIT}
done

for METHOD in m4_blind_displacement m5_flow_displacement; do
    for SPLIT in train valid test; do
        python inference_displacement.py \
            --checkpoint ${BASE_LOG}/${METHOD}/best_model.pth \
            --data_root $DATA --split $SPLIT --num_samples 3 \
            --save_dir ${BASE_LOG}/${METHOD}/inference_${SPLIT}
    done
done

for METHOD in m7_seed_displacement m7a_seed_no_local m7b_seed_l2 m7c_seed_cd_squared m7d_seed_l2_no_local m7e_seed_cd_squared_no_local; do
    for SPLIT in train valid test; do
        python inference_seed_displacement.py \
            --checkpoint ${BASE_LOG}/${METHOD}/best_model.pth \
            --data_root $DATA --split $SPLIT --num_samples 3 \
            --save_dir ${BASE_LOG}/${METHOD}/inference_${SPLIT}
    done
done

# ============================================================
# FULL TEST SET EVALUATION
# ============================================================
echo ""
echo "=== Full test set evaluation ==="

python evaluate_all.py \
    --data_root $DATA \
    --output_dir ${BASE_LOG}/results \
    --m1_checkpoint ${BASE_LOG}/m1_pure_pcn/best_model.pth \
    --m3_checkpoint ${BASE_LOG}/m3_lrpcn_label_guided/best_model.pth \
    --m4_checkpoint ${BASE_LOG}/m4_blind_displacement/best_model.pth \
    --m5_checkpoint ${BASE_LOG}/m5_flow_displacement/best_model.pth \
    --m7_checkpoint ${BASE_LOG}/m7_seed_displacement/best_model.pth \
    --m7_ablation_checkpoint ${BASE_LOG}/m7a_seed_no_local/best_model.pth \
    --m7b_checkpoint ${BASE_LOG}/m7b_seed_l2/best_model.pth \
    --m7c_checkpoint ${BASE_LOG}/m7c_seed_cd_squared/best_model.pth \
    --m7d_checkpoint ${BASE_LOG}/m7d_seed_l2_no_local/best_model.pth \
    --m7e_checkpoint ${BASE_LOG}/m7e_seed_cd_squared_no_local/best_model.pth

# ============================================================
# ARTICLE FIGURES
# ============================================================
echo ""
echo "=== Generating article figures ==="

python generate_article_figures.py \
    --data_root $DATA \
    --output_dir ${BASE_LOG}/figures \
    --num_samples 3 \
    --m1_checkpoint ${BASE_LOG}/m1_pure_pcn/best_model.pth \
    --m3_checkpoint ${BASE_LOG}/m3_lrpcn_label_guided/best_model.pth \
    --m4_checkpoint ${BASE_LOG}/m4_blind_displacement/best_model.pth \
    --m5_checkpoint ${BASE_LOG}/m5_flow_displacement/best_model.pth \
    --m7_checkpoint ${BASE_LOG}/m7_seed_displacement/best_model.pth

# ============================================================
# SAVE M7 PCDs
# ============================================================
echo ""
echo "=== Saving M7 PCDs ==="

python inference_seed_save_pcd.py \
    --checkpoint ${BASE_LOG}/m7_seed_displacement/best_model.pth \
    --data_root $DATA --split test --num_samples 10 \
    --save_dir ${BASE_LOG}/m7_seed_displacement/pcds_test

# ============================================================
echo ""
echo "============================================================"
echo "EXPERIMENT COMPLETE"
echo "============================================================"
echo ""
echo "Dataset:  $DATA"
echo "Results:  $BASE_LOG"
echo ""
echo "Key outputs:"
echo "  ${BASE_LOG}/results/summary_table.csv"
echo "  ${BASE_LOG}/figures/"
echo "  ${BASE_LOG}/*/training_curves.png"
echo "  ${BASE_LOG}/*/inference_*/"
