#!/bin/bash
# ============================================================
# LEGACY ORCHESTRATION SCRIPT (historical — not the main pipeline)
# ------------------------------------------------------------
# This script reproduces the original article comparison run and invokes the
# original repository's root-level entry points (train_pure_pcn.py,
# train_seed_displacement.py, evaluate_all.py, inference_*.py). In the clean
# repo these have been consolidated into config-driven entry points:
#   Train:     python scripts/train.py --config configs/<name>.yaml
#   Evaluate:  python scripts/evaluate.py --data_root ... --output_dir ...
#   Inference: python scripts/inference.py --checkpoint ... --split ...
# It is kept here only to document the exact original experiment invocation.
# ============================================================
# Final Article Comparison: 5 Methods + 1 Ablation
# ============================================================
#
# M1: Pure PCN (full reconstruction baseline)
# M3: LR-PCN Label-Guided (best completion method)
# M4: BlindDisplacementNet (PointCleanNet-style)
# M5: FlowDisplacementNet (FlowNet3D-style)
# M7: SeedDisplacementNet + CD (OURS)
# M7a: SeedDisplacementNet + CD without local features (ablation)
#
# ============================================================

EPOCHS=500
BATCH=24
POINTS=2048
DATA=data/DMSurface
BASE_LOG=log/article_final

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

echo "============================================================"
echo "M1: Pure PCN (full reconstruction)"
echo "============================================================"
python train_pure_pcn.py \
    --data_root $DATA \
    --log_dir ${BASE_LOG}/m1_pure_pcn \
    --epochs $EPOCHS --batch_size $BATCH \
    --lr $PCN_LR --lr_decay_step $PCN_DECAY_STEP --lr_decay_rate $PCN_DECAY_RATE \
    --num_points $POINTS

echo ""
echo "============================================================"
echo "M3: LR-PCN Label-Guided"
echo "============================================================"
python train_local_reconstruction.py \
    --data_root $DATA \
    --log_dir ${BASE_LOG}/m3_lrpcn_label_guided \
    --epochs $EPOCHS --batch_size $BATCH \
    --lr $PCN_LR --lr_decay_step $PCN_DECAY_STEP --lr_decay_rate $PCN_DECAY_RATE \
    --coverage_weight 0.5 --imbalance_aware \
    --num_points $POINTS

echo ""
echo "============================================================"
echo "M4: BlindDisplacementNet (PointCleanNet-style)"
echo "============================================================"
python train_displacement.py \
    --model blind \
    --data_root $DATA \
    --log_dir ${BASE_LOG}/m4_blind_displacement \
    --epochs $EPOCHS --batch_size $BATCH \
    --lr $DISP_LR --lr_decay_step $DISP_DECAY_STEP --lr_decay_rate $DISP_DECAY_RATE \
    --feature_dim $FEAT_DIM --num_points $POINTS

echo ""
echo "============================================================"
echo "M5: FlowDisplacementNet (FlowNet3D-style)"
echo "============================================================"
python train_displacement.py \
    --model flow \
    --data_root $DATA \
    --log_dir ${BASE_LOG}/m5_flow_displacement \
    --epochs $EPOCHS --batch_size $BATCH \
    --lr $DISP_LR --lr_decay_step $DISP_DECAY_STEP --lr_decay_rate $DISP_DECAY_RATE \
    --feature_dim $FEAT_DIM --num_points $POINTS

echo ""
echo "============================================================"
echo "M7: SeedDisplacementNet + CD (OURS)"
echo "============================================================"
python train_seed_displacement.py \
    --data_root $DATA \
    --log_dir ${BASE_LOG}/m7_seed_displacement \
    --epochs $EPOCHS --batch_size $BATCH \
    --lr $DISP_LR --lr_decay_step $DISP_DECAY_STEP --lr_decay_rate $DISP_DECAY_RATE \
    --latent_dim 512 --feature_dim $FEAT_DIM --num_points $POINTS \
    --use_cd

echo ""
echo "============================================================"
echo "M7a: SeedDisplacementNet + CD (Ablation: no local features)"
echo "============================================================"
python train_seed_displacement.py \
    --data_root $DATA \
    --log_dir ${BASE_LOG}/m7a_seed_no_local \
    --epochs $EPOCHS --batch_size $BATCH \
    --lr $DISP_LR --lr_decay_step $DISP_DECAY_STEP --lr_decay_rate $DISP_DECAY_RATE \
    --latent_dim 512 --feature_dim $FEAT_DIM --num_points $POINTS \
    --use_cd --no_local

# ============================================================
# INFERENCE IMAGES (3 samples per split)
# ============================================================
echo ""
echo "============================================================"
echo "Generating inference images"
echo "============================================================"

# M1
for SPLIT in train valid test; do
    python inference_pure_pcn.py \
        --checkpoint ${BASE_LOG}/m1_pure_pcn/best_model.pth \
        --data_root $DATA --split $SPLIT --num_samples 3 \
        --save_dir ${BASE_LOG}/m1_pure_pcn/inference_${SPLIT}
done

# M3
for SPLIT in train valid test; do
    python inference_local_reconstruction.py \
        --checkpoint ${BASE_LOG}/m3_lrpcn_label_guided/best_model.pth \
        --data_root $DATA --split $SPLIT --num_samples 3 \
        --save_dir ${BASE_LOG}/m3_lrpcn_label_guided/inference_${SPLIT}
done

# M4, M5
for METHOD in m4_blind_displacement m5_flow_displacement; do
    for SPLIT in train valid test; do
        python inference_displacement.py \
            --checkpoint ${BASE_LOG}/${METHOD}/best_model.pth \
            --data_root $DATA --split $SPLIT --num_samples 3 \
            --save_dir ${BASE_LOG}/${METHOD}/inference_${SPLIT}
    done
done

# M7, M7a
for METHOD in m7_seed_displacement m7a_seed_no_local; do
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
echo "============================================================"
echo "Full test set evaluation (all metrics)"
echo "============================================================"

python evaluate_all.py \
    --data_root $DATA \
    --output_dir ${BASE_LOG}/results \
    --m1_checkpoint ${BASE_LOG}/m1_pure_pcn/best_model.pth \
    --m3_checkpoint ${BASE_LOG}/m3_lrpcn_label_guided/best_model.pth \
    --m4_checkpoint ${BASE_LOG}/m4_blind_displacement/best_model.pth \
    --m5_checkpoint ${BASE_LOG}/m5_flow_displacement/best_model.pth \
    --m7_checkpoint ${BASE_LOG}/m7_seed_displacement/best_model.pth \
    --m7_ablation_checkpoint ${BASE_LOG}/m7a_seed_no_local/best_model.pth

# ============================================================
# SAVE PCDs for M7 (best method)
# ============================================================
echo ""
echo "============================================================"
echo "Saving M7 PCDs for visualization"
echo "============================================================"

python inference_seed_save_pcd.py \
    --checkpoint ${BASE_LOG}/m7_seed_displacement/best_model.pth \
    --data_root $DATA --split test --num_samples 10 \
    --save_dir ${BASE_LOG}/m7_seed_displacement/pcds_test

# ============================================================
echo ""
echo "============================================================"
echo "COMPLETE!"
echo "============================================================"
echo ""
echo "Results in: ${BASE_LOG}/"
echo ""
echo "Key outputs:"
echo "  ${BASE_LOG}/results/summary_table.csv  ← ARTICLE TABLE"
echo "  ${BASE_LOG}/results/*_results.csv      ← Per-sample metrics"
echo "  ${BASE_LOG}/*/inference_*/              ← Visualization images"
echo "  ${BASE_LOG}/*/training_curves.png      ← Training curves"
echo "  ${BASE_LOG}/m7_seed_displacement/pcds_test/ ← 3D point clouds"
