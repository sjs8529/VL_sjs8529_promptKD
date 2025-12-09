#!/bin/bash

# ============================================================
# PromptKD Hyperparameter Experiments on EuroSAT
# ============================================================
# Baseline: 20 epochs, temp=1.0, LR=0.005, 4 tokens, 2 proj layers, depth=9
# Your baseline result: 82.2% Novel Accuracy
# ============================================================

# Configuration
DATA="/scratch/sjs8529/promptkd_eurosat_project/datasets"
DATASET="eurosat"
TRAINER="PromptKD"
CFG="vit_b16_c2_ep20_batch8_4+4ctx"
SEED=1

echo "============================================================"
echo "Starting PromptKD Experiments on EuroSAT"
echo "============================================================"
echo ""

# ------------------------------------------------------------
# Experiment 1: Temperature = 0.5 (sharper distribution)
# ------------------------------------------------------------
echo "[Exp 1/6] Running Temperature = 0.5..."
echo "Start time: $(date)"

python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir output/experiments/exp1_temp_0.5 \
    DATASET.NUM_SHOTS 0 \
    TRAINER.MODAL base2novel \
    TRAINER.PROMPTKD.TEMPERATURE 0.5

echo "[Exp 1/6] Completed Temperature = 0.5"
echo "End time: $(date)"
echo ""

# ------------------------------------------------------------
# Experiment 2: Temperature = 2.0 (softer distribution)
# ------------------------------------------------------------
echo "[Exp 2/6] Running Temperature = 2.0..."
echo "Start time: $(date)"

python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir output/experiments/exp2_temp_2.0 \
    DATASET.NUM_SHOTS 0 \
    TRAINER.MODAL base2novel \
    TRAINER.PROMPTKD.TEMPERATURE 2.0

echo "[Exp 2/6] Completed Temperature = 2.0"
echo "End time: $(date)"
echo ""

# ------------------------------------------------------------
# Experiment 3: Learning Rate = 0.001 (lower, more conservative)
# ------------------------------------------------------------
echo "[Exp 3/6] Running LR = 0.001..."
echo "Start time: $(date)"

python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir output/experiments/exp3_lr_0.001 \
    DATASET.NUM_SHOTS 0 \
    TRAINER.MODAL base2novel \
    OPTIM.LR 0.001

echo "[Exp 3/6] Completed LR = 0.001"
echo "End time: $(date)"
echo ""

# ------------------------------------------------------------
# Experiment 4: Learning Rate = 0.01 (higher, more aggressive)
# ------------------------------------------------------------
echo "[Exp 4/6] Running LR = 0.01..."
echo "Start time: $(date)"

python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir output/experiments/exp4_lr_0.01 \
    DATASET.NUM_SHOTS 0 \
    TRAINER.MODAL base2novel \
    OPTIM.LR 0.01

echo "[Exp 4/6] Completed LR = 0.01"
echo "End time: $(date)"
echo ""

# ------------------------------------------------------------
# Experiment 5: Prompt Tokens = 8 (more capacity)
# ------------------------------------------------------------
echo "[Exp 5/6] Running N_CTX = 8..."
echo "Start time: $(date)"

python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir output/experiments/exp5_ctx_8 \
    DATASET.NUM_SHOTS 0 \
    TRAINER.MODAL base2novel \
    TRAINER.PROMPTKD.N_CTX_TEXT 8 \
    TRAINER.PROMPTKD.N_CTX_VISION 8

echo "[Exp 5/6] Completed N_CTX = 8"
echo "End time: $(date)"
echo ""

# ------------------------------------------------------------
# Experiment 6: Projector Layers = 1 (simpler architecture)
# ------------------------------------------------------------
echo "[Exp 6/6] Running PROJECT_LAYER = 1..."
echo "Start time: $(date)"

python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir output/experiments/exp6_proj_1 \
    DATASET.NUM_SHOTS 0 \
    TRAINER.MODAL base2novel \
    TRAINER.PROMPTKD.PROJECT_LAYER 1

echo "[Exp 6/6] Completed PROJECT_LAYER = 1"
echo "End time: $(date)"
echo ""

# ============================================================
# Summary
# ============================================================
echo "============================================================"
echo "ALL EXPERIMENTS COMPLETED!"
echo "============================================================"
echo ""
echo "Results saved in:"
echo "  - output/experiments/exp1_temp_0.5"
echo "  - output/experiments/exp2_temp_2.0"
echo "  - output/experiments/exp3_lr_0.001"
echo "  - output/experiments/exp4_lr_0.01"
echo "  - output/experiments/exp5_ctx_8"
echo "  - output/experiments/exp6_proj_1"
echo ""
echo "Check the log.txt file in each directory for results."
echo ""
echo "============================================================"
echo "RESULTS TABLE (fill in Novel Acc from each log.txt):"
echo "============================================================"
echo ""
echo "| Exp | Parameter      | Value | Novel Acc | Δ vs Baseline |"
echo "|-----|----------------|-------|-----------|---------------|"
echo "| BL  | baseline       | -     | 82.2%     | -             |"
echo "| E1  | TEMPERATURE    | 0.5   |           |               |"
echo "| E2  | TEMPERATURE    | 2.0   |           |               |"
echo "| E3  | LR             | 0.001 |           |               |"
echo "| E4  | LR             | 0.01  |           |               |"
echo "| E5  | N_CTX          | 8     |           |               |"
echo "| E6  | PROJECT_LAYER  | 1     |           |               |"
echo ""
echo "Total estimated time: ~4.5 hours (6 x ~46 min each)"