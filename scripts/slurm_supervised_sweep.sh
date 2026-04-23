#!/usr/bin/env bash
# slurm_supervised_sweep.sh — Supervised Transformer Baseline Sweep
#
# Runs supervised training for cross-domain evaluation tasks to
# establish a robust apples-to-apples baseline for CPC probes.
#
# Submit:
#   sbatch scripts/slurm_supervised_sweep.sh

#SBATCH --job-name=sup-base
#SBATCH --array=0-1
#SBATCH -N1 --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --constraint=gpu-rtxpro-blackwell|gpu-h100|gpu-h200
#SBATCH --mem=32G
#SBATCH --cpus-per-task=5
#SBATCH --time=04:00:00
#SBATCH --qos=coc-ice
#SBATCH --output=./logs/supervised_%A_%a.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ckirby38@gatech.edu

set -eo pipefail
mkdir -p logs

TASKS=(
    "HumanActivityRecognition"
    "ProximityRecognition"
)

IDX=${SLURM_ARRAY_TASK_ID:-0}
TASK=${TASKS[$IDX]}

echo "=============================="
echo "Job:    $SLURM_JOB_ID  (array task $IDX)"
echo "Task:   $TASK"
echo "Node:   $SLURMD_NODENAME"
echo "=============================="

cd "$SLURM_SUBMIT_DIR"
source ~/.bashrc

export WANDB_ENTITY="cs8803hsi"
export WANDB_PROJECT="ood-detection"

SAVE_DIR="results/supervised"

echo ""
echo ">>> Running Supervised Baseline for $TASK"

pixi run -e cuda128 python -u scripts/train_supervised.py \
    --task "$TASK" \
    --data_dir data \
    --model transformer \
    --emb_dim 256 \
    --depth 4 \
    --num_heads 8 \
    --dropout 0.1 \
    --learning_rate 0.001 \
    --batch_size 128 \
    --epochs 100 \
    --use_adapter \
    --save_dir "$SAVE_DIR" \
    --wandb_project "$WANDB_PROJECT" \
    --wandb_entity "$WANDB_ENTITY" \
    --wandb_run_name "supervised_${TASK}_job${SLURM_JOB_ID}" \
    2>&1 | tee "logs/supervised_${SLURM_JOB_ID}_${IDX}.log"

echo ""
echo ">>> Done. Results: $SAVE_DIR/$TASK"
