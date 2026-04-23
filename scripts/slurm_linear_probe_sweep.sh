#!/usr/bin/env bash
# slurm_linear_probe_sweep.sh — Few-shot linear probe evaluation sweep.
#
# Each array task loads the encoder saved by slurm_pretrain_sweep.sh
# and runs a k-shot linear probe sweep, logging results to W&B.
#
# Expected encoder path (written by pretrain.py):
#   pretrain_results/sweep/all_tasks/cpc/${NAME}/encoder_weights.pt
#
# Submit via submit_sweep.sh (recommended — handles aftercorr dependency):
#   bash scripts/submit_sweep.sh
#
# Direct submit (pretrain must have already completed):
#   sbatch scripts/slurm_linear_probe_sweep.sh
#
# Index → config (must match slurm_pretrain_sweep.sh):
#   0  overnight_v1
#   1  baseline
#   2  k_steps_4
#   3  no_domain_aware

#SBATCH --job-name=cpc-probe
#SBATCH --array=0-8
#SBATCH -N1 --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --constraint=gpu-rtxpro-blackwell|gpu-h100|gpu-h200
#SBATCH --mem=32G
#SBATCH --cpus-per-task=5
#SBATCH --time=02:00:00
#SBATCH --qos=coc-ice
#SBATCH --output=./logs/probe_%A_%a.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ckirby38@gatech.edu

set -eo pipefail
mkdir -p logs

NAMES=(
    "overnight_v1"
    "baseline"
    "k_steps_4"
    "no_domain_aware"
    "k_steps_2"
    "k_steps_6"
    "k_steps_12"
    "random_k"
    "multi_k"
)

IDX=${SLURM_ARRAY_TASK_ID:-0}
NAME=${NAMES[$IDX]}

echo "=============================="
echo "Job:    $SLURM_JOB_ID  (array task $IDX)"
echo "Run:    $NAME"
echo "Node:   $SLURMD_NODENAME"
echo "=============================="

echo ""
echo "=== GPU ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo "==========="

cd "$SLURM_SUBMIT_DIR"
source ~/.bashrc

export WANDB_ENTITY="cs8803hsi"
export WANDB_PROJECT="ood-detection"
export WANDB_DIR="$SLURM_SUBMIT_DIR/wandb"
mkdir -p "$WANDB_DIR"

SAVE_DIR="pretrain_results/sweep"
ENCODER="$SAVE_DIR/all_tasks/cpc/${NAME}/encoder_weights.pt"

if [ ! -f "$ENCODER" ]; then
    echo "ERROR: encoder not found at $ENCODER"
    echo "  Make sure slurm_pretrain_sweep.sh completed successfully for array task $IDX."
    exit 1
fi

echo ""
echo ">>> Encoder: $ENCODER"

PROBE_DIR="results/linear_probe/${NAME}"

echo ""
echo ">>> Running linear probe for $NAME"

pixi run -e cuda128 python -u scripts/linear_probe.py \
    --encoder "$ENCODER" \
    --data_dir data \
    --tasks HumanActivityRecognition ProximityRecognition \
    --k_shots 1 2 4 8 16 32 64 full \
    --seeds 42 43 44 \
    --batch_size 512 \
    --num_workers 5 \
    --save_dir "$PROBE_DIR" \
    --wandb_entity "$WANDB_ENTITY" \
    --wandb_project "$WANDB_PROJECT" \
    --wandb_run_name "${NAME}_job${SLURM_JOB_ID}_probe" \
    2>&1 | tee "logs/probe_${SLURM_JOB_ID}_${IDX}.log"

echo ""
echo ">>> Done. Results: $PROBE_DIR"
