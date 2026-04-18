#!/usr/bin/env bash
# slurm_pretrain_sweep.sh — CPC pretraining sweep + linear probe evaluation.
#
# Each array task: pretrain CPC → find saved encoder → run few-shot linear probe.
#
# Submit all:
#   sbatch scripts/slurm_pretrain_sweep.sh
#
# Submit subset:
#   sbatch --array=0-1 scripts/slurm_pretrain_sweep.sh
#
# Index → config:
#   0  overnight_v1   k_steps=8, 512 neg, hidden=512, domain_neg=0.8   ← main run
#   1  baseline       k_steps=4, 256 neg, hidden=256, domain_neg=0.5   ← ablation: default CPC
#   2  k_steps_4      k_steps=4, 512 neg, hidden=512, domain_neg=0.8   ← ablation: k_steps only
#   3  no_domain      k_steps=8, 512 neg, hidden=512, domain_neg=0.0   ← ablation: no hard neg

#SBATCH --job-name=cpc-sweep
#SBATCH --array=0-3
#SBATCH -N1 --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --constraint=gpu-rtxpro-blackwell|gpu-h100|gpu-h200
#SBATCH --mem=48G
#SBATCH --cpus-per-task=5
#SBATCH --time=02:30:00
#SBATCH --qos=coc-ice
#SBATCH --output=./logs/pretrain_%A_%a.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ckirby38@gatech.edu

set -eo pipefail
mkdir -p logs

NAMES=(
    "overnight_v1"
    "baseline"
    "k_steps_4"
    "no_domain_aware"
)

EXTRA_ARGS=(
    "--cpc_k_steps 8 --cpc_num_negatives 512 --emb_dim 512 --domain_aware --domain_neg_ratio 0.8"
    "--cpc_k_steps 4 --cpc_num_negatives 256 --emb_dim 256 --domain_aware --domain_neg_ratio 0.5"
    "--cpc_k_steps 4 --cpc_num_negatives 512 --emb_dim 512 --domain_aware --domain_neg_ratio 0.8"
    "--cpc_k_steps 8 --cpc_num_negatives 512 --emb_dim 512 --domain_neg_ratio 0.0"
)

IDX=${SLURM_ARRAY_TASK_ID:-0}
NAME=${NAMES[$IDX]}
EXTRA=${EXTRA_ARGS[$IDX]}

echo "=============================="
echo "Job:    $SLURM_JOB_ID  (array task $IDX)"
echo "Run:    $NAME"
echo "Extra:  $EXTRA"
echo "Node:   $SLURMD_NODENAME"
echo "=============================="

cd "$SLURM_SUBMIT_DIR"
source ~/.bashrc

export WANDB_ENTITY="cs8803hsi"
export WANDB_PROJECT="ood-detection"
export WANDB_RUN_NAME="${NAME}_job${SLURM_JOB_ID}"
export WANDB_DIR="$SLURM_SUBMIT_DIR/wandb"
mkdir -p "$WANDB_DIR"

SAVE_DIR="pretrain_results/sweep"

# ------------------------------------------------------------------
# STEP 1: Pretrain CPC
# ------------------------------------------------------------------
echo ""
echo ">>> STEP 1: CPC pretraining ($NAME)"

pixi run -e cuda128 python -u scripts/pretrain.py \
    --pretrain_method cpc \
    --all_tasks \
    --data_dir data \
    --epochs 150 \
    --batch_size 600 \
    --num_workers 5 \
    --learning_rate 5e-4 \
    --warmup_epochs 10 \
    --pin_memory \
    --persistent_workers \
    --prefetch_factor 4 \
    --amp \
    --save_dir "$SAVE_DIR" \
    --wandb_entity "$WANDB_ENTITY" \
    --wandb_project "$WANDB_PROJECT" \
    --wandb_run_name "${WANDB_RUN_NAME}_pretrain" \
    --seed 42 \
    $EXTRA 2>&1 | tee "logs/pretrain_${SLURM_JOB_ID}_${IDX}_step1.log"

# ------------------------------------------------------------------
# STEP 2: Find saved encoder
# ------------------------------------------------------------------
ENCODER=$(find "$SAVE_DIR/all_tasks/cpc" -name "encoder_weights.pt" \
    -newer "logs/pretrain_${SLURM_JOB_ID}_${IDX}_step1.log" \
    | sort -t/ -k1 | tail -1)

if [ -z "$ENCODER" ]; then
    echo "ERROR: encoder_weights.pt not found under $SAVE_DIR/all_tasks/cpc"
    exit 1
fi

echo ""
echo ">>> Encoder: $ENCODER"

# ------------------------------------------------------------------
# STEP 3: Linear probe — few-shot sweep across OOD splits
# ------------------------------------------------------------------
echo ""
echo ">>> STEP 2: Linear probe ($NAME)"

PROBE_DIR="results/linear_probe/${NAME}_${SLURM_JOB_ID}"

pixi run -e cuda128 python -u scripts/linear_probe.py \
    --encoder "$ENCODER" \
    --data_dir data \
    --tasks FallDetection BreathingDetection \
    --k_shots 1 2 4 8 16 32 64 full \
    --seeds 42 43 44 \
    --batch_size 512 \
    --num_workers 5 \
    --save_dir "$PROBE_DIR" \
    2>&1 | tee "logs/pretrain_${SLURM_JOB_ID}_${IDX}_step2.log"

echo ""
echo ">>> Done. Results: $PROBE_DIR"
