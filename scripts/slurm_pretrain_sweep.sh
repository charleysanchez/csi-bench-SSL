#!/usr/bin/env bash
# slurm_pretrain_sweep.sh — CPC pretraining sweep.
#
# Each array task pretrains one CPC config and saves encoder_weights.pt.
# The linear probe runs as a dependent job via submit_sweep.sh.
#
# Submit standalone (pretrain only):
#   sbatch scripts/slurm_pretrain_sweep.sh
#
# Submit with chained probe (recommended):
#   bash scripts/submit_sweep.sh
#
# Index → config:
#   0  overnight_v1   k_steps=8, 512 neg, hidden=512, domain_neg=0.8   ← main run
#   1  baseline       k_steps=4, 256 neg, hidden=256, domain_neg=0.5   ← ablation: default CPC
#   2  k_steps_4      k_steps=4, 512 neg, hidden=512, domain_neg=0.8   ← ablation: k_steps only
#   3  no_domain      k_steps=8, 512 neg, hidden=512, domain_neg=0.0   ← ablation: no hard neg

#SBATCH --job-name=cpc-pretrain
#SBATCH --array=0-3
#SBATCH -N1 --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --constraint=gpu-rtxpro-blackwell|gpu-h100|gpu-h200
#SBATCH --mem=48G
#SBATCH --cpus-per-task=5
#SBATCH --time=04:00:00
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

# GPU diagnostic — prints GPU model and VRAM so we can spot bad allocations
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

echo ""
echo ">>> CPC pretraining ($NAME)"

# max_train_hours=3.5 gives 30 min of buffer within the 4h wall limit for
# encoder saving, CSV, plot, and wandb sync before SLURM kills the job.
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
    --wandb_run_name "${NAME}_job${SLURM_JOB_ID}_pretrain" \
    --seed 42 \
    --max_train_hours 3.5 \
    --run_name "$NAME" \
    $EXTRA 2>&1 | tee "logs/pretrain_${SLURM_JOB_ID}_${IDX}.log"

ENCODER="$SAVE_DIR/all_tasks/cpc/${NAME}/encoder_weights.pt"
if [ -f "$ENCODER" ]; then
    echo ""
    echo ">>> Encoder saved: $ENCODER"
else
    echo ""
    echo "ERROR: encoder not found at $ENCODER"
    exit 1
fi

echo ""
echo ">>> Pretraining complete for $NAME"
