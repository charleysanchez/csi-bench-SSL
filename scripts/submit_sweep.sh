#!/usr/bin/env bash
# submit_sweep.sh — Submit pretrain + linear probe as chained SLURM array jobs.
#
# Each probe task waits for its CORRESPONDING pretrain task via aftercorr:
#   probe[0] starts only after pretrain[0] succeeds
#   probe[1] starts only after pretrain[1] succeeds
#   ... and so on
#
# Usage:
#   bash scripts/submit_sweep.sh           # all 4 configs (array 0-3)
#   bash scripts/submit_sweep.sh 0-1       # subset

set -eo pipefail

ARRAY="${1:-0-8}"

echo "Submitting pretrain sweep (array $ARRAY)..."
PRETRAIN_JID=$(sbatch --parsable --array="$ARRAY" scripts/slurm_pretrain_sweep.sh)
echo "  Pretrain job ID: $PRETRAIN_JID"

echo "Submitting linear probe sweep (array $ARRAY, aftercorr:$PRETRAIN_JID)..."
PROBE_JID=$(sbatch --parsable --array="$ARRAY" \
    --dependency=aftercorr:"$PRETRAIN_JID" \
    scripts/slurm_linear_probe_sweep.sh)
echo "  Probe job ID:    $PROBE_JID"

echo "Submitting supervised baseline sweep (array 0-1)..."
SUPERVISED_JID=$(sbatch --parsable --array="0-1" scripts/slurm_supervised_sweep.sh)
echo "  Supervised job ID: $SUPERVISED_JID"

echo ""
echo "Monitor: squeue -j $PRETRAIN_JID,$PROBE_JID,$SUPERVISED_JID"
echo "Logs:    logs/pretrain_${PRETRAIN_JID}_*.out"
echo "         logs/probe_${PROBE_JID}_*.out"
echo "         logs/supervised_${SUPERVISED_JID}_*.out"
