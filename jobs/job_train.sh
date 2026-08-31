#!/bin/bash
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=30G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=60
#SBATCH --job-name=train
#SBATCH --output=output_%x_%j.out
#SBATCH --error=error_%x_%j.err

set -euo pipefail

module load cuda/12.2
source ~/venvs/techjam312/bin/activate
export ENV=tc1

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-/tc1home/FYP/faye0004/track5}"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/src:${PYTHONPATH:-}"

FEATURES_DIR="$(python -c 'import config; print(config.FEATURES_TRAIN)')"
MODEL_PATH="$(python -c 'import config; print(config.MODEL_PATH)')"

if ! compgen -G "$FEATURES_DIR/shard_*.npz" > /dev/null; then
    echo "ERROR: no feature shards found in $FEATURES_DIR"
    echo "Run: sbatch jobs/job_extract.sh"
    exit 1
fi

if [ ! -f "$FEATURES_DIR/meta.json" ]; then
    echo "ERROR: meta.json missing in $FEATURES_DIR"
    exit 1
fi

echo "Project:  $PROJECT_ROOT"
echo "Features: $FEATURES_DIR"
echo "Model:    $MODEL_PATH"
echo "Job:      $SLURM_JOB_ID"
echo "Node:     $SLURMD_NODENAME"
echo "Started:  $(date)"

echo "Starting training..."
python src/train.py

echo "Finished: $(date)"

