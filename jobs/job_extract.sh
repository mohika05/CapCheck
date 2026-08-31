#!/bin/bash
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=30G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=300
#SBATCH --job-name=extract
#SBATCH --output=output_%x_%j.out
#SBATCH --error=error_%x_%j.err

set -euo pipefail

module load cuda/12.2
source ~/venvs/techjam312/bin/activate

# This is the only switch needed for the full GPU configuration.
export ENV=tc1

# Resolve the repo root from this job file, so the repo does NOT have to be
# named ~/track5.
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-/tc1home/FYP/faye0004/track5}"
cd "$PROJECT_ROOT"

export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/src:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-$HOME/.cache/track5/huggingface}"
export TORCH_HOME="${TORCH_HOME:-$HOME/.cache/track5/torch}"
mkdir -p "$HF_HOME" "$TORCH_HOME"

echo "Project: $PROJECT_ROOT"
echo "Job:     $SLURM_JOB_ID"
echo "Node:    $SLURMD_NODENAME"
echo "Started: $(date)"
python -c "import config; print('ENV:', config.ENV); print('Features:', config.FEATURES_TRAIN)"

echo "Starting fresh feature extraction..."
python src/stream_extract.py --overwrite

echo "Finished: $(date)"
