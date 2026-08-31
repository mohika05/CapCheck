#!/bin/bash
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=30G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=180
#SBATCH --job-name=predict
#SBATCH --output=output_%x_%j.out
#SBATCH --error=error_%x_%j.err

set -euo pipefail

module load cuda/12.2
source ~/venvs/techjam312/bin/activate
export ENV=tc1

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-/tc1home/FYP/faye0004/track5}"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/src:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-$HOME/.cache/track5/huggingface}"
export TORCH_HOME="${TORCH_HOME:-$HOME/.cache/track5/torch}"

MODEL_PATH="$(python -c 'import config; print(config.MODEL_PATH)')"
PREDICT_IMAGES="$(python -c 'import config; print(config.PREDICT_IMAGES)')"
PREDICTIONS_OUT="$(python -c 'import config; print(config.PREDICTIONS_OUT)')"
ROBUSTNESS_OUT="$(python -c 'import config; print(config.ROBUSTNESS_OUT)')"

if [ ! -f "$MODEL_PATH" ]; then
    echo "ERROR: model not found at $MODEL_PATH"
    echo "Run: sbatch jobs/job_train.sh"
    exit 1
fi

if [ ! -d "$PREDICT_IMAGES" ]; then
    echo "ERROR: WildFake validation directory not found: $PREDICT_IMAGES"
    exit 1
fi

echo "Project:    $PROJECT_ROOT"
echo "Model:      $MODEL_PATH"
echo "Images:     $PREDICT_IMAGES"
echo "Predictions:$PREDICTIONS_OUT"
echo "Robustness: $ROBUSTNESS_OUT"
echo "Job:        $SLURM_JOB_ID"
echo "Node:       $SLURMD_NODENAME"
echo "Started:    $(date)"

echo "Starting WildFake clean + robustness evaluation..."
python predict.py --report-transforms

echo "Finished: $(date)"
