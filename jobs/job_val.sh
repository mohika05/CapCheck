#!/bin/bash
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=30
#SBATCH --job-name=predict
#SBATCH --output=output_%x_%j.out
#SBATCH --error=error_%x_%j.err

module load cuda/12.2
source ~/venvs/techjam312/bin/activate

export ENV=tc1
export PYTHONPATH=/tc1home/FYP/faye0004/track5

cd ~/track5
python src/predict.py /tc1home/FYP/faye0004/techjam-track5/data/validation/
