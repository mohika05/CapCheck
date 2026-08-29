#!/bin/bash
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=30G
#SBATCH --ntasks-per-node=4
#SBATCH --time=120
#SBATCH --job-name=augment
#SBATCH --output=output_%x_%j.out
#SBATCH --error=error_%x_%j.err

module load cuda/12.2
module load anaconda
source activate techjam

export ENV=tc1

cd ~/techjam-track5

echo "Starting augmentation..."
python src/augment.py
echo "Augmentation done"