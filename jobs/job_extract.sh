#!/bin/bash
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=20G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=60
#SBATCH --job-name=extract
#SBATCH --output=output_%x_%j.out
#SBATCH --error=error_%x_%j.err

module load cuda/12.2
module load anaconda
source activate techjam

export ENV=tc1

cd ~/techjam-track5

echo "Starting feature extraction..."
python src/extract_features.py
echo "CLIP extraction done"

python src/extract_dct_features.py
echo "DCT extraction done"