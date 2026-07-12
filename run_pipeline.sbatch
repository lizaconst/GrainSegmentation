#!/bin/bash
#SBATCH --job-name=wc_pipeline
#SBATCH --partition=rocky
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --array=0-4
#SBATCH --output=logs/pipeline_%A_%a.out
#SBATCH --error=logs/pipeline_%A_%a.err

# SLURM job array: one task per grade.
# Adjust --partition / --gres / conda activation to your cluster setup.

ALLOYS=(Ultra_Co6_2 Ultra_Co8 Ultra_Co11 Ultra_Co15 Ultra_Co25)
ALLOY=${ALLOYS[$SLURM_ARRAY_TASK_ID]}

mkdir -p logs

source ~/.bashrc
conda activate sam_env

echo "Task $SLURM_ARRAY_TASK_ID -> $ALLOY on $(hostname)"
python -u pipeline.py --alloy "$ALLOY"
