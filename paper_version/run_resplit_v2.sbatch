#!/bin/bash
#SBATCH --job-name=resplit_v2
#SBATCH --partition=rocky
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --output=resplit_v2_%j.out
#SBATCH --error=resplit_v2_%j.err

# ── активируем окружение ──
module purge
module load Python

eval "$(conda shell.bash hook)"
conda activate sam_env

# ── запуск ──
echo "Start: $(date)"
python -u resplit_postprocess_v2.py
echo "End: $(date)"
