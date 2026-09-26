#!/bin/bash
#SBATCH --job-name=analysis_v2
#SBATCH --partition=rocky
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=analysis_v2_%j.out
#SBATCH --error=analysis_v2_%j.err

# ── активируем окружение ──
module purge
module load Python

eval "$(conda shell.bash hook)"
conda activate sam_env

# ── запуск ──
echo "Start: $(date)"
python -u final_results_analysis_v2.py
echo "End: $(date)"
