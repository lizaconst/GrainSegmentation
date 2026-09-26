#!/bin/bash
#SBATCH --job-name=grain_v2
#SBATCH --partition=rocky
#SBATCH --gpus=1
#SBATCH --cpus-per-task=1
#SBATCH --time=20:00:00
#SBATCH --array=0-4
#SBATCH --output=grain_v2_%A_%a.out
#SBATCH --error=grain_v2_%A_%a.err

# ── сплав по индексу массива ──
ALLOYS=(Ultra_Co6_2 Ultra_Co8 Ultra_Co11 Ultra_Co15 Ultra_Co25)
ALLOY=${ALLOYS[$SLURM_ARRAY_TASK_ID]}

# ── активируем окружение ──
module purge
module load Python

eval "$(conda shell.bash hook)"
conda activate sam_env

# ── запуск ──
echo "Start: $(date) | alloy: $ALLOY"
nvidia-smi
python -u pipeline_hpc_v2.py --alloy "$ALLOY"
echo "End: $(date)"
