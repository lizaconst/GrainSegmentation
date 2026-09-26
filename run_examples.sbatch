#!/bin/bash
#SBATCH --job-name=wc_examples
#SBATCH --partition=rocky
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --output=logs/examples_%j.out

# Runs pipeline.py on the five example images (one per grade) and compares
# the output with the reference segmentation used in the paper.

mkdir -p logs
module purge
module load Python
eval "$(conda shell.bash hook)"
conda activate sam_env

for G in Ultra_Co6_2 Ultra_Co8 Ultra_Co11 Ultra_Co15 Ultra_Co25; do
  python -u pipeline.py --alloy "$G" \
      --images "examples/images/$G" --out "out/$G" --no-debug-panel
done

python -u compare_reference.py --ref examples/reference_output --new out
