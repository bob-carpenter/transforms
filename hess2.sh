#!/bin/bash
#SBATCH --partition=genx  --mem=32000 --job-name="hessians"
source /mnt/home/mjhajaria/miniconda3/etc/profile.d/conda.sh
conda activate stan
cd /mnt/home/mjhajaria/transforms
python3 /mnt/home/mjhajaria/transforms/utils/simplex-scripts/hess2.py