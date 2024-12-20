#!/bin/bash

#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:0
#SBATCH --error=Lables-CLIPGG-%J.err
#SBATCH --output=Lables-CLIPGG-%J.out
#SBATCH --job-name=Lables-CLIPGG

# load anaconda and cuda modules
module load anaconda
conda activate pytorch2.2.0+py3.11+cuda12.1
pip install datasets
module load cuda/cuda-12.2.2 

# list the modules 
module list

echo "Start:"
echo "Lables-CLIPGG"
time python extract_labels.py
echo "Over."
