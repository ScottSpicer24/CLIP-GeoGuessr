#!/bin/bash

#SBATCH --time=72:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --error=CLIPGG-pretrain-%J.err
#SBATCH --output=CLIPGG-pretrain-%J.out
#SBATCH --job-name=CLIPGG

# load anaconda and cuda modules
module load anaconda
conda activate pytorch2.2.0+py3.11+cuda12.1
pip install datasets
pip install git+https://github.com/openai/CLIP.git
module load cuda/cuda-12.2.2 

# list the modules 
module list

echo "Start:"
echo "CLIPGG-pretrain"
time python pretrain.py
echo "Over."
