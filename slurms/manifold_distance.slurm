#!/bin/bash -l
#SBATCH -N 1 -n 1 --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00
#SBATCH -V
#SBATCH --output=logs/manifold_distance_%j.out
#SBATCH --error=logs/manifold_distance_%j.err

cd $SLURM_SUBMIT_DIR
/bin/hostname
export PATH=/home/victor_nardi/anaconda3/bin:$PATH
source /home/victor_nardi/.bashrc
mkdir -p logs
source CrystaLLM/crystallm_venv/bin/activate

# Needs the class bank: compute_centroid_target.py --save-bank
#   sbatch --export=ALL,TARGET=30 slurms/manifold_distance.slurm
python scripts/analysis/manifold_distance.py \
    --property ${PROPERTY:-density_atomic} \
    --target ${TARGET:-30} \
    --layer ${LAYER:-14} \
    --n-prompts ${N_PROMPTS:-40}
