#!/bin/bash -l
#SBATCH -N 1 -n 1 --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --ntasks-per-node=1
#SBATCH --output=logs/test_kv_cache_%j.out
#SBATCH --error=logs/test_kv_cache_%j.err

cd $SLURM_SUBMIT_DIR
/bin/hostname
export PATH=/home/victor_nardi/anaconda3/bin:$PATH
source /home/victor_nardi/.bashrc

mkdir -p logs
source CrystaLLM/crystallm_venv/bin/activate
pip install -q "numpy<2"

python scripts/steering/test_kv_cache.py --n-prompts 3 --max-new-tokens 200
