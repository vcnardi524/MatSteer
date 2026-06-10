#!/bin/bash -l
#SBATCH -N 1 -n 1 --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH -V
#SBATCH --ntasks-per-node=1
#SBATCH --output=logs/cif_embeddings_%j.out
#SBATCH --error=logs/cif_embeddings_%j.err

cd $SLURM_SUBMIT_DIR
/bin/hostname
export PATH=/home/victor_nardi/anaconda3/bin:$PATH
source /home/victor_nardi/.bashrc

mkdir -p logs

source CrystaLLM/crystallm_venv/bin/activate

pip install -q "numpy<2"

python scripts/extract_cif_embeddings.py \
    --model CrystaLLM/crystallm_v1_large \
    --pkl CrystaLLM/cifs_v1_prep.pkl.gz \
    --layers 0,1,2,3,4,5,6,7,8,9,10,11,12,13,15 \
    --batch-size 32 \
    --checkpoint-every 200
