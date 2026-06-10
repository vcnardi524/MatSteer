#!/bin/bash -l
#SBATCH -N 1 -n 1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=logs/novelty_steered_%j.out
#SBATCH --error=logs/novelty_steered_%j.err

cd $SLURM_SUBMIT_DIR
/bin/hostname
export PATH=/home/victor_nardi/anaconda3/bin:$PATH
source /home/victor_nardi/.bashrc

mkdir -p logs

source CrystaLLM/crystallm_venv/bin/activate
pip install -q "numpy<2"

INPUT=${INPUT:-"validation/steered_test_p5_alpha1.0_layer14.parquet"}
BASE=${BASE:-"CrystaLLM/cifs_v1_train.pkl.gz"}

python scripts/novelty_steered_cifs.py \
    --input ${INPUT} \
    --base ${BASE}
