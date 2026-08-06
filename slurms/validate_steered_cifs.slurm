#!/bin/bash -l
#SBATCH -N 1 -n 1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=16
#SBATCH --time=4:00:00
#SBATCH --output=logs/validate_steered_%j.out
#SBATCH --error=logs/validate_steered_%j.err

cd $SLURM_SUBMIT_DIR
/bin/hostname
export PATH=/home/victor_nardi/anaconda3/bin:$PATH
source /home/victor_nardi/.bashrc

mkdir -p logs

source CrystaLLM/crystallm_venv/bin/activate
pip install -q "numpy<2"

INPUT=${INPUT:-"steering_results/bandgap/generated_cifs/steered_test_clean_alpha16.0_layer14.parquet"}
WORKERS=${WORKERS:-16}
RESULTS_DIR=${RESULTS_DIR:-steering_results}

python scripts/validate_steered_cifs.py \
    --input ${INPUT} \
    --results-dir ${RESULTS_DIR} \
    --workers ${WORKERS}
