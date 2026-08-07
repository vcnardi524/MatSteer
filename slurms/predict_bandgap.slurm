#!/bin/bash -l
#SBATCH -N 1 -n 1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --output=logs/predict_bandgap_%j.out
#SBATCH --error=logs/predict_bandgap_%j.err

cd $SLURM_SUBMIT_DIR
/bin/hostname
export PATH=/home/victor_nardi/anaconda3/bin:$PATH
source /home/victor_nardi/.bashrc

mkdir -p logs

source megnet_venv/bin/activate

# Point INPUT at a CIF file: generated_cifs/<stem> for unrelaxed (raw) gaps, or
# relaxed/<stem> for relaxed gaps. Source column is auto-detected; only valid
# CIFs (is_valid from validation/<stem>) are predicted. Both accumulate into
# <RESULTS_DIR>/property_predictions/<stem>.parquet.
RESULTS_DIR=${RESULTS_DIR:-"steering_results/bandgap"}
INPUT=${INPUT:-"${RESULTS_DIR}/generated_cifs/steered_test_clean_alpha16.0_layer14.parquet"}

python scripts/eval/compute_predictions.py \
    --property band_gap \
    --input ${INPUT} \
    --results-dir ${RESULTS_DIR} \
    --fidelity 0
