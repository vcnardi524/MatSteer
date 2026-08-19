#!/bin/bash -l
#SBATCH -N 1 -n 1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --output=logs/predict_bandgap_testset_%j.out
#SBATCH --error=logs/predict_bandgap_testset_%j.err

cd $SLURM_SUBMIT_DIR
/bin/hostname
export PATH=/home/victor_nardi/anaconda3/bin:$PATH
source /home/victor_nardi/.bashrc

mkdir -p logs

source megnet_venv/bin/activate

# Baseline gaps for the reference test CIFs. RESULTS_DIR picks the property tree;
# output lands in <RESULTS_DIR>/property_predictions/testset_baseline.parquet.
# The script resumes from that file, so delete it first to force a full recompute.
RESULTS_DIR=${RESULTS_DIR:-"steering_results/bandgap"}

python scripts/eval/predict_bandgap_testset.py --results-dir ${RESULTS_DIR}
