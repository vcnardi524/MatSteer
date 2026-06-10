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

INPUT=${INPUT:-"validation/steered_test_p5_alpha1.0_layer14.parquet"}
# Output written back to same parquet as new column: predicted_bandgap_ev

python scripts/predict_bandgap.py \
    --input ${INPUT} \
    --fidelity 0 \
    --source ${SOURCE:-cif_relaxed}
