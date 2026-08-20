#!/bin/bash -l
#SBATCH -N 1 -n 1 --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --output=logs/relax_steered_%j.out
#SBATCH --error=logs/relax_steered_%j.err

cd $SLURM_SUBMIT_DIR
/bin/hostname
export PATH=/home/victor_nardi/anaconda3/bin:$PATH
source /home/victor_nardi/.bashrc

mkdir -p logs

# relax_venv: PyTorch 2.4.1+cu121 (supports V100/sm_70) + matgl 4.0.1
source relax_venv/bin/activate

INPUT=${INPUT:-"steering_results/bandgap/validation/steered_test_clean_alpha16.0_layer14.parquet"}
RESULTS_DIR=${RESULTS_DIR:-steering_results}
# --input is the validation (flags) file; raw CIFs are read from the matching
# <results-dir>/generated_cifs/<stem>, output goes to <results-dir>/relaxed/<stem>.

python scripts/eval/relax_steered_cifs.py \
    --input ${INPUT} \
    --results-dir ${RESULTS_DIR} \
    --fmax 0.05 \
    --steps 500 \
    --timeout 60 \
    --checkpoint-every 10
