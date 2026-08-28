#!/bin/bash -l
#SBATCH -N 1 -n 1 --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00
#SBATCH -V
#SBATCH --output=logs/layer_causal_probe_%j.out
#SBATCH --error=logs/layer_causal_probe_%j.err

cd $SLURM_SUBMIT_DIR
/bin/hostname
export PATH=/home/victor_nardi/anaconda3/bin:$PATH
source /home/victor_nardi/.bashrc
mkdir -p logs
source CrystaLLM/crystallm_venv/bin/activate

# Teacher-forced forward passes only, every layer. Needs a per-layer steering vector:
#   for L in $(seq 0 15); do python scripts/steering/compute_steering_vector.py \
#       --property density_atomic --name density_atomic --metadata metadata_mp.parquet \
#       --dataset v1_mp --id-col material_id --layer $L --pct 10; done
#
#   sbatch --export=ALL,PROPERTY=density_atomic,FRAC=0.21 slurms/layer_causal_probe.slurm
python scripts/analysis/layer_causal_probe.py \
    --property ${PROPERTY:-density_atomic} \
    --frac ${FRAC:-0.21} \
    --n-cifs ${N_CIFS:-60}
