#!/bin/bash -l
#SBATCH -N 1 -n 1 --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=36:00:00
#SBATCH -V
#SBATCH --ntasks-per-node=1
#SBATCH --output=logs/steer_generate_%j.out
#SBATCH --error=logs/steer_generate_%j.err

cd $SLURM_SUBMIT_DIR
/bin/hostname
export PATH=/home/victor_nardi/anaconda3/bin:$PATH
source /home/victor_nardi/.bashrc

mkdir -p logs

source CrystaLLM/crystallm_venv/bin/activate

pip install -q "numpy<2"

# WITH_SG=1 (default) includes the space-group header in the prompt;
# WITH_SG=0 uses composition only (output gets a _nosg suffix).
SG_FLAG="--with-spacegroup"
if [ "${WITH_SG:-1}" = "0" ]; then SG_FLAG=""; fi

# USE_CACHE=1 enables KV-cached decoding (generate_cached); default off.
CACHE_FLAG=""
if [ "${USE_CACHE:-0}" = "1" ]; then CACHE_FLAG="--use-cache"; fi

python scripts/steer_generate_cif.py \
    --model CrystaLLM/crystallm_v1_large \
    --pkl CrystaLLM/cifs_v1_test.pkl.gz \
    --temperature ${TEMPERATURE:-1.0} \
    --alpha ${ALPHA:-1.0} \
    --layer ${LAYER:-14} \
    --n-prompts ${N_PROMPTS:-0} \
    --n-samples 3 \
    ${SG_FLAG} \
    ${CACHE_FLAG} \
    --out generated_cifs/
