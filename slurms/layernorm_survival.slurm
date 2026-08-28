#!/bin/bash -l
#SBATCH -N 1 -n 1 --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00
#SBATCH -V
#SBATCH --output=logs/layernorm_survival_%j.out
#SBATCH --error=logs/layernorm_survival_%j.err

cd $SLURM_SUBMIT_DIR
/bin/hostname
export PATH=/home/victor_nardi/anaconda3/bin:$PATH
source /home/victor_nardi/.bashrc
mkdir -p logs
source CrystaLLM/crystallm_venv/bin/activate

# Forward passes only, no generation -- a few minutes.
#   sbatch --export=ALL,METHOD=pca_centroid,TARGET=30,T=0.5 slurms/layernorm_survival.slurm
METHOD=${METHOD:-linear}
case "$METHOD" in
  pca_centroid) FLAGS="--target ${TARGET:-30} --t ${T:-0.5} --k ${K:-64}" ;;
  *)            FLAGS="--alpha ${ALPHA:-40}" ;;
esac

python scripts/analysis/layernorm_survival.py \
    --method "$METHOD" \
    ${FLAGS} \
    --layer ${LAYER:-14} \
    --steering-property ${STEERING_PROPERTY:-density_atomic} \
    --n-prompts ${N_PROMPTS:-40}
