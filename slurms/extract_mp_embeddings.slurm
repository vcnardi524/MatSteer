#!/bin/bash -l
#SBATCH -N 1 -n 1 --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH -V
#SBATCH --ntasks-per-node=1
#SBATCH --output=logs/mp_embeddings_%j.out
#SBATCH --error=logs/mp_embeddings_%j.err

cd $SLURM_SUBMIT_DIR
/bin/hostname
export PATH=/home/victor_nardi/anaconda3/bin:$PATH
source /home/victor_nardi/.bashrc

mkdir -p logs

# MODEL picks both the weights loaded and the embeddings/<dataset>/<model>/ subdir.
# llamat2 needs a venv with transformers -- crystallm_venv does not have it.
#   sbatch --export=ALL,MODEL=llamat2,CKPT_DIR=<hf-dir> slurms/extract_cif_embeddings.slurm
MODEL=${MODEL:-crystallm}
CKPT_DIR=${CKPT_DIR:-CrystaLLM/crystallm_v1_large}

# crystallm_venv has no transformers, so llamat2 needs its own venv. Override with
# VENV=<path> if it lands somewhere else.
if [ "$MODEL" = "crystallm" ]; then
    VENV=${VENV:-CrystaLLM/crystallm_venv}
else
    VENV=${VENV:-llamat_venv}
fi
[ -d "$VENV" ] || { echo "venv $VENV not found"; exit 1; }
source "$VENV/bin/activate"
echo "venv: $VENV"

pip install -q "numpy<2"

# Dataset (v1_mp) is inferred from the pkl stem, so embeddings land in
# embeddings/v1_mp/$MODEL/$VARIANT/. See slurms/extract_cif_embeddings.slurm for what the
# variants mean. Submit nosym with:
#   sbatch --export=ALL,VARIANT=nosym slurms/extract_mp_embeddings.slurm
# Resumable: re-running skips ids already checkpointed per layer.
VARIANT=${VARIANT:-full}
echo "Extracting model=$MODEL variant=$VARIANT from $CKPT_DIR"

python scripts/embeddings/extract_cif_embeddings.py \
    --model "$MODEL" \
    --ckpt-dir "$CKPT_DIR" \
    --pkl CrystaLLM/cifs_v1_mp_prep.pkl.gz \
    --variant "$VARIANT" \
    --batch-size 32 \
    --checkpoint-every 200
