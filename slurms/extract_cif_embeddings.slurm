#!/bin/bash -l
#SBATCH -N 1 -n 1 --partition=gpu
#SBATCH --gres=gpu:1
# Partition allows up to 10 days. 24h was not enough: a full 2.29M-CIF pass over all
# 16 layers runs ~20h with length-sorted batching (and ~46h without), so 24h left no
# margin for node contention.
#SBATCH --time=3-00:00:00
#SBATCH -V
#SBATCH --ntasks-per-node=1
#SBATCH --output=logs/cif_embeddings_%j.out
#SBATCH --error=logs/cif_embeddings_%j.err

cd $SLURM_SUBMIT_DIR
/bin/hostname
export PATH=/home/victor_nardi/anaconda3/bin:$PATH
source /home/victor_nardi/.bashrc

mkdir -p logs

source CrystaLLM/crystallm_venv/bin/activate

pip install -q "numpy<2"

# VARIANT=full  (default) embeds the CIFs as-is.
# VARIANT=nosym drops the _symmetry_space_group_name_H-M / _symmetry_Int_Tables_number
#   lines first, so the space group is not handed to the model verbatim. Output goes to
#   embeddings/v1_all/$VARIANT/, so the two never overwrite each other.
#   Submit with: sbatch --export=ALL,VARIANT=nosym slurms/extract_cif_embeddings.slurm
VARIANT=${VARIANT:-full}
echo "Extracting variant: $VARIANT"

python scripts/embeddings/extract_cif_embeddings.py \
    --model CrystaLLM/crystallm_v1_large \
    --pkl CrystaLLM/cifs_v1_prep.pkl.gz \
    --variant "$VARIANT" \
    --batch-size 32 \
    --checkpoint-every 200
