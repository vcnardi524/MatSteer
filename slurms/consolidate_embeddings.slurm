#!/bin/bash -l
#SBATCH -N 1 -n 1 --partition=batch
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH -V
#SBATCH --output=logs/consolidate_%j.out
#SBATCH --error=logs/consolidate_%j.err

cd $SLURM_SUBMIT_DIR
/bin/hostname
export PATH=/home/victor_nardi/anaconda3/bin:$PATH
source /home/victor_nardi/.bashrc

mkdir -p logs

source CrystaLLM/crystallm_venv/bin/activate

# Merges the ~358 checkpoint parquets per layer into one cif_layer{N}.parquet.
# No GPU needed -- this is pure parquet I/O. The script streams one checkpoint at a
# time, so 32G is generous even though a finished layer is ~14 GB; it never holds a
# whole layer in memory. Layers already consolidated are skipped, so re-running after
# an interruption is safe and cheap.
#
#   sbatch --export=ALL,VARIANT=full  slurms/consolidate_embeddings.slurm
#   sbatch --export=ALL,VARIANT=nosym slurms/consolidate_embeddings.slurm
VARIANT=${VARIANT:-full}
DATASET=${DATASET:-v1_all}
echo "Consolidating dataset=$DATASET variant=$VARIANT"

python scripts/embeddings/consolidate_embeddings.py \
    --dataset "$DATASET" \
    --variant "$VARIANT"
