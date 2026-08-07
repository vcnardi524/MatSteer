#!/bin/bash -l
#SBATCH -N 1 -n 1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=logs/plot_wyckoff_%j.out
#SBATCH --error=logs/plot_wyckoff_%j.err

cd $SLURM_SUBMIT_DIR
/bin/hostname
export PATH=/home/victor_nardi/anaconda3/bin:$PATH
source /home/victor_nardi/.bashrc

mkdir -p logs

source CrystaLLM/crystallm_venv/bin/activate

pip install -q "numpy<2"

# Single-space-group t-SNE/PCA coloured by Wyckoff signature.
# LAYER selects the embedding layer; SG the space group.
# Output: plots/layer<LAYER>/tsne_pca_<sg>_wyckoff.png
LAYER=${LAYER:-14}
SG=${SG:-"F-43m"}

python scripts/plots/plot_tsne_f43m_wyckoff.py \
    --sg "${SG}" \
    --layer ${LAYER} \
    --n-samples ${N_SAMPLES:-10000} \
    --top-n ${TOP_N:-8}
