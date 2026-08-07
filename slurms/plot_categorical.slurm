#!/bin/bash -l
#SBATCH -N 1 -n 1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=logs/plot_categorical_%j.out
#SBATCH --error=logs/plot_categorical_%j.err

cd $SLURM_SUBMIT_DIR
/bin/hostname
export PATH=/home/victor_nardi/anaconda3/bin:$PATH
source /home/victor_nardi/.bashrc

mkdir -p logs

source CrystaLLM/crystallm_venv/bin/activate

pip install -q "numpy<2"

# Categorical t-SNE/PCA plots (point_group, space_group_symbol) at a chosen
# t-SNE perplexity. PERPLEXITY != 30 gets a _p<N> filename suffix, so the
# default p=30 plots are never overwritten.
# Output: plots/layer<L>/<property>_layer<L>[_p<N>].png
LAYERS=${LAYERS:-"14"}
PERPLEXITY=${PERPLEXITY:-30}

for LAYER in ${LAYERS}; do
    echo "======== layer ${LAYER}  perplexity ${PERPLEXITY} ========"
    python scripts/plots/plot_tsne_pca_categorical.py \
        --layer ${LAYER} \
        --property point_group \
        --tsne-perplexity ${PERPLEXITY} \
        --n-samples ${N_SAMPLES:-10000}
    python scripts/plots/plot_tsne_pca_categorical.py \
        --layer ${LAYER} \
        --property space_group_symbol \
        --top-n 20 \
        --tsne-perplexity ${PERPLEXITY} \
        --n-samples ${N_SAMPLES:-10000}
done

echo "All layers done."
