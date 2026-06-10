#!/bin/bash -l
#SBATCH -N 1 -n 1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=logs/plot_tsne_pca_%j.out
#SBATCH --error=logs/plot_tsne_pca_%j.err

cd $SLURM_SUBMIT_DIR
/bin/hostname
export PATH=/home/victor_nardi/anaconda3/bin:$PATH
source /home/victor_nardi/.bashrc

mkdir -p logs

source CrystaLLM/crystallm_venv/bin/activate

pip install -q "numpy<2"

LAYERS=${LAYERS:-"1 2 3 4 5 6 7 8 9 10 11 12 13 15"}

for LAYER in ${LAYERS}; do
    echo "========================================"
    echo "Plotting layer ${LAYER}"
    echo "========================================"
    python scripts/plot_tsne_pca_bandgap.py --layer ${LAYER} --n-samples 10000
    python scripts/plot_tsne_pca_categorical.py --layer ${LAYER} --property point_group
    python scripts/plot_tsne_pca_categorical.py --layer ${LAYER} --property space_group_symbol --top-n 20
done

echo "All layers done."
