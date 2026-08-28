#!/bin/bash -l
#SBATCH -N 1 -n 1 --partition=batch
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=4:00:00
#SBATCH -V
#SBATCH --output=logs/centroid_pca_%j.out
#SBATCH --error=logs/centroid_pca_%j.err

cd $SLURM_SUBMIT_DIR
/bin/hostname
export PATH=/home/victor_nardi/anaconda3/bin:$PATH
source /home/victor_nardi/.bashrc
mkdir -p logs
source CrystaLLM/crystallm_venv/bin/activate

# No GPU: streams the layer parquet and averages per bucket.
#   sbatch --export=ALL,PARTITION=train slurms/centroid_pca_plots.slurm
#   sbatch --export=ALL,PARTITION=val,PROPERTY=dos_electronic.band_gap,\
#          LABELS=metadata.parquet,WIDTH=0.25 slurms/centroid_pca_plots.slurm
python scripts/plots/centroid_pca_plots.py \
    --partition ${PARTITION:-train} \
    --property ${PROPERTY:-density_atomic} \
    --labels ${LABELS:-density_atomic_v1.parquet} \
    --width ${WIDTH:-1.0} \
    --layer ${LAYER:-14} \
    --k ${K:-64} \
    --min-count ${MIN_COUNT:-30} \
    --basis ${BASIS:-corpus}
