#!/bin/bash -l
#SBATCH -N 1 -n 1 --partition=batch
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH -V
#SBATCH --output=logs/pca_basis_%j.out
#SBATCH --error=logs/pca_basis_%j.err

cd $SLURM_SUBMIT_DIR
/bin/hostname
export PATH=/home/victor_nardi/anaconda3/bin:$PATH
source /home/victor_nardi/.bashrc

mkdir -p logs

source CrystaLLM/crystallm_venv/bin/activate

# Fits the top-K PCA basis of one layer's activations, then the target centroid that
# pca_centroid steering interpolates toward. No GPU: IncrementalPCA streams the parquet
# row groups, so memory stays at roughly BATCH_SIZE x 1024 floats regardless of layer size.
#
#   sbatch --export=ALL,LAYER=14,K=64,TARGET=30 slurms/compute_pca_basis.slurm
#
# TARGET unset fits the basis only. CLASS_SIZE is how many nearest-to-target structures
# form the centroid class; PROPERTY names both the label column in LABELS and the
# steering_vectors/pca_centroid/ subdir the centroid is written to.
LAYER=${LAYER:-14}
K=${K:-64}
PARTITION=${PARTITION:-train}
echo "layer=$LAYER k=$K partition=$PARTITION target=${TARGET:-<none>}"

python scripts/steering/compute_pca_basis.py \
    --layer "$LAYER" \
    --k "$K" \
    --partition "$PARTITION" \
    --dataset ${DATASET:-v1_all} \
    --variant ${VARIANT:-full} \
    --batch-size ${BATCH_SIZE:-50000}

if [ -n "${TARGET}" ]; then
    python scripts/steering/compute_centroid_target.py \
        --target "$TARGET" \
        --property ${PROPERTY:-density_atomic} \
        --labels ${LABELS:-density_atomic_v1.parquet} \
        --class-size ${CLASS_SIZE:-100000} \
        --layer "$LAYER" \
        --k "$K" \
        --partition "$PARTITION" \
        --dataset ${DATASET:-v1_all} \
        --variant ${VARIANT:-full} \
        --batch-size ${BATCH_SIZE:-50000}
fi
