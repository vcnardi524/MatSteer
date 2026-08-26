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

# KV-cached decoding (generate_cached) is on by default; USE_CACHE=0 disables it.
CACHE_FLAG="--use-cache"
if [ "${USE_CACHE:-1}" = "0" ]; then CACHE_FLAG="--no-use-cache"; fi

# METHOD=linear (default) steers with ALPHA * the mean-difference vector.
# METHOD=pca_centroid interpolates a fraction T toward the TARGET centroid inside the
# top-K PCA subspace; it reads steering_vectors/pca_centroid/ and needs TARGET set.
# METHOD=pca_local does the same but rebuilds the centroid per prompt from the
# NEIGHBOURS class members nearest that prompt; it needs the matching *_bank.parquet.
#
#   sbatch --export=ALL,METHOD=pca_centroid,STEERING_PROPERTY=density_atomic,TARGET=30,T=0.5 \
#          slurms/steer_generate_cif.slurm
METHOD=${METHOD:-linear}
case "$METHOD" in
  pca_centroid) METHOD_FLAGS="--target ${TARGET:?TARGET is required for $METHOD} --t ${T:-0.5} --k ${K:-64}" ;;
  pca_local)    METHOD_FLAGS="--target ${TARGET:?TARGET is required for $METHOD} --t ${T:-0.5} --k ${K:-64} --neighbours ${NEIGHBOURS:-256}" ;;
  *)            METHOD_FLAGS="--alpha ${ALPHA:-1.0}" ;;
esac

python scripts/steering/steer_generate_cif.py \
    --model CrystaLLM/crystallm_v1_large \
    --pkl ${PKL:-CrystaLLM/cifs_v1_test.pkl.gz} \
    --temperature ${TEMPERATURE:-1.0} \
    --method ${METHOD} \
    ${METHOD_FLAGS} \
    --layer ${LAYER:-14} \
    --steering-property ${STEERING_PROPERTY:-bandgap} \
    --n-prompts ${N_PROMPTS:-0} \
    --n-samples ${N_SAMPLES:-3} \
    ${SG_FLAG} \
    ${CACHE_FLAG} \
    --results-dir steering_results/${STEERING_PROPERTY:-bandgap} \
    ${OUT:+--out $OUT}
