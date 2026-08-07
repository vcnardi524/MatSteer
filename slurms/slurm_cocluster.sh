#!/bin/bash
#SBATCH --job-name=cocluster
#SBATCH --output=logs/cocluster_%j.log
#SBATCH --error=logs/cocluster_%j.log
#SBATCH --mem=150G
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00

source /home/victor_nardi/MatSteer/CrystaLLM/crystallm_venv/bin/activate
cd /home/victor_nardi/MatSteer

# LAYERS: space-separated list, e.g. LAYERS="0 7 14" sbatch slurm_cocluster.sh
LAYERS=${LAYERS:-"14"}

for LAYER in ${LAYERS}; do
    echo "========================================"
    echo "Starting cocluster K=${N_CLUSTERS} name=${RUN_NAME} layer=${LAYER}"
    echo "========================================"
    python scripts/analysis/spec_cocluster_analysis.py ${N_CLUSTERS} ${RUN_NAME} ${LAYER}
    echo "Finished layer=${LAYER}, clearing memory..."
    sync
done

echo "All layers done."
