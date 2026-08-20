#!/bin/bash
#SBATCH --job-name=sym_sep
#SBATCH --output=logs/symmetry_separability_%j.log
#SBATCH --error=logs/symmetry_separability_%j.log
#SBATCH --mem=120G
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --exclude=node11

source /home/victor_nardi/MatSteer/CrystaLLM/crystallm_venv/bin/activate
cd /home/victor_nardi/MatSteer

# Peak RSS is ~50G per layer: the embedding DataFrame holds 2.28M per-row numpy
# arrays, plus the stacked float32 matrix and a float64 temporary in
# group_pair_sums. Memory is freed between layers, so one job does all 16.
#
# LAYERS: space-separated list, e.g. LAYERS="0 5 14" sbatch slurms/symmetry_separability.slurm
# Empty (the default) runs all 16 layers, ~3 min each on a compute node.
#
# LABEL_COL: which symmetry label to test — space_group_symbol (default) or point_group.
# VARIANT:   full (default) or nosym — which CIF text the embeddings came from.
# PARTITION: all|train|val|test. REQUIRED; the script exits if it is unset, so no run
#            silently lands on the model's own training data.
#
# All read from the environment by the script, so:
#   sbatch --export=ALL,LABEL_COL=point_group,VARIANT=nosym,PARTITION=val \
#          slurms/symmetry_separability.slurm
#
# Output goes to analysis/<DATASET>/<VARIANT>/<PARTITION>/symmetry_separability_<LABEL_COL>.csv,
# so runs never overwrite each other.
#
# node11 is excluded: it reports 192G configured but has been seen with <5G actually
# free, which OOM-killed job 323209 despite a 120G request.
LAYERS=${LAYERS:-""}

python -u scripts/analysis/symmetry_separability.py ${LAYERS}
