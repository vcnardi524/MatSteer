#!/bin/bash
#SBATCH --job-name=sym_sep
#SBATCH --output=logs/symmetry_separability_%j.log
#SBATCH --error=logs/symmetry_separability_%j.log
#SBATCH --mem=120G
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00

source /home/victor_nardi/MatSteer/CrystaLLM/crystallm_venv/bin/activate
cd /home/victor_nardi/MatSteer

# Peak RSS is ~50G per layer: the embedding DataFrame holds 2.28M per-row numpy
# arrays, plus the stacked float32 matrix and a float64 temporary in
# group_pair_sums. Memory is freed between layers, so one job does all 16.
#
# LAYERS: space-separated list, e.g. LAYERS="0 5 14" sbatch slurms/symmetry_separability.slurm
# Empty (the default) runs all 16 layers, ~3 min each on a compute node.
#
# LABEL_COL: which symmetry label to test — space_group_symbol (default) or
# point_group. Read from the environment by the script; sbatch forwards it, so
#   LABEL_COL=point_group sbatch slurms/symmetry_separability.slurm
# Output goes to analysis/symmetry_separability_<LABEL_COL>.csv, so the two runs
# do not overwrite each other.
LAYERS=${LAYERS:-""}

python -u scripts/analysis/symmetry_separability.py ${LAYERS}
