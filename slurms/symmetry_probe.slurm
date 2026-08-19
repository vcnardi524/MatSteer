#!/bin/bash
#SBATCH --job-name=sym_probe
#SBATCH --output=logs/symmetry_probe_%j.log
#SBATCH --error=logs/symmetry_probe_%j.log
#SBATCH --mem=120G
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --exclude=node11

source /home/victor_nardi/MatSteer/CrystaLLM/crystallm_venv/bin/activate
cd /home/victor_nardi/MatSteer

# Exp 1.3: logistic-regression probe vs lookup / composition baselines and a majority floor.
#
# LAYERS:     space-separated, e.g. LAYERS="0 5 14". Empty runs the script default.
# LABEL_COLS: comma-separated subset of space_group_symbol,point_group,wyckoff_letters.
#             One job per label: the output filename is keyed on LABEL_COLS, so running
#             two labels together writes a combined CSV the table plot cannot split.
# VARIANT:    full (default) or nosym.
#
# The probe always fits on the train split and scores on val -- that is fixed in the
# script, not a knob, so there is no PARTITION variable any more.
#
#   sbatch --export=ALL,LABEL_COLS=point_group,VARIANT=nosym,LAYERS="0 1 2 ..." \
#          slurms/symmetry_probe.slurm
#
# Memory is dominated by the embedding frame (~50G per layer), not the fit. Layers are
# processed one at a time and freed in between.
#
# scikit-learn is CPU-only, so this asks for cpus, not a GPU.
#
# node11 excluded for the same reason as symmetry_separability.slurm: it advertises
# 192G but has been seen with <5G free, which OOM-killed job 323209.
LAYERS=${LAYERS:-""}

python -u scripts/analysis/symmetry_probe.py ${LAYERS}
