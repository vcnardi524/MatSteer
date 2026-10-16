#!/bin/bash -l
#SBATCH -N 1 -n 1
# The one sbatch template. Everything that varies per experiment -- queue, walltime,
# memory, cpus, gpu, job name, log paths, node exclusions -- is passed by run.sh as an
# sbatch CLI flag, so this file carries no resource directives to disagree with.
#
# run.sh exports MATSTEER_VENV and MATSTEER_CMD in the submitting shell and passes a bare
# --export=ALL. Nothing of the form VAR=value ever reaches the sbatch command line, which
# is what makes a comma inside a value safe (see CLAUDE.md on --export splitting).
#
# Do not sbatch this file directly; it has no defaults. Use: ./run.sh <experiment>
set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
/bin/hostname
export PATH=/home/victor_nardi/anaconda3/bin:$PATH
source /home/victor_nardi/.bashrc
mkdir -p logs

source "${MATSTEER_VENV}/bin/activate"
echo "+ venv: ${MATSTEER_VENV}"
echo "+ ${MATSTEER_CMD}"
eval "${MATSTEER_CMD}"
