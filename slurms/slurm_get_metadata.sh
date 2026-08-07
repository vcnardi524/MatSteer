#!/bin/bash
#SBATCH --job-name=get_metadata
#SBATCH --output=logs/get_metadata_%j.log
#SBATCH --error=logs/get_metadata_%j.log
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00

source /home/victor_nardi/MatSteer/CrystaLLM/crystallm_venv/bin/activate
cd /home/victor_nardi/MatSteer

python scripts/data/get_preparsed_metadata_nomad.py \
    --pkl CrystaLLM/cifs_v1_prep.pkl.gz \
    --out preparsed_metadata_nomad.parquet \
    --batch-size 200 \
    --workers 8
