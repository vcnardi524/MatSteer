#!/bin/bash
#SBATCH --job-name=get_metadata_mp
#SBATCH --output=logs/get_metadata_mp_%j.log
#SBATCH --error=logs/get_metadata_mp_%j.log
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00

source /home/victor_nardi/MatSteer/CrystaLLM/crystallm_venv/bin/activate
cd /home/victor_nardi/MatSteer

python scripts/data/get_preparsed_metadata_mp.py \
    --out preparsed_metadata_mp.parquet \
    --batch-size 1000
