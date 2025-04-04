#!/bin/bash
#SBATCH --account=def-mmehride_cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=0-10:00:00
#SBATCH --output=%N-%j.out

module load python/3.10.13 gcc/12.3 arrow/18.1.0

virtualenv --no-download $SLURM_TMPDIR/venv
source $SLURM_TMPDIR/venv/bin/activate

pip install --no-index torch transformers datasets accelerate tensorboard sentencepiece protobuf zstandard

export HF_HUB_OFFLINE=1
export HF_HOME=./huggingface

python combine_datasets.py