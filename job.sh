#!/bin/bash
#SBATCH --account=rrg-mmehride
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=20000M     
#SBATCH --time=0-20:00:00
#SBATCH --output=%N-%j.out


# Signal handler to save data before job times out
# #SBATCH --signal=B:SIGUSR1@3600

# function sig_handler_USR1()
# {
#     cp -r $SLURM_TMPDIR/generated_dataset ./generated_dataset_sig
#     exit 2
# }

# trap 'sig_handler_USR1' SIGUSR1

# datasets requires arrow
module load python/3.10.13 gcc/12.3 arrow/18.1.0

virtualenv --no-download $SLURM_TMPDIR/venv
source $SLURM_TMPDIR/venv/bin/activate

pip install --no-index torch transformers datasets accelerate sentencepiece protobuf zstandard

# use model and dataset directly instead of through HF cache
# (issues with identifying correct cache)
# cp -r slimpajama $SLURM_TMPDIR/
mkdir -p $SLURM_TMPDIR/slimpajama && cp slimpajama/train/chunk1/example_train_[0-9].jsonl.zst $SLURM_TMPDIR/slimpajama

cp -r Llama-2-7b-hf $SLURM_TMPDIR/

export HF_HUB_OFFLINE=1
export HF_HOME=./huggingface

python main.py \
    --base-dir $SLURM_TMPDIR \
    --model-name Llama-2-7b-hf \
    --dataset-name slimpajama \
    --output-dir $SLURM_TMPDIR/generated_dataset \
    --max-examples -1 \
    --buffer-size 1 \
    --batch-size 1 \
    --max-length 4096 \
    --dtype bfloat16 \
    --disable-datasets-progress

cp -r $SLURM_TMPDIR/generated_dataset ./generated_dataset
