#!/bin/bash
#SBATCH --account=rrg-mmehride
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=20G
#SBATCH --time=0-20:00:00
#SBATCH --output=%N-%j.out

module load python/3.10.13 gcc/12.3 arrow/18.1.0

virtualenv --no-download $SLURM_TMPDIR/venv
source $SLURM_TMPDIR/venv/bin/activate

pip install --no-index torch transformers datasets accelerate sentencepiece protobuf zstandard

PROJ_DIR=$HOME/llama-experiment

# use model and dataset directly instead of through HF cache (has issues with identifying correct files)
mkdir -p $SLURM_TMPDIR/slimpajama
cp $PROJ_DIR/datasets/slimpajama/example_train_[2-3].jsonl.zst $SLURM_TMPDIR/slimpajama

cp -r $PROJ_DIR/models/Llama-2-7b-hf $SLURM_TMPDIR

export HF_HUB_OFFLINE=1
export HF_HOME=./huggingface

# 10000 examples creates ~1.7TB dataset
python main.py \
    --base-dir $SLURM_TMPDIR \
    --model-name Llama-2-7b-hf \
    --dataset-name slimpajama \
    --output-dir $SLURM_TMPDIR/generated_dataset \
    --max-examples 20000 \
    --buffer-size 1 \
    --batch-size 1 \
    --max-length 4096 \
    --dtype bfloat16 \
    --disable-datasets-progress

PROJ_DIR=$(pwd)
cd $SLURM_TMPDIR/generated_dataset
for i in {0..31}; do
  layer="layer_$i"
  tar -cf "${layer}.tar" "$layer"
  rm -rf "$layer"
done
cp -r $SLURM_TMPDIR/generated_dataset $PROJ_DIR
