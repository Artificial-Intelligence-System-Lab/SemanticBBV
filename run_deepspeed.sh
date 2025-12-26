#!/bin/bash

## Check if the number of GPUs is provided as an argument
if [ -z "$1" ]; then
  echo "Please provide the number of GPUs as an argument"
  exit 1
fi

## Check if a checkpoint name is provided (optional)
CHECKPOINT=""
if [ ! -z "$2" ]; then
  CHECKPOINT="--checkpoint $2"
  echo "Resuming training from checkpoint $2"
fi

nvidia-smi

## Create necessary directories
mkdir -p ./checkpoints
mkdir -p ./logs

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11.8/targets/x86_64-linux/lib

## Start training with DeepSpeed using the provided number of GPUs
#--vocab_size 5105
deepspeed --num_gpus=$1 train_rwkv7_deepspeed.py \
    --n_embd 768 \
    --n_layer 6 \
    --head_size 64 \
    --data_dir ./dataset/train/train_dataset.pkl \
    --test_data_dir ./dataset/test/test_dataset.pkl \
    --max_seq_len 512 \
    --min_seq_len 0 \
    --task_type autoregressive \
    --epochs 3 \
    --early_stopping \
    --patience 3 \
    --eval_steps 4000 \
    $CHECKPOINT \
    --deepspeed_config ./ds_config.json
