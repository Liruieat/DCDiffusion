#!/bin/bash
# CoNSeP Dataset Training Script
# Base 256x256 model training

export OMP_NUM_THREADS=1
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

accelerate launch --config_file scripts/consep/train_base_256x256.yaml \
    train.py \
    --root_dir /path/to/consep/data/ \
    --caption_list_dir /path/to/captions/consep/ \
    --test_caption_files /path/to/samples/consep_caption.txt \
    --dataset consep \
    --num_classes 5 \
    --exp_name base_256x256 \
    --model_type base_256x256 \
    --num_iters 300000 \
    --log_every 1000 \
    --save_every 10000 \
    --max_batch_size 2 \
    --batch_size 2 \
    --checkpoint_dir checkpoints \
    --test_batch_size 2 \
    --augmentation_type consep \
    --split train \
    --fp16  \
    --num_workers 0 \
    --no_condition_on_text