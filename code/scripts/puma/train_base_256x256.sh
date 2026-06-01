#!/bin/bash
# PUMA Dataset Training Script
# Base 256x256 model training

export OMP_NUM_THREADS=1
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

accelerate launch --config_file scripts/puma/train_base_256x256.yaml \
    train.py \
    --root_dir /path/to/puma/data/ \
    --caption_list_dir /path/to/captions/puma/ \
    --test_caption_files /path/to/samples/puma_caption.txt \
    --dataset puma \
    --num_classes 4 \
    --exp_name base_256x256 \
    --model_type base_256x256 \
    --num_iters 300000 \
    --log_every 1000 \
    --save_every 10000 \
    --max_batch_size 1 \
    --batch_size 1 \
    --checkpoint_dir checkpoints \
    --test_batch_size 1 \
    --augmentation_type puma \
    --split train \
    --fp16  \
    --num_workers 0 \
    --no_condition_on_text