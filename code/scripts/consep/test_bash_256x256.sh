#!/bin/bash
# CoNSeP Dataset Testing Script
# Base 256x256 model testing

python test.py --model_type=base_256x256 \
    --checkpoint_path checkpoints/consep/joint/base_256x256/model_name/checkpoint.N.pt \
    --root_dir /path/to/consep/data/ \
    --sample_timesteps 100 \
    --start_sample_idx 0 \
    --num_samples=972 \
    --test_batch_size=1 \
    --dataset consep \
    --num_classes 5 \
    --save_path=results/consep/base_256x256_test/ \
    --save_img_path=outputs/consep/base_256x256_test/ \
    --test_captions None
