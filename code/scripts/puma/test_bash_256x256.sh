#!/bin/bash
# PUMA Dataset Testing Script
# Base 256x256 model testing

python test.py --model_type=base_256x256 \
    --checkpoint_path checkpoints/puma/joint/base_256x256/model_name/checkpoint.N.pt \
    --root_dir /path/to/puma/data/ \
    --sample_timesteps 100 \
    --start_sample_idx 0 \
    --num_samples=4100 \
    --test_batch_size=1 \
    --dataset puma \
    --num_classes 4 \
    --save_path=results/puma/base_256x256_test/ \
    --save_img_path=outputs/puma/base_256x256_test/ \
    --test_captions None
