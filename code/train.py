"""
DCDiffusion Training Script

This script trains diffusion models for medical image nuclear segmentation
and classification. Supports multiple datasets (CoNSeP, GLySAC, PUMA) and
implements joint image-label generation.

Usage:
    python train.py --dataset consep --model_type base_256x512 --num_iters 300000
    python train.py --dataset glysac --batch_size 64 --lr 1.5e-4
    python train.py --dataset puma --exp_name my_experiment

Author: DCDiffusion Framework
"""

import argparse
import os
import time

import torch
import torchvision
import torchvision.transforms as T
import wandb
from PIL import Image

# Dataset imports
from datasets.consep_sme import ConsepDataset
from datasets.consep_sme import mapping_id as mapping_id_consep
from datasets.consep_sme import transform_lbl as transform_lbl_consep

from datasets.lizard import LizardDataset
from datasets.lizard import mapping_id as mapping_id_lizard
from datasets.lizard import transform_lbl as transform_lbl_lizard

from datasets.glysac_sme import GlysacDataset
from datasets.glysac_sme import mapping_id as mapping_id_glysac
from datasets.glysac_sme import transform_lbl as transform_lbl_glysac

from datasets.puma_sme import PumaDataset
from datasets.puma_sme import mapping_id as mapping_id_puma
from datasets.puma_sme import transform_lbl as transform_lbl_puma

# Model imports
from imagen_pytorch import BaseJointUnet, JointImagen, JointImagenTrainer


def parse_args():
    """Parse training arguments and configurations."""
    parser = argparse.ArgumentParser(description="Train DCDiffusion models")

    # Training iterations
    parser.add_argument('--start_iters', type=int, default=0)
    parser.add_argument('--num_iters', type=int, default=300000)
    parser.add_argument('--log_every', type=int, default=10000)
    parser.add_argument('--save_every', type=int, default=10000)
    parser.add_argument('--print_every', type=int, default=100)

    # Logging
    parser.add_argument('--log_wandb', action='store_true')
    parser.add_argument('--no_log_wandb', action='store_false', dest='log_wandb')
    parser.set_defaults(log_wandb=True)

    # Training parameters
    parser.add_argument('--lr', type=float, default=1.2e-4)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--max_batch_size', type=int, default=16)
    parser.add_argument('--cond_drop_prob', type=float, default=0.1)
    parser.add_argument('--lambdas', type=float, nargs=2, default=(1., 1.))
    parser.add_argument('--pred_objectives', type=str, default='noise')

    # Experiment configuration
    parser.add_argument('--entity', type=str, default=None)
    parser.add_argument('--project', type=str, default='imagen')
    parser.add_argument('--exp_name', type=str, default='imagen')
    parser.add_argument('--model_type', type=str, default='base_128x256')
    parser.add_argument('--diffusion_type', type=str, default='joint')
    parser.add_argument('--random_crop_size', type=int, nargs='*', default=(256, 512))
    parser.add_argument('--condition_on_text', action='store_true')
    parser.add_argument('--no_condition_on_text', action='store_false', dest='condition_on_text')
    parser.set_defaults(condition_on_text=True)
    parser.add_argument('--lowres_max_thres', type=float, default=0.999)
    parser.add_argument('--augmentation_type', type=str, default='flip')
    parser.add_argument('--noise_schedules', type=str, nargs='*', default=('cosine',))
    parser.add_argument('--noise_schedules_lbl', type=str, nargs='*', default=('cosine_p',))
    parser.add_argument('--cosine_p_lbl', type=float, default=1.0)

    # Model parameters
    parser.add_argument('--channels_lbl', type=int, default=3)
    parser.add_argument('--num_classes', type=int, default=5)

    # Dataset parameters
    parser.add_argument('--dataset', type=str, default='consep',
                       choices=['consep', 'glysac', 'puma', 'lizard'])
    parser.add_argument('--split', type=str, default='100')
    parser.add_argument('--root_dir', type=str, default='')
    parser.add_argument('--caption_list_dir', type=str, default='')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--resume', type=str, default='')

    # Evaluation samples
    parser.add_argument('--test_caption_files', type=str, nargs='*',
                       default=['data/eval_samples/caption.txt'])
    parser.add_argument('--start_image_or_video', type=str, nargs='*',
                       default=['data/eval_samples/image.png'])
    parser.add_argument('--start_label_or_video', type=str, nargs='*',
                       default=['data/eval_samples/label.png'])
    parser.add_argument('--test_batch_size', type=int, default=1)

    # Performance
    parser.add_argument('--fp16', default=True, action='store_true')
    parser.add_argument('--no_fp16', action='store_false', dest='fp16')
    parser.set_defaults(fp16=True)
    parser.add_argument('--num_workers', type=int, default=8)

    args = parser.parse_args()

    # Parse model dimensions
    args.side_x, args.side_y = [int(i) for i in args.model_type.split('_')[-1].split('x')]

    # Handle caption files for non-medical datasets
    if args.dataset not in ["consep", "glysac", "puma", "lizard"]:
        if len(args.test_caption_files) == 1 and args.test_batch_size != 1:
            args.test_caption_files = args.test_caption_files * args.test_batch_size
        args.test_caption = []
        for test_caption_file in args.test_caption_files:
            with open(test_caption_file, 'r') as f:
                args.test_caption.append(f.read())
        assert len(args.test_caption) == args.test_batch_size

    assert args.test_batch_size <= args.max_batch_size

    if len(args.random_crop_size) == 0:
        args.random_crop_size = None
    else:
        assert len(args.random_crop_size) == 2
        args.random_crop_size = tuple(args.random_crop_size)

    # Setup checkpoint directory
    if args.resume:
        args.checkpoint_dir = os.path.dirname(args.resume)
        args.start_iters = int(os.path.basename(args.resume).split('.')[1])
        args.exp_name = '_'.join(os.path.basename(args.checkpoint_dir).split('_')[2:])
    else:
        strtime = time.strftime("%y%m%d_%H%M%S")
        args.checkpoint_dir = os.path.join(args.checkpoint_dir, args.dataset, args.diffusion_type,
                                           args.model_type, f'{strtime}_{args.exp_name}')
        os.makedirs(args.checkpoint_dir, exist_ok=True)

    print(f"Configuration: {args}")
    return args


def main():
    """Main training function."""
    args = parse_args()

    # Create UNet models
    print('Creating JointUNets...')

    if args.model_type.startswith('base'):
        addi_kwargs = dict()
        addi_kwargs.update(dict(
            layer_attns=(False, True, True, True),
            layer_cross_attns=(False, True, True, True)
            if args.condition_on_text else False,
        ))
        unet1 = BaseJointUnet(channels_lbl=args.channels_lbl, num_classes=args.num_classes, **addi_kwargs)
        unets = (unet1,)
        h1, w1 = [int(i) for i in args.model_type.split('_')[1].split('x')]
        image_sizes = ((h1, w1),)
        random_crop_sizes = (None,)
        args.unet_number = 1
    else:
        raise NotImplementedError(f"Model type {args.model_type} not implemented")

    # Create diffusion model
    imagen = JointImagen(
        unets=unets,
        text_encoder_name='t5-large',
        image_sizes=image_sizes,
        random_crop_sizes=random_crop_sizes,
        num_classes=args.num_classes,
        timesteps=1000,
        cond_drop_prob=args.cond_drop_prob,
        condition_on_text=args.condition_on_text,
        lowres_max_thres=args.lowres_max_thres,
        pred_objectives=args.pred_objectives,
        noise_schedules=args.noise_schedules,
        noise_schedules_lbl=args.noise_schedules_lbl,
        cosine_p_lbl=args.cosine_p_lbl,
    )

    # Create trainer
    trainer = JointImagenTrainer(
        imagen,
        lr=args.lr,
        fp16=args.fp16,
        checkpoint_every=args.save_every,
        checkpoint_path=args.checkpoint_dir,
        max_checkpoints_keep=3,
        lambdas=args.lambdas,
    )
    args.world_size = trainer.accelerator.num_processes

    if args.resume:
        trainer.load(args.resume)
    print('Model setup complete!')

    # Setup dataset
    print('Loading dataset...')

    if args.dataset == 'consep':
        dataset = ConsepDataset(
            root=args.root_dir,
            split=args.split,
            side_x=args.side_x,
            side_y=args.side_y,
            augmentation_type=args.augmentation_type,
        )
        mapping_id = mapping_id_consep
        transform_lbl = transform_lbl_consep

    elif args.dataset == 'lizard':
        dataset = LizardDataset(
            root=args.root_dir,
            split=args.split,
            side_x=args.side_x,
            side_y=args.side_y,
            augmentation_type=args.augmentation_type,
        )
        mapping_id = mapping_id_lizard
        transform_lbl = transform_lbl_lizard

    elif args.dataset == 'glysac':
        dataset = GlysacDataset(
            root=args.root_dir,
            split=args.split,
            side_x=args.side_x,
            side_y=args.side_y,
            augmentation_type=args.augmentation_type,
        )
        mapping_id = mapping_id_glysac
        transform_lbl = transform_lbl_glysac

    elif args.dataset == 'puma':
        dataset = PumaDataset(
            root=args.root_dir,
            split=args.split,
            side_x=args.side_x,
            side_y=args.side_y,
            augmentation_type=args.augmentation_type,
        )
        mapping_id = mapping_id_puma
        transform_lbl = transform_lbl_puma

    else:
        raise NotImplementedError(f"Dataset {args.dataset} not implemented")

    trainer.add_train_dataset(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=True,
        shuffle=True,
        pin_memory=True,
    )
    print(f'Dataset {args.dataset} loaded!')

    # Setup start images for cascaded models (not used for single UNet)
    start_image_or_video = start_label_or_video = None

    # Initialize wandb logging
    if trainer.is_main and args.log_wandb:
        wandb.init(
            entity=args.entity,
            project=args.project,
            name=args.exp_name,
            config=args,
            id=os.path.basename(args.checkpoint_dir),
            dir='wandb_dir',
        )

    print('Starting training...')

    # Training loop
    for i in range(args.start_iters, args.num_iters):
        if i % args.print_every == 0 and trainer.is_main:
            print(f'{i} / {args.num_iters}')

        # Training step
        loss, loss_2, loss_seg = trainer.train_step(
            unet_number=args.unet_number,
            max_batch_size=args.max_batch_size
        )

        # Logging
        if trainer.is_main:
            log = {
                f'{args.model_type}_loss': loss,
                f'{args.model_type}_loss_2': loss_2,
                f'{args.model_type}_loss_seg': loss_seg
            }

            if args.log_wandb and (i % args.print_every == 0 or i == 0):
                wandb.log(log, step=i)

            # Generate samples
            if i % args.log_every == 0 or i == 0:
                saved_images, saved_labels, saved_points, saved_texts = trainer.sample(
                    cond_scale=3.,
                    batch_size=args.test_batch_size,
                    start_at_unet_number=args.unet_number,
                    stop_at_unet_number=args.unet_number,
                    start_image_or_video=start_image_or_video,
                    start_label_or_video=start_label_or_video,
                    lowres_sample_noise_level=0.0,
                )

                # Transform labels for visualization
                saved_labels = transform_lbl(saved_labels, 'train_id')
                saved_points = transform_lbl(saved_points, 'train_id')

                # Process distance channels
                saved_distances = saved_images[:, [3, 4]]
                average_channel = torch.mean(saved_distances, dim=1, keepdim=True)
                expanded_distances = average_channel.repeat(1, 3, 1, 1)
                saved_images = saved_images[:, :3]

                # Create visualization grid
                sampled = torchvision.utils.make_grid(
                    torch.cat([saved_images, expanded_distances, saved_labels, saved_points]),
                    nrow=max(4, args.test_batch_size),
                    pad_value=1.
                )

                if args.log_wandb:
                    wandb.log({
                        **log,
                        f'{args.model_type}_samples': wandb.Image(sampled)
                    }, step=i)

    # Finalize wandb logging
    if trainer.is_main and args.log_wandb:
        wandb.finish()

    print('Training complete!')


if __name__ == '__main__':
    main()