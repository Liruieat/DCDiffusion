"""
DCDiffusion Prototype-based Testing Script

This script performs prototype-based sampling for nuclear segmentation
using trained diffusion models. It computes class prototypes and uses them
for conditioned generation.

Usage:
    python test_prototype.py --dataset consep --model_type base_256x512 \
                             --checkpoint_path /path/to/checkpoint \
                             --save_path /path/to/save \
                             --save_img_path /path/to/save_images \
                             --num_samples 50

Author: DCDiffusion Framework
"""

import argparse
import os
import glob
import os.path as osp
import numpy as np

import torch
import torchvision
import torchvision.transforms as T
from torchvision.transforms import ToTensor
from PIL import Image

# Dataset imports
from datasets.consep_sme import transform_lbl as transform_lbl_consep
from datasets.consep_sme import ToTensorNoNorm
from datasets.consep_sme import get_text as get_text_consep

from datasets.glysac_sme import transform_lbl as transform_lbl_glysac
from datasets.glysac_sme import ToTensorNoNorm
from datasets.glysac_sme import get_text as get_text_glysac

from datasets.puma_sme import transform_lbl as transform_lbl_puma
from datasets.puma_sme import ToTensorNoNorm
from datasets.puma_sme import get_text as get_text_puma

# Model imports
from imagen_pytorch import BaseJointUnet, JointImagen
from imagen_pytorch.trainer_test import JointImagenTrainer_Test

from torchsummary import summary
from thop import profile
from thop import clever_format


def read_jsonl(jsonl_path):
    """Read jsonl file and return list of lines."""
    import jsonlines
    lines = []
    with jsonlines.open(jsonl_path, 'r') as f:
        for line in f.iter():
            lines.append(line)
    return lines


def parse_args():
    """Parse testing arguments."""
    parser = argparse.ArgumentParser(description="Test DCDiffusion with prototypes")

    # Required arguments
    parser.add_argument('--model_type', type=str, required=True)
    parser.add_argument('--checkpoint_path', type=str, nargs='+', required=True)
    parser.add_argument('--save_path', type=str, required=True)
    parser.add_argument('--save_img_path', type=str, required=True)

    # Dataset configuration
    parser.add_argument('--dataset', type=str, default='consep',
                       choices=['consep', 'glysac', 'puma', 'lizard'])
    parser.add_argument('--root_dir', type=str, default='')
    parser.add_argument('--split', type=str, default='val')
    parser.add_argument('--num_classes', type=int, default=5)
    parser.add_argument('--start_sample_idx', type=int, default=0)
    parser.add_argument('--end_sample_idx', type=int, default=2975)
    parser.add_argument('--num_samples', type=int, default=50)
    parser.add_argument('--test_batch_size', type=int, default=1)

    # Sampling configuration
    parser.add_argument('--test_captions', type=str, nargs='*', default=['', ])
    parser.add_argument('--caption_list_dir', type=str, default='')
    parser.add_argument('--timesteps', type=int, default=1000)
    parser.add_argument('--sample_timesteps', type=int, default=100)
    parser.add_argument('--cond_scale', type=float, nargs='+', default=(2.0,))
    parser.add_argument('--lowres_sample_noise_level', type=float, default=0.2)
    parser.add_argument('--start_at_unet_number', type=int, default=1)
    parser.add_argument('--stop_at_unet_number', type=int, default=1)
    parser.add_argument('--return_all_unet_outputs', action='store_true')

    # Model configuration
    parser.add_argument('--lowres_dir', type=str, default='')
    parser.add_argument('--noise_schedules', type=str, nargs='*', default=('cosine',))
    parser.add_argument('--noise_schedules_lbl', type=str, nargs='*', default=('cosine_p',))
    parser.add_argument('--cosine_p_lbl', type=float, default=1.0)
    parser.add_argument('--channels_lbl', type=int, default=3)
    parser.add_argument('--pred_objectives', type=str, default='noise')
    parser.add_argument('--cond_drop_prob', type=float, default=0.1)
    parser.add_argument('--condition_on_text', action='store_true')
    parser.add_argument('--no_condition_on_text', action='store_false', dest='condition_on_text')
    parser.set_defaults(condition_on_text=True)

    # Performance
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--no_fp16', action='store_false', dest='fp16')
    parser.set_defaults(fp16=True)
    parser.add_argument('--num_workers', type=int, default=8)

    args = parser.parse_args()

    # Process cond_scale
    args.cond_scale = args.cond_scale[0] if len(args.cond_scale) == 1 else args.cond_scale

    if len(args.test_captions) == 1 and args.test_batch_size != 1:
        args.test_captions = args.test_captions * args.test_batch_size
    assert len(args.test_captions) == args.test_batch_size

    args.end_sample_idx = args.start_sample_idx + args.num_samples
    print(f'Sample Indices: {args.start_sample_idx} - {args.end_sample_idx}')

    return args


def main():
    """Main testing function with prototype computation."""
    args = parse_args()

    print(f'Creating JointUNets for model type: {args.model_type}')

    start_at_unet_number = args.start_at_unet_number
    stop_at_unet_number = args.stop_at_unet_number

    # Create model
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
        args.unet_number = 1
    else:
        raise NotImplementedError(f"Model type {args.model_type} not implemented")

    # Create diffusion model
    imagen = JointImagen(
        unets=unets,
        text_encoder_name='t5-large',
        image_sizes=image_sizes,
        num_classes=args.num_classes,
        timesteps=args.timesteps,
        sample_timesteps=args.sample_timesteps,
        cond_drop_prob=args.cond_drop_prob,
        condition_on_text=args.condition_on_text,
        pred_objectives=args.pred_objectives,
        noise_schedules=args.noise_schedules,
        noise_schedules_lbl=args.noise_schedules_lbl,
        cosine_p_lbl=args.cosine_p_lbl,
    )

    # Create trainer
    trainer = JointImagenTrainer_Test(
        imagen,
        fp16=args.fp16,
        dl_tuple_output_keywords_names=('images', 'labels', 'texts', 'points', 'class_image'),
        split_valid_fraction=1.0,
    )

    trainer.load(args.checkpoint_path[0])

    # Print model parameters
    total = sum([param.nelement() for param in trainer.parameters()])
    print(f'Model parameters: {total/1e6:.2f}M')

    # Setup dataset paths and transforms
    if args.dataset == 'consep':
        transform_lbl = transform_lbl_consep
        dir_points = sorted(glob.glob(os.path.join(args.root_dir, 'points', 'test', '*.png')))
        dir_images = sorted(glob.glob(os.path.join(args.root_dir, 'cls_images', 'test', '*.npy')))
    elif args.dataset == 'glysac':
        transform_lbl = transform_lbl_glysac
        dir_points = sorted(glob.glob(os.path.join(args.root_dir, 'points', 'test', '*.png')))
        dir_images = sorted(glob.glob(os.path.join(args.root_dir, 'cls_images', 'test', '*.npy')))
    elif args.dataset == 'puma':
        transform_lbl = transform_lbl_puma
        dir_points = sorted(glob.glob(os.path.join(args.root_dir, 'points', 'test', '*.png')))
        dir_images = sorted(glob.glob(os.path.join(args.root_dir, 'cls_images', 'test', '*.npy')))
    else:
        raise NotImplementedError(f"Dataset {args.dataset} not implemented")

    print('Dataset setup complete!')

    start_image_or_video, start_label_or_video = None, None

    # Phase 1: Compute class prototypes
    print('Phase 1: Computing prototypes...')

    prototype = None
    counts = [0] * args.num_classes
    n_idx = args.start_sample_idx

    while n_idx < (args.start_sample_idx + args.num_samples):
        os.makedirs(osp.dirname(args.save_path), exist_ok=True)
        os.makedirs(osp.dirname(args.save_img_path), exist_ok=True)

        idx_from = n_idx
        batch_size = 0
        texts = []

        for test_caption in args.test_captions:
            if n_idx >= args.end_sample_idx:
                break
            batch_size += 1
            n_idx += 1

        idx_to = n_idx
        points_pths_name = dir_points[idx_from:idx_to]
        images_pths_name = dir_images[idx_from:idx_to]

        # Count class occurrences
        for point_pth in points_pths_name:
            points = np.unique(Image.open(point_pth))
            for num in points:
                counts[num] += 1

    # Avoid division by zero
    counts = [1 if x == 0 else x for x in counts]

    # Phase 2: Compute prototypes using profiling
    print('Phase 2: Computing prototypes with profiling...')

    n_idx = args.start_sample_idx

    while n_idx < (args.start_sample_idx + args.num_samples):
        idx_from = n_idx
        batch_size = 0
        texts = []

        for test_caption in args.test_captions:
            if n_idx >= args.end_sample_idx:
                break
            batch_size += 1
            n_idx += 1

        idx_to = n_idx
        points_pths_name = dir_points[idx_from:idx_to]
        images_pths_name = dir_images[idx_from:idx_to]

        # Load data
        points = torch.cat([ToTensorNoNorm()(Image.open(point_pth)).float()
                           for point_pth in points_pths_name]).unsqueeze(1)

        if args.test_batch_size == 1:
            class_images = torch.cat([ToTensor()(np.load(image_pth)).float()
                                      for image_pth in images_pths_name]).unsqueeze(0)
        else:
            tensor_list = [torch.from_numpy(np.load(image_pth)).float()
                           for image_pth in images_pths_name]
            class_images = torch.stack(tensor_list, dim=0).permute(0, 3, 1, 2)

        # Generate text descriptions
        if args.dataset == 'consep':
            for i in range(points.size(0)):
                texts.append(get_text_consep(points[i]))
        elif args.dataset == 'glysac':
            for i in range(points.size(0)):
                texts.append(get_text_glysac(points[i]))
        elif args.dataset == 'puma':
            for i in range(points.size(0)):
                texts.append(get_text_puma(points[i]))

        print(f'{n_idx} / {args.end_sample_idx}: {texts}')
        print(f'Sample idx: {idx_from} - {idx_to}')

        # Profile and compute prototype
        MACs, params = profile(trainer.sample_prototype(
            point=points,
            class_image=class_images,
            texts=texts,
            cond_scale=args.cond_scale,
            batch_size=batch_size,
            start_at_unet_number=start_at_unet_number,
            stop_at_unet_number=stop_at_unet_number,
            start_image_or_video=start_image_or_video,
            start_label_or_video=start_label_or_video,
            lowres_sample_noise_level=args.lowres_sample_noise_level,
            return_all_unet_outputs=args.return_all_unet_outputs,
            use_tqdm=True
        ))

        MACs, params = clever_format([MACs, params], '%.3f')
        print(f"Computational cost: {MACs}, Parameters: {params}")

        # Update prototype
        outputs = outputs[0]
        if prototype is None:
            prototype = [proto / counts[idx] for idx, proto in enumerate(outputs)]
        else:
            for point_pth in points_pths_name:
                points = np.unique(Image.open(point_pth))
                for num in points:
                    prototype[num] += outputs[num] / counts[num]

    # Phase 3: Generate samples using prototypes
    print('Phase 3: Generating samples with prototypes...')

    n_idx = args.start_sample_idx

    while n_idx < (args.start_sample_idx + args.num_samples):
        os.makedirs(osp.dirname(args.save_path), exist_ok=True)
        os.makedirs(osp.dirname(args.save_img_path), exist_ok=True)

        idx_from = n_idx
        batch_size = 0
        texts = []

        for test_caption in args.test_captions:
            if n_idx >= args.end_sample_idx:
                break
            batch_size += 1
            n_idx += 1

        idx_to = n_idx
        points_pths_name = dir_points[idx_from:idx_to]
        images_pths_name = dir_images[idx_from:idx_to]

        # Load data
        points = torch.cat([ToTensorNoNorm()(Image.open(point_pth)).float()
                           for point_pth in points_pths_name]).unsqueeze(1)

        if args.test_batch_size == 1:
            class_images = torch.cat([ToTensor()(np.load(image_pth)).float()
                                      for image_pth in images_pths_name]).unsqueeze(0)
        else:
            tensor_list = [torch.from_numpy(np.load(image_pth)).float()
                           for image_pth in images_pths_name]
            class_images = torch.stack(tensor_list, dim=0).permute(0, 3, 1, 2)

        # Generate text descriptions
        if args.dataset == 'consep':
            for i in range(points.size(0)):
                texts.append(get_text_consep(points[i]))
        elif args.dataset == 'glysac':
            for i in range(points.size(0)):
                texts.append(get_text_glysac(points[i]))
        elif args.dataset == 'puma':
            for i in range(points.size(0)):
                texts.append(get_text_puma(points[i]))

        print(f'{n_idx} / {args.end_sample_idx}: {texts}')
        print(f'Sample idx: {idx_from} - {idx_to}')

        # Generate samples with prototype conditioning
        outputs = trainer.sample_prototype_sample(
            point=points,
            class_image=class_images,
            prototype=prototype,
            texts=texts,
            cond_scale=args.cond_scale,
            batch_size=batch_size,
            start_at_unet_number=start_at_unet_number,
            stop_at_unet_number=stop_at_unet_number,
            start_image_or_video=start_image_or_video,
            start_label_or_video=start_label_or_video,
            lowres_sample_noise_level=args.lowres_sample_noise_level,
            return_all_unet_outputs=args.return_all_unet_outputs,
            use_tqdm=True
        )

        if not args.return_all_unet_outputs:
            outputs = [outputs]

        # Save results
        for idx_unet, output in enumerate(outputs):
            saved_images, saved_labels, saved_points = output

            # Convert to numpy
            img = (saved_images[:, :3].squeeze(0).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            dist = (saved_images[:, [3, 4]].squeeze().numpy() * 255).astype(np.uint8)
            lbl = saved_labels.squeeze(0).squeeze(0).numpy().astype(np.uint8)
            pnt = saved_points.squeeze(0).squeeze(0).numpy().astype(np.uint8)

            # Setup save paths
            fn = os.path.basename(points_pths_name[0])
            pth_img = os.path.join(args.save_img_path, 'samples', fn)
            pth_dis = os.path.join(args.save_img_path, 'dists')
            pth_lbl = os.path.join(args.save_img_path, 'labels', fn)
            pth_pnt = os.path.join(args.save_img_path, 'points', fn)

            os.makedirs(os.path.join(args.save_img_path, 'samples'), exist_ok=True)
            os.makedirs(os.path.join(args.save_img_path, 'dists'), exist_ok=True)
            os.makedirs(os.path.join(args.save_img_path, 'labels'), exist_ok=True)
            os.makedirs(os.path.join(args.save_img_path, 'points'), exist_ok=True)

            base_name, _ = os.path.splitext(fn)
            pth_dis = os.path.join(pth_dis, f"{base_name}.npy")
            np.save(pth_dis, dist)

            # Save individual files
            Image.fromarray(img).save(pth_img)
            Image.fromarray(lbl).save(pth_lbl)
            Image.fromarray(pnt).save(pth_pnt)

            # Create visualization
            saved_labels = transform_lbl(saved_labels, 'train_id')
            saved_points = transform_lbl(saved_points, 'train_id')

            saved_distances = saved_images[:, [3, 4]]
            average_channel = torch.mean(saved_distances, dim=1, keepdim=True)
            expanded_distances = average_channel.repeat(1, 3, 1, 1)
            saved_images = saved_images[:, :3]

            save_grid_pth = os.path.join(args.save_path, fn)
            saved_grid = [saved_images, expanded_distances, saved_labels, saved_points]
            torchvision.utils.save_image(
                torch.cat(saved_grid),
                save_grid_pth,
                nrow=max(4, batch_size),
                pad_value=1.
            )
            print(f'{fn} has been saved.')

    print('Testing complete!')


if __name__ == '__main__':
    main()