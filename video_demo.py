# Copyright (c) Meta Platforms, Inc. and affiliates.

import argparse
import os
import sys
import re

import cv2
import mmcv
import numpy as np
import torch

from model.builder import build_model
from mmcv.runner import load_checkpoint
from apis import video_inference


def init_model(config, checkpoint=None, device='cuda:0'):
    """Initialize a model from config file.

    Args:
        config (str or :obj:`mmcv.Config`): Config file path or the config
            object.
        checkpoint (str, optional): Checkpoint path. If left as None, the model
            will not load any weights.
        device (str): Which device the model will deploy. Default: 'cuda:0'.

    Returns:
        nn.Module: The constructed model.
    """
    if isinstance(config, str):
        config = mmcv.Config.fromfile(config)
    elif not isinstance(config, mmcv.Config):
        raise TypeError('config must be a filename or Config object, '
                        f'but got {type(config)}')
    config.model.pretrained = None
    config.test_cfg.metrics = None
    model = build_model(config.model, test_cfg=config.test_cfg)
    if checkpoint is not None:
        if device == torch.device('cpu'):
            checkpoint = load_checkpoint(model, checkpoint, map_location = 'cpu')
        else:
            checkpoint = load_checkpoint(model, checkpoint)
    
    model.cfg = config  # save the config in the model for convenience
    model.to(device)
    model.eval()
    return model


def modify_args():
    for i, v in enumerate(sys.argv):
        if i == 0:
            assert v.endswith('.py')
        elif re.match(r'--\w+_.*', v):
            new_arg = v.replace('_', '-')
            warnings.warn(
                f'command line argument {v} is deprecated, '
                f'please use {new_arg} instead.',
                category=DeprecationWarning,
            )
            sys.argv[i] = new_arg

def parse_args():
    modify_args()
    parser = argparse.ArgumentParser(description='Restoration demo')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('input_dir', help='directory of the input video')
    parser.add_argument('output_dir', help='directory of the output video')
    parser.add_argument(
        '--start-idx',
        type=int,
        default=0,
        help='index corresponds to the first frame of the sequence')
    parser.add_argument(
        '--filename-tmpl',
        default='{:08d}.npy',
        help='template of the file names')
    parser.add_argument(
        '--window-size',
        type=int,
        default=0,
        help='window size if sliding-window framework is used')
    parser.add_argument(
        '--max-seq-len',
        type=int,
        default=None,
        help='maximum sequence length if recurrent framework is used')
    parser.add_argument('--device', type=str, default=0, help='CUDA device id')
    args = parser.parse_args()
    return args


def main():
    """ Demo for dtof video super-resolution models.
    """

    args = parse_args()
    
    if args.device == 'cpu':
        model = init_model(
            args.config, args.checkpoint, device=torch.device('cpu'))
    else:
        model = init_model(
            args.config, args.checkpoint, device=torch.device('cuda', int(args.device)))

    output = video_inference(model, args.input_dir,
                                         args.window_size, args.start_idx,
                                         args.filename_tmpl, args.max_seq_len)

    file_extension = os.path.splitext(args.output_dir)[1]
    
    ## save results
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    for i in range(args.start_idx, args.start_idx + output.size(1)):
        output_i = output[:, i - args.start_idx, :, :, :]
        save_path_i = f'{args.output_dir}/{args.filename_tmpl.format(i)}'
        np.save(save_path_i, output_i.detach().cpu().numpy())

if __name__ == '__main__':
    main()
