# Copyright (c) Meta Platforms, Inc. and affiliates.
import glob
import os
import os.path as osp
import re
from functools import reduce

import mmcv
import numpy as np
import torch

from datasets import Compose

def pad_sequence(data, window_size):
    padding = window_size // 2

    data = torch.cat([
        data[:, 1 + padding:1 + 2 * padding].flip(1), data,
        data[:, -1 - 2 * padding:-1 - padding].flip(1)
    ],
                     dim=1)

    return data


def video_inference(model,
                                root_dir,
                                window_size,
                                start_idx,
                                filename_tmpl,
                                max_seq_len=None):
    """Inference image with the model.

    Args:
        model (nn.Module): The loaded model.
        root_dir (str): Directory of the input video.
        window_size (int): The window size used in sliding-window framework.
            This value should be set according to the settings of the network.
            A value smaller than 0 means using recurrent framework.
        start_idx (int): The index corresponds to the first frame in the
            sequence.
        filename_tmpl (str): Template for file name.
        max_seq_len (int | None): The maximum sequence length that the model
            processes. If the sequence length is larger than this number,
            the sequence is split into multiple segments. If it is None,
            the entire sequence is processed at once.

    Returns:
        Tensor: The predicted restoration result.
    """

    device = next(model.parameters()).device  # model device

    # build the data pipeline
    if model.cfg.get('demo_pipeline', None):
        test_pipeline = model.cfg.demo_pipeline
    elif model.cfg.get('test_pipeline', None):
        test_pipeline = model.cfg.test_pipeline
    else:
        test_pipeline = model.cfg.val_pipeline

    # specify start_idx and filename_tmpl
    test_pipeline[0]['start_idx'] = start_idx

    # prepare data
    sequence_length = len(glob.glob(osp.join(root_dir, 'color', '*')))
    root_dir_split = re.split(r'[\\/]', root_dir)
    gt_folder = [osp.join(root_dir, 'depth', f) for f in sorted(os.listdir(osp.join(root_dir, 'depth')))]
    guide_folder = [osp.join(root_dir, 'color', f) for f in sorted(os.listdir(osp.join(root_dir, 'color')))]
    data = dict(
        guide_path=guide_folder,
        gt_path=gt_folder,
        sequence_length=sequence_length)

    # compose the pipeline
    test_pipeline = Compose(test_pipeline)
    data = test_pipeline(data)
    lqs = data['lq'].unsqueeze(0)  # in cpu
    guides = data['guide'].unsqueeze(0)
    # forward the model
    with torch.no_grad():
        if window_size > 0:  # sliding window framework
            lqs = pad_sequence(lqs, window_size)
            guides = pad_sequence(guides, window_size)
            result = []
            for i in range(0, lqs.size(1) - 2 * (window_size // 2)):
                lq_i = lqs[:, i:i + window_size].to(device)
                guide_i = guides[:, i:i + window_size].to(device)
                result.append(model(lq=lq_i, guide=guide_i, test_mode=True)['output'].cpu())
            result = torch.stack(result, dim=1)
        else:  # recurrent framework
            if max_seq_len is None:
                result = model(
                    lq=lqs.to(device), guide=guides.to(device), test_mode=True)['output'].cpu()
            else:
                result = []
                for i in range(0, lqs.size(1), max_seq_len):
                    result.append(
                        model(
                            lq=lqs[:, i:i + max_seq_len].to(device),
                            guide=guides[:, i:i + max_seq_len].to(device),
                            test_mode=True)['output'].cpu())
                result = torch.cat(result, dim=1)
    return result
