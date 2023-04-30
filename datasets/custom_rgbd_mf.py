# Copyright (c) Meta Platforms, Inc. and affiliates.

import copy
import os.path as osp
import re
import sys

import mmcv
import numpy as np
from mmcv.utils import print_log
from mmseg.datasets import CustomDataset
from .registry import DATASETS
from .pipelines import Compose
from mmseg.utils import get_root_logger
from terminaltables import AsciiTable
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
import os

@DATASETS.register_module()
class CustomRGBDMultiFrameDataset(Dataset):
    def __init__(self,
                 pipeline,
                 guide_folder,
                 gt_folder,
                 split_file,
                 rgb_prefix,
                 rgb_suffix,
                 d_prefix,
                 d_suffix,
                 num_input_frames=None,
                 temp_offset=0,
                 test_mode=True,
                 test_all=True,
                 test_idx_start=0):
        super(CustomRGBDMultiFrameDataset, self).__init__()

        self.pipeline = Compose(pipeline)

        self.guide_folder = str(guide_folder)
        self.gt_folder = str(gt_folder)
        self.split_file = str(split_file)

        self.rgb_prefix = rgb_prefix
        self.rgb_suffix = rgb_suffix
        self.d_prefix = d_prefix
        self.d_suffix = d_suffix

        if num_input_frames is not None and num_input_frames <= 0:
            raise ValueError('"num_input_frames" must be None or positive, '
                             f'but got {num_input_frames}.')
        self.num_input_frames = num_input_frames
        self.temp_offset = temp_offset

        self.test_mode = test_mode
        self.test_all = test_all
        self.test_idx_start = test_idx_start

        self.data_infos = self.load_annotations()

    def get_cont_sub_sequence(
        self, seq_list, out_len, interval = 1
    ):
        """
        This function takes a list of target sequences, and otuputs a continuous list of sub-sequences of desired length
        Input:
          seq_list: [x1, ..., xn]
          out_len: length of the desired output
          interval: the sampling interval for the start of each sub-sequence. default: 1
        Output:
          seq: the list of sub-sequences.
          For example: out_len=2 and interval=1 will return [[x1, x2], [x2, x3], ..., [xn-1, xn]]
        """
        idxs = (
            torch.Tensor(range(len(seq_list)))
            .type(torch.long)
            .view(1, -1)
            .unfold(1, size=out_len, step=interval)
            .squeeze(0)
        )
        seq = []
        for idxSet in idxs:
            inputs = [seq_list[idx_] for idx_ in idxSet]
            seq.append(inputs)
        return seq


    def get_cont_sub_sequence_from_lists(
        self, seq_list, out_len, interval = 1
    ):
        """
        This function takes an input path, and otuputs a continuous list of sub-sequences of the files with desired length
        Input:
          seq_list: a list of sequences list
          out_len: length of the desired output
          interval: the sampling interval for the start of each sub-sequence. default: 1
        Output:
          seq: the list of sub-sequences.
          For example: out_len=2 and interval=1 will return [[x1, x2], [x2, x3], ..., [xn-1, xn]]
        """
        seq = []
        for item in seq_list:
            subs = self.get_cont_sub_sequence(item, out_len, interval)
            seq += subs
        return seq

    def load_annotations(self):
        with open(self.split_file, 'r') as f:
            seqlist = f.readlines()

        gt_seqlist = []
        guide_seqlist = []
        for seq in seqlist:
            seq = seq.split(", ")
            """
            seq: seq[0]: scene name, seq[1]: subscene name,
            seq[2]: number of frames in the subscene
            """
            gt_seqlist.append([])
            guide_seqlist.append([])
            for idx in range(self.temp_offset+1, int(seq[2]) - self.temp_offset - 1):
                gt_seqlist[-1].append(os.path.join(self.gt_folder, seq[0], seq[1], \
                                                   self.d_prefix + '{:06d}'.format(idx) + self.d_suffix))
                guide_seqlist[-1].append(os.path.join(self.guide_folder, seq[0], seq[1], \
                                                   self.rgb_prefix + '{:06d}'.format(idx) + self.rgb_suffix))

        data_infos = {}
        data_infos['gt_path'] = \
        self.get_cont_sub_sequence_from_lists(gt_seqlist, self.num_input_frames + 2*self.temp_offset)
        data_infos['guide_path'] = \
        self.get_cont_sub_sequence_from_lists(guide_seqlist, self.num_input_frames + 2*self.temp_offset)

        if self.test_mode:
            if self.test_all:
                ### maximum 600 images
                data_infos['gt_path'] = data_infos['gt_path'][self.test_idx_start:600:n_frames]
                data_infos['guide_path'] = data_infos['guide_path'][self.test_idx_start:600:n_frames]
            else:
                data_infos['gt_path'] = data_infos['gt_path'][self.test_idx_start:self.test_idx_start+10]
                data_infos['guide_path'] = data_infos['guide_path'][self.test_idx_start:self.test_idx_start+10]

        return data_infos

    def prepare_train_data(self, idx):
        """Prepare training data.

        Args:
            idx (int): Index of the training batch data.

        Returns:
            dict: Returned training batch.
        """
        results = {'sequence_length': self.num_input_frames, \
                   'gt_path': self.data_infos['gt_path'][idx], \
                   'guide_path': self.data_infos['guide_path'][idx]}
        return self.pipeline(results)

    def prepare_test_data(self, idx):
        """Prepare testing data.

        Args:
            idx (int): Index for getting each testing batch.

        Returns:
            Tensor: Returned testing batch.
        """
        results = {'sequence_length': self.num_input_frames, \
                   'gt_path': self.data_infos['gt_path'][idx], \
                   'guide_path': self.data_infos['guide_path'][idx]}
        return self.pipeline(results)

    def __len__(self):
        """Length of the dataset.

        Returns:
            int: Length of the dataset.
        """
        return len(self.data_infos['gt_path'])

    def __getitem__(self, idx):
        """Get item at each call.

        Args:
            idx (int): Index for getting each item.
        """
        if self.test_mode:
            return self.prepare_test_data(idx)

        return self.prepare_train_data(idx)

    def evaluate(self, results, logger=None):
        """Evaluate with different metrics.

        Args:
            results (list[tuple]): The output of forward_test() of the model.

        Return:
            dict: Evaluation results dict.
        """
        if not isinstance(results, list):
            raise TypeError(f'results must be a list, but got {type(results)}')
        assert len(results) == len(self), (
            'The length of results is not equal to the dataset len: '
            f'{len(results)} != {len(self)}')

        results = [res['eval_result'] for res in results]  # a list of dict
        eval_result = defaultdict(list)  # a dict of list

        for res in results:
            for metric, val in res.items():
                eval_result[metric].append(val)
        for metric, val_list in eval_result.items():
            assert len(val_list) == len(self), (
                f'Length of evaluation result of {metric} is {len(val_list)}, '
                f'should be {len(self)}')

        # average the results
        eval_result = {
            metric: sum(values) / len(self)
            for metric, values in eval_result.items()
        }

        return eval_result
