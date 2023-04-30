# Copyright (c) Meta Platforms, Inc. and affiliates.

from .registry import DATASETS
from .custom_rgbd_mf import CustomRGBDMultiFrameDataset


@DATASETS.register_module()
class TartanAirMultiFrameDataset(CustomRGBDMultiFrameDataset):
    def __init__(self, **kwargs):
        super(TartanAirMultiFrameDataset, self).__init__(
            guide_folder='data/tartanair/color',
            gt_folder = 'data/tartanair/depth',
            split_file = 'data/tartanair/tartanair_train.txt',
            rgb_prefix = 'image_left',
            rgb_suffix = '_left.png',
            d_prefix = 'depth_left',
            d_suffix = '_left_depth.npy',
            **kwargs,
        )
