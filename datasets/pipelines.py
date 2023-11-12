# Copyright (c) Meta Platforms, Inc. and affiliates.

import copy
import math
import numbers
import os
import os.path as osp
import random

import cv2
import mmcv
from mmcv.utils import build_from_cfg
from mmcv.fileio import FileClient
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from collections.abc import Sequence

from .registry import PIPELINES
    
## TODO: not useful
@PIPELINES.register_module()
class GenerateRGBDSegmentIndices:
    """Generate frame indices for a RGBD segment.

    Required keys: guide_path, gt_path, key, num_input_frames, sequence_length
    Added or modified keys:  guide_path, gt_path, interval, reverse

    Args:
        interval_list (list[int]): Interval list for temporal augmentation.
            It will randomly pick an interval from interval_list and sample
            frame index with the interval.
        start_idx (int): The index corresponds to the first frame in the
            sequence. Default: 0.
        filename_tmpl (str): Template for file name. Default: '{:08d}.png'.
    """
    
    def __init__(self, interval_list, start_idx=0):
        self.interval_list = interval_list
        self.start_idx = start_idx

    def __call__(self, results):
        """Call function.

        Args:
            results (dict): A dict containing the necessary information and
                data for augmentation.

        Returns:
            dict: A dict containing the processed data and information.
        """
        # key example: '000', 'calendar' (sequence name)
        interval = np.random.choice(self.interval_list)
        
        self.sequence_length = results['sequence_length']
        num_input_frames = results.get('num_input_frames',
                                       self.sequence_length)

        # randomly select a frame as start
        if self.sequence_length - num_input_frames * interval < 0:
            raise ValueError('The input sequence is not long enough to '
                             'support the current choice of [interval] or '
                             '[num_input_frames].')
        start_frame_idx = 0
        end_frame_idx = start_frame_idx + num_input_frames * interval
        
        neighbor_list = list(range(start_frame_idx, end_frame_idx, interval))
        neighbor_list = [v + self.start_idx for v in neighbor_list]
        
        # add the corresponding file paths
        guide_path_root = results['guide_path']
        gt_path_root = results['gt_path']
        guide_path = [
            osp.join(guide_path_root, '{:06d}.jpg'.format(v))
            for v in neighbor_list
        ]
        gt_path = [
            osp.join(gt_path_root, '{:06d}.npy'.format(v))
            for v in neighbor_list
        ]

        results['guide_path'] = guide_path
        results['gt_path'] = gt_path
        results['interval'] = interval
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(interval_list={self.interval_list})')
        return repr_str

    
@PIPELINES.register_module()
class LoadImageFromFile:
    """Load image from file.

    Args:
        io_backend (str): io backend where images are store. Default: 'disk'.
        key (str): Keys in results to find corresponding path. Default: 'guide'.
        flag (str): Loading flag for images. Default: 'color'.
        channel_order (str): Order of channel, candidates are 'bgr' and 'rgb'.
            Default: 'bgr'.
        convert_to (str | None): The color space of the output image. If None,
            no conversion is conducted. Default: None.
        save_original_img (bool): If True, maintain a copy of the image in
            `results` dict with name of `f'ori_{key}'`. Default: False.
        use_cache (bool): If True, load all images at once. Default: False.
        backend (str): The image loading backend type. Options are `cv2`,
            `pillow`, and 'turbojpeg'. Default: None.
        kwargs (dict): Args for file client.
    """

    def __init__(self,
                 io_backend='disk',
                 key='gt',
                 flag='color',
                 channel_order='bgr',
                 convert_to=None,
                 save_original_img=False,
                 use_cache=False,
                 backend=None,
                 **kwargs):

        self.io_backend = io_backend
        self.key = key
        self.flag = flag
        self.save_original_img = save_original_img
        self.channel_order = channel_order
        self.convert_to = convert_to
        self.kwargs = kwargs
        self.file_client = None
        self.use_cache = use_cache
        self.cache = None
        self.backend = backend

    def __call__(self, results):
        """Call function.

        Args:
            results (dict): A dict containing the necessary information and
                data for augmentation.

        Returns:
            dict: A dict containing the processed data and information.
        """
        filepath = str(results[f'{self.key}_path'])
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend)
        if self.use_cache:
            if self.cache is None:
                self.cache = dict()
            if filepath in self.cache:
                img = self.cache[filepath]
            else:
                img_bytes = self.file_client.get(filepath)
                img = mmcv.imfrombytes(
                    img_bytes,
                    flag=self.flag,
                    channel_order=self.channel_order,
                    backend=self.backend)  # HWC
                self.cache[filepath] = img
        else:
            img_bytes = self.file_client.get(filepath)
            img = mmcv.imfrombytes(
                img_bytes,
                flag=self.flag,
                channel_order=self.channel_order,
                backend=self.backend)  # HWC

        if self.convert_to is not None:
            if self.channel_order == 'bgr' and self.convert_to.lower() == 'y':
                img = mmcv.bgr2ycbcr(img, y_only=True)
            elif self.channel_order == 'rgb':
                img = mmcv.rgb2ycbcr(img, y_only=True)
            else:
                raise ValueError('Currently support only "bgr2ycbcr" or '
                                 '"bgr2ycbcr".')
            if img.ndim == 2:
                img = np.expand_dims(img, axis=2)

        results[self.key] = img
        results[f'{self.key}_path'] = filepath
        results[f'{self.key}_ori_shape'] = img.shape
        if self.save_original_img:
            results[f'ori_{self.key}'] = img.copy()
        
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (
            f'(io_backend={self.io_backend}, key={self.key}, '
            f'flag={self.flag}, save_original_img={self.save_original_img}, '
            f'channel_order={self.channel_order}, use_cache={self.use_cache})')
        return repr_str


@PIPELINES.register_module()
class LoadImageFromFileList(LoadImageFromFile):
    """Load image from file list.

    It accepts a list of path and read each frame from each path. A list
    of frames will be returned.

    Args:
        io_backend (str): io backend where images are store. Default: 'disk'.
        key (str): Keys in results to find corresponding path. Default: 'guide'.
        flag (str): Loading flag for images. Default: 'color'.
        channel_order (str): Order of channel, candidates are 'bgr' and 'rgb'.
            Default: 'bgr'.
        convert_to (str | None): The color space of the output image. If None,
            no conversion is conducted. Default: None.
        save_original_img (bool): If True, maintain a copy of the image in
            `results` dict with name of `f'ori_{key}'`. Default: False.
        kwargs (dict): Args for file client.
    """

    def __call__(self, results):
        """Call function.

        Args:
            results (dict): A dict containing the necessary information and
                data for augmentation.

        Returns:
            dict: A dict containing the processed data and information.
        """
        
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend)
        filepaths = results[f'{self.key}_path']
        if not isinstance(filepaths, list):
            raise TypeError(
                f'filepath should be list, but got {type(filepaths)}')

        filepaths = [str(v) for v in filepaths]

        imgs = []
        shapes = []
        if self.save_original_img:
            ori_imgs = []
        for filepath in filepaths:
            img_bytes = self.file_client.get(filepath)
            img = mmcv.imfrombytes(
                img_bytes, flag=self.flag,
                channel_order=self.channel_order)  # HWC

            # convert to y-channel, if specified
            if self.convert_to is not None:
                if self.channel_order == 'bgr' and self.convert_to.lower(
                ) == 'y':
                    img = mmcv.bgr2ycbcr(img, y_only=True)
                elif self.channel_order == 'rgb':
                    img = mmcv.rgb2ycbcr(img, y_only=True)
                else:
                    raise ValueError('Currently support only "bgr2ycbcr" or '
                                     '"bgr2ycbcr".')

            if img.ndim == 2:
                img = np.expand_dims(img, axis=2)

            imgs.append(img)
            shapes.append(img.shape)
            if self.save_original_img:
                ori_imgs.append(img.copy())
        
        results[self.key] = imgs
        results[f'{self.key}_path'] = filepaths
        results[f'{self.key}_ori_shape'] = shapes
        if self.save_original_img:
            results[f'ori_{self.key}'] = ori_imgs
        
        return results


@PIPELINES.register_module()
class LoadDFromFileList:
    """Load ground truth (high resolution) depth from file list.

    It accepts a list of path and read each frame from each path. A list
    of frames will be returned.

    Args:
        io_backend (str): io backend where images are store. Default: 'disk'.
        key (str): Keys in results to find corresponding path. Default: 'gt'.
        kwargs (dict): Args for file client.
    """
    
    def __init__(self,
                 io_backend='disk',
                 with_conf=False,
                 key='gt',
                 **kwargs):

        self.io_backend = io_backend
        self.with_conf = with_conf
        self.key = key
        self.kwargs = kwargs
    
    def _scale_depth(self, ds):
        d_scale = 1
        for idx in range(len(ds)):
            ds[idx] = np.clip(ds[idx], 0.0, 40.0)
            if np.quantile(ds[idx], 0.9) >= 40:
                d_scale = 4
                break
            elif np.quantile(ds[idx], 0.9) >= 20:
                d_scale = 2

        for idx in range(len(ds)):
            ds[idx] = ds[idx] / d_scale
            ds[idx] = np.clip(ds[idx], 0.0, 10.0) / 10.0
        return ds
    
    def __call__(self, results):
        """Call function.

        Args:
            results (dict): A dict containing the necessary information and
                data for augmentation.

        Returns:
            dict: A dict containing the processed data and information.
        """
        filepaths = results[f'{self.key}_path']
        if not isinstance(filepaths, list):
            raise TypeError(
                f'filepath should be list, but got {type(filepaths)}')

        filepaths = [str(v) for v in filepaths]
        
        ds = []
        shapes = []
        if self.with_conf:
            confs = []
        for filepath in filepaths:
            d = np.load(filepath)
            ds.append(d)
            shapes.append(d.shape)
            
            if self.with_conf:
                conf = np.load(filepath.replace('depth', 'conf'))
                confs.append(conf)
        
        """
        scale and crop the depth maps to fit into
        the 0-1 range
        """
        ds = self._scale_depth(ds)
        
        results[self.key] = ds
        results[f'{self.key}_path'] = filepaths
        results[f'{self.key}_ori_shape'] = shapes
        
        if self.with_conf:
            results['conf'] = confs
            
        return results
    
    
@PIPELINES.register_module()
class LoadLQDFromFileList:
    """Load low quality (low resolution) depth from file list.

    It accepts a list of path and read each frame from each path. A list
    of frames will be returned.

    Args:
        io_backend (str): io backend where images are store. Default: 'disk'.
        key (str): Keys in results to find corresponding path. Default: 'lq'.
        kwargs (dict): Args for file client.
    """

    def __init__(self,
                 io_backend='disk',
                 key='lq',
                 **kwargs):

        self.io_backend = io_backend
        self.key = key
        self.kwargs = kwargs
    
    def _scale_depth(self, ds):
        d_scale = 1
        for idx in range(len(ds)):
            ds[idx] = np.clip(ds[idx], 0.0, 40.0)
            if np.quantile(ds[idx], 0.9) >= 40:
                d_scale = 4
                break
            elif np.quantile(ds[idx], 0.9) >= 20:
                d_scale = 2

        for idx in range(len(ds)):
            ds[idx] = ds[idx] / d_scale
            ds[idx] = np.clip(ds[idx], 0.0, 10.0) / 10.0
        return ds
    
    def __call__(self, results):
        """Call function.

        Args:
            results (dict): A dict containing the necessary information and
                data for augmentation.

        Returns:
            dict: A dict containing the processed data and information.
        """
        filepaths = results[f'{self.key}_path']
        if not isinstance(filepaths, list):
            raise TypeError(
                f'filepath should be list, but got {type(filepaths)}')
        
        filepaths = [str(v) for v in filepaths]
        
        lqs = []
        shapes = []
        for filepath in filepaths:
            lq = np.load(filepath)
            lqs.append(lq)
            shapes.append(lq.shape)
        
        """
        scale and crop the depth maps to fit into
        the 0-1 range
        """
        lqs = self._scale_depth(lqs)
        
        results[self.key] = lqs
        results[f'{self.key}_path'] = filepaths
        results[f'{self.key}_ori_shape'] = shapes

        return results
    
    
@PIPELINES.register_module()
class LoadHistFromFileList:
    """Load captured/simulated histograms from file list.

    It accepts a list of path and read each frame from each path. A list
    of frames will be returned.
    
    Args:
        io_backend (str): io backend where images are store. Default: 'disk'.
        key (str): Keys in results to find corresponding path. Default: 'hist'.
        kwargs (dict): Args for file client.
    """

    def __init__(self,
                 io_backend='disk',
                 key='hist',
                 **kwargs):

        self.io_backend = io_backend
        self.key = key
        self.kwargs = kwargs
    
    def __call__(self, results):
        """Call function.

        Args:
            results (dict): A dict containing the necessary information and
                data for augmentation.

        Returns:
            dict: A dict containing the processed data and information.
        """
        filepaths = results[f'{self.key}_path']
        if not isinstance(filepaths, list):
            raise TypeError(
                f'filepath should be list, but got {type(filepaths)}')
        
        filepaths = [str(v) for v in filepaths]
        
        hists = []
        shapes = []
        for filepath in filepaths:
            hist = np.load(filepath)
            hists.append(hist)
            shapes.append(hist.shape)
        
        results[self.key] = hists
        results[f'{self.key}_path'] = filepaths
        results[f'{self.key}_ori_shape'] = shapes

        return results
    
    
@PIPELINES.register_module()
class MissingDepth:
    """Randomly Remove Part of the Depth Values
    
    NOTE: the depth value removal is temporally consistent 

    Args:
        ratio: in each frame, generate missing depth value
            positions with probability "ratio"
        drop_ratio: in each frame, drop missing depth value
            positions with probability "drop_ratio"
    """

    def __init__(self,
                 ratio,
                 drop_ratio):
        
        self.ratio = ratio
        self.drop_ratio = drop_ratio
    
    def __call__(self, results):
        """Call function.

        Args:
            results (dict): A dict containing the necessary information and
                data for augmentation.

        Returns:
            dict: A dict containing the processed data and information.
        """
        h, w = results['lq'][0].shape
        missing_x = np.asarray([])
        missing_y = np.asarray([])
        for tt in range(len(results['lq'])):
            missing_ratio = np.random.uniform(0.0, self.ratio*2)
            missing_x = np.append(missing_x, np.random.randint(low=0, high=h, size = int(missing_ratio*h*w)))
            missing_y = np.append(missing_y, np.random.randint(low=0, high=w, size = int(missing_ratio*h*w)))
            results['lq'][tt][missing_x.astype(np.int64), missing_y.astype(np.int64)] = 0.0
            drop = np.random.uniform(self.drop_ratio, 1.0)
            missing_x = missing_x[int(drop*len(missing_x)):]
            missing_y = missing_y[int(drop*len(missing_y)):]
        return results   
    
    
    
@PIPELINES.register_module()
class RandomTempShift:
    """Random Temporal Unsynchronization

    Args:
        maxoffset: maximum temporal offset (unsynchronization)
    """

    def __init__(self,
                 maxoffset):
        self.maxoffset = maxoffset
    
    def __call__(self, results):
        """Call function.

        Args:
            results (dict): A dict containing the necessary information and
                data for augmentation.

        Returns:
            dict: A dict containing the processed data and information.
        """
        if np.random.uniform() < 0.5:
            offset = np.random.randint(-self.maxoffset, -1)
        else:
            offset = np.random.randint(1, self.maxoffset)
        results['gt_misalign'] = results['gt'][self.maxoffset - offset: len(results['gt'])-(self.maxoffset + offset)]
        results['gt'] = results['gt'][self.maxoffset:-self.maxoffset]
        results['guide'] = results['guide'][self.maxoffset:-self.maxoffset]

        return results   
    
    
    
@PIPELINES.register_module()
class RescaleToZeroOne:
    """Transform the images into a range between 0 and 1.

    Required keys are the keys in attribute "keys", added or modified keys are
    the keys in attribute "keys".
    It also supports rescaling a list of images.

    Args:
        keys (Sequence[str]): The images to be transformed.
    """

    def __init__(self, keys):
        self.keys = keys

    def __call__(self, results):
        """Call function.

        Args:
            results (dict): A dict containing the necessary information and
                data for augmentation.

        Returns:
            dict: A dict containing the processed data and information.
        """
        for key in self.keys:
            if isinstance(results[key], list):
                results[key] = [
                    v.astype(np.float32) / 255. for v in results[key]
                ]
            else:
                results[key] = results[key].astype(np.float32) / 255.
        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(keys={self.keys})'
    
    
@PIPELINES.register_module()
class PairedRandomCrop:
    """Paried random crop.

    It crops a pair of lq, guide, and gt with corresponding locations.
    It also supports accepting lq list, guide list and gt list.
    Required keys are "scale", "lq", "guide", and "gt",
    added or modified keys are "lq", "guide", and "gt".

    Args:
        gt_patch_size (int): cropped gt patch size.
    """

    def __init__(self, gt_patch_size):
        self.gt_patch_size = gt_patch_size

    def __call__(self, results):
        """Call function.

        Args:
            results (dict): A dict containing the necessary information and
                data for augmentation.

        Returns:
            dict: A dict containing the processed data and information.
        """
        scale = results['scale']
        lq_patch_size = self.gt_patch_size // scale

        lq_is_list = isinstance(results['lq'], list)
        if not lq_is_list:
            results['lq'] = [results['lq']]
        gt_is_list = isinstance(results['gt'], list)
        if not gt_is_list:
            results['gt'] = [results['gt']]
        
        h_lq, w_lq = results['lq'][0].shape[:2]
        h_gt, w_gt = results['gt'][0].shape

        if h_gt != h_lq * scale or w_gt != w_lq * scale:
            raise ValueError(
                f'Scale mismatches. GT ({h_gt}, {w_gt}) is not {scale}x ',
                f'multiplication of LQ ({h_lq}, {w_lq}).')
        if h_lq < lq_patch_size or w_lq < lq_patch_size:
            raise ValueError(
                f'LQ ({h_lq}, {w_lq}) is smaller than patch size ',
                f'({lq_patch_size}, {lq_patch_size}). Please check '
                f'{results["lq_path"][0]} and {results["gt_path"][0]}.')

        # randomly choose top and left coordinates for lq patch
        top = np.random.randint(h_lq - lq_patch_size + 1)
        left = np.random.randint(w_lq - lq_patch_size + 1)
        # crop lq patch
        results['lq'] = [
            v[top:top + lq_patch_size, left:left + lq_patch_size, ...]
            for v in results['lq']
        ]
        # crop corresponding gt patch
        top_gt, left_gt = int(top * scale), int(left * scale)
        results['gt'] = [
            v[top_gt:top_gt + self.gt_patch_size,
              left_gt:left_gt + self.gt_patch_size, ...] for v in results['gt']
        ]
        results['guide'] = [
            v[top_gt:top_gt + self.gt_patch_size,
              left_gt:left_gt + self.gt_patch_size, ...] for v in results['guide']
        ]
        
        if not lq_is_list:
            results['lq'] = results['lq'][0]
        if not gt_is_list:
            results['gt'] = results['gt'][0]
            results['guide'] = results['guide'][0]
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(gt_patch_size={self.gt_patch_size})'
        return repr_str
                                          
                                          
                                          
@PIPELINES.register_module()
class Flip:
    """Flip the input data with a probability.

    Reverse the order of elements in the given data with a specific direction.
    The shape of the data is preserved, but the elements are reordered.
    Required keys are the keys in attributes "keys", added or modified keys are
    "flip", "flip_direction" and the keys in attributes "keys".
    It also supports flipping a list of images with the same flip.

    Args:
        keys (list[str]): The images to be flipped.
        flip_ratio (float): The propability to flip the images.
        direction (str): Flip images horizontally or vertically. Options are
            "horizontal" | "vertical". Default: "horizontal".
    """
    _directions = ['horizontal', 'vertical']

    def __init__(self, keys, flip_ratio=0.5, direction='horizontal'):
        if direction not in self._directions:
            raise ValueError(f'Direction {direction} is not supported.'
                             f'Currently support ones are {self._directions}')
        self.keys = keys
        self.flip_ratio = flip_ratio
        self.direction = direction

    def __call__(self, results):
        """Call function.

        Args:
            results (dict): A dict containing the necessary information and
                data for augmentation.

        Returns:
            dict: A dict containing the processed data and information.
        """
        flip = np.random.random() < self.flip_ratio

        if flip:
            for key in self.keys:
                if isinstance(results[key], list):
                    for v in results[key]:
                        mmcv.imflip_(v, self.direction)
                else:
                    mmcv.imflip_(results[key], self.direction)

        results['flip'] = flip
        results['flip_direction'] = self.direction

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(keys={self.keys}, flip_ratio={self.flip_ratio}, '
                     f'direction={self.direction})')
        return repr_str
    

@PIPELINES.register_module()
class RandomTransposeHW:
    """Randomly transpose images in H and W dimensions with a probability.

    (TransposeHW = horizontal flip + anti-clockwise rotatation by 90 degrees)
    When used with horizontal/vertical flips, it serves as a way of rotation
    augmentation.
    It also supports randomly transposing a list of images.

    Required keys are the keys in attributes "keys", added or modified keys are
    "transpose" and the keys in attributes "keys".

    Args:
        keys (list[str]): The images to be transposed.
        transpose_ratio (float): The propability to transpose the images.
    """

    def __init__(self, keys, transpose_ratio=0.5):
        self.keys = keys
        self.transpose_ratio = transpose_ratio

    def __call__(self, results):
        """Call function.

        Args:
            results (dict): A dict containing the necessary information and
                data for augmentation.

        Returns:
            dict: A dict containing the processed data and information.
        """
        transpose = np.random.random() < self.transpose_ratio
                
        if transpose:
            for key in self.keys:
                if isinstance(results[key], list):
                    if results[key][0].ndim == 3:
                        results[key] = [v.transpose(1, 0, 2) for v in results[key]]
                    elif results[key][0].ndim == 2:
                        results[key] = [v[...,np.newaxis].transpose(1, 0, 2) for v in results[key]]
                    else:
                        raise ValueError()
                else:
                    if results[key].ndim == 3:
                        results[key] = results[key].transpose(1, 0, 2)
                    elif results[key].ndim == 2:
                        results[key] = results[key][...,np.newaxis].transpose(1, 0, 2)
                    else:
                        raise ValueError()

        results['transpose'] = transpose

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (
            f'(keys={self.keys}, transpose_ratio={self.transpose_ratio})')
        return repr_str


@PIPELINES.register_module()
class ColorJitter:
    """An interface for torch color jitter so that it can be invoked in
    mmseging pipeline.

    Randomly change the brightness, contrast and saturation of an image.
    Modified keys are the attributes specified in "keys".

    Args:
        keys (list[str]): The images to be resized.
        to_rgb (bool): Whether to convert channels from BGR to RGB.
            Default: False.
    """

    def __init__(self, keys, **kwargs):
        assert keys, 'Keys should not be empty.'

        self.keys = keys
        self.transform = transforms.ColorJitter(**kwargs)

    def __call__(self, results):
        for k in self.keys:
            num_frames = len(results[k])
            results_comb = np.concatenate(results[k], axis=0)
            results_comb = Image.fromarray(results_comb)
            results_comb = np.asarray(self.transform(results_comb))
            results[k] = np.split(results_comb, num_frames, axis = 0)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(keys={self.keys}, to_rgb={self.to_rgb})')

        return repr_str
    
    
@PIPELINES.register_module()
class Compose:
    """Compose a data pipeline with a sequence of transforms.

    Args:
        transforms (list[dict | callable]):
            Either config dicts of transforms or transform objects.
    """

    def __init__(self, transforms):
        assert isinstance(transforms, Sequence)
        self.transforms = []
        for transform in transforms:
            if isinstance(transform, dict):
                transform = build_from_cfg(transform, PIPELINES)
                self.transforms.append(transform)
            elif callable(transform):
                self.transforms.append(transform)
            else:
                raise TypeError(f'transform must be callable or a dict, '
                                f'but got {type(transform)}')

    def __call__(self, data):
        """Call function.

        Args:
            data (dict): A dict containing the necessary information and
                data for augmentation.

        Returns:
            dict: A dict containing the processed data and information.
        """
        for t in self.transforms:
            data = t(data)
            if data is None:
                return None
        return data

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += f'    {t}'
        format_string += '\n)'
        return format_string
