# Copyright (c) Meta Platforms, Inc. and affiliates.

import math

import numpy as np
import torch
from mmcv import imresize

from .registry import PIPELINES


@PIPELINES.register_module()
class DToFSimulator:
    """Generate dToF synthetic data from GT depth and guide image.

    Args:
        scale (int): The downsampling scale
            Default: 16.0.
        temp_res (int): dToF senor number of time bins.
            Default: 1024
        dtof_sampler (string): dToF operation mode: 
            peak (return peak sampled depth map)/hist (return compressed histogram)
        num_peaks (int): number of peaks to maintain in the rebinned (compressed)
            histogram for HVSR
        threshold (float): threshold to filter out noise/MPI induced artifacts in
            histogram for HVSR
    """

    def __init__(self,
                 scale=16,
                 temp_res=1024,
                 dtof_sampler='peak',
                 with_conf=False,
                 num_peaks = 1,
                 threshold = 0.1, 
                 key='lq'):
        self.scale = scale
        self.temp_res = temp_res
        self.dtof_sampler = dtof_sampler
        self.with_conf = with_conf
        self.num_peaks = num_peaks
        self.threshold = threshold
        self.temp_bins = np.arange(self.temp_res)

    def dtof_hist(self, d, img):
        """
        generate full dToF histogram using Eq.1 in paper
        """
        pitch = self.scale
        temp_res = self.temp_res
        albedo = np.mean(img, axis=2)
        hist = np.zeros((d.shape[0] // pitch, d.shape[1] // pitch, temp_res))
        for ii in range(d.shape[0] // pitch):
            for jj in range(d.shape[1] // pitch):
                ch, cw = ii * pitch, jj * pitch
                albedo_block = albedo[ch : ch + pitch, cw : cw + pitch]
                d_block = d[ch : ch + pitch, cw : cw + pitch]
                idx = np.round(d_block * (temp_res - 1)).reshape(-1)
                r = (albedo_block / (1e-3 + d_block**2)).reshape(-1)
                r[d_block.reshape(-1) == 0] = 0
                idx = np.concatenate((idx, np.asarray([0, temp_res - 1]))).astype(
                    np.int64
                )
                r = np.concatenate((r, np.asarray([0, 0]))).astype(np.float32)
                hist[ii, jj] = np.bincount(idx, weights=r)
        return hist
        
    def rebin_hist(self, hist):
        """
        compress histogram with rebinning 
        based on both uniform interval divisions and local peaks
        """
        hist[hist <= 10] = 0
        if np.max(hist) == 0:
            return (
                np.zeros(self.num_peaks),
                np.zeros(self.num_peaks * 2 + 3),
                np.zeros(self.num_peaks * 2 + 3).astype(np.int64),
            )
        threshold = np.max(hist) * self.threshold
        idx_start = np.min(np.argwhere(hist >= threshold))
        idx_end = np.max(np.argwhere(hist >= threshold)) + 1

        idx_end_round = (
            idx_start + ((idx_end - idx_start) // self.num_peaks + 1) * self.num_peaks
        )
        idx_start_round = (
            idx_end - ((idx_end - idx_start) // self.num_peaks + 1) * self.num_peaks
        )
        if idx_end_round <= hist.shape[0]:
            idx_end = idx_end_round
        elif idx_start_round >= 0:
            idx_start = idx_start_round
        else:
            raise ValueError("not well-defined histogram shape")
        rebin_idx = (
            np.arange(self.num_peaks + 1) / (self.num_peaks) * (idx_end - idx_start)
            + idx_start
        ).astype(np.int64)

        hist_split = hist[idx_start:idx_end].reshape(self.num_peaks, -1)
        mpeaks = np.argmax(hist_split, axis=1) + rebin_idx[:-1]
        rebin_idx_all = np.zeros(self.num_peaks * 2 + 1)
        rebin_idx_all[::2] = rebin_idx.copy()
        rebin_idx_all[1::2] = mpeaks.copy()

        rpeaks = hist[mpeaks]
        mpeaks[rpeaks == 0] = 0
        mpeaks = mpeaks[np.argsort(rpeaks)[::-1]]

        cdf = np.cumsum(np.insert(hist, 0, 0))
        rebin_idx_all = np.clip(rebin_idx_all, 0.0, self.temp_res - 1)
        rebin_idx_all = np.insert(
            rebin_idx_all, [0, rebin_idx_all.shape[0]], [0, self.temp_res]
        ).astype(np.int64)
        cdf_rebin = cdf[rebin_idx_all]
        return mpeaks, cdf_rebin, rebin_idx_all

    def dtof_mpeak(self, d, img):
        """
        generate compressed (rebinned) dToF histogram
        """
        pitch = self.scale
        temp_res = self.temp_res
        albedo = np.mean(img, axis=2)
        hist_full = np.zeros((d.shape[0] // pitch, d.shape[1] // pitch, temp_res))
        d_mpeak = np.zeros((d.shape[0] // pitch, d.shape[1] // pitch, self.num_peaks))
        cdf_mpeak = np.zeros(
            (d.shape[0] // pitch, d.shape[1] // pitch, self.num_peaks * 2 + 3)
        )
        rebin_mpeak = np.zeros(
            (d.shape[0] // pitch, d.shape[1] // pitch, self.num_peaks * 2 + 3)
        )
        for ii in range(d.shape[0] // pitch):
            for jj in range(d.shape[1] // pitch):
                ch, cw = ii * pitch, jj * pitch
                albedo_block = albedo[ch : ch + pitch, cw : cw + pitch]
                d_block = d[ch : ch + pitch, cw : cw + pitch]
                idx = np.round(d_block * (temp_res - 1)).reshape(-1)
                r = (albedo_block / (1e-3 + d_block**2)).reshape(-1)
                r[d_block.reshape(-1) == 0] = 0
                idx = np.concatenate((idx, np.asarray([0, temp_res - 1]))).astype(
                    np.int64
                )
                r = np.concatenate((r, np.asarray([0, 0]))).astype(np.float32)
                hist_full[ii, jj] = np.bincount(idx, weights=r)
                (
                    d_mpeak[ii, jj],
                    cdf_mpeak[ii, jj],
                    rebin_mpeak[ii, jj],
                ) = self.rebin_hist(hist_full[ii, jj])
        lq_comb = np.concatenate([d_mpeak, cdf_mpeak, rebin_mpeak], axis = 2)
        return lq_comb
        
        
    def __call__(self, results):
        """Call function.

        Args:
            results (dict): A dict containing the necessary information and
                data for augmentation. 'gt' is required.

        Returns:
            dict: A dict containing the processed data and information.
                modified 'gt', supplement 'lq' and 'scale' to keys.
        """
        ds = results['gt']
        imgs = results['guide']
        
        if self.with_conf:
            if (self.dtof_sampler == 'peak') and \
                ('conf' in results) and \
                (results['conf'] is not None):
                confs = results['conf']
            else:
                raise ValueError()
        
        if self.dtof_sampler == 'peak':
            ## peak mode, only use peak depth as input
            if not self.with_conf:
                results['lq'] = [
                    np.argmax(self.dtof_hist(d, img), axis=2)/ (self.temp_res - 1)
                    for (d, img) in zip(ds, imgs)
                ]
            
            else:
                results['lq'] = [
                        np.argmax(self.dtof_hist(d, img * \
                                    (np.tile(conf[..., np.newaxis], (1,1,3)) + 0.01)), axis=2)/ (self.temp_res - 1)
                        for (d, img, conf) in zip(ds, imgs, confs)
                ]
        elif self.dtof_sampler == 'mpeak':
            ## multi-peak mode, use multiple depth peaks as input
            results['lq'] = [self.dtof_mpeak(d, img) for (d, img) in zip(ds, imgs)]
        elif self.dtof_sampler == 'rebin':
            ## compressed histogram mode
            results['lq'] = []
            for kk in range(len(results['hist'])):
                d_mpeak = np.zeros((results['hist'][0].shape[0], \
                                    results['hist'][0].shape[1], self.num_peaks))
                cdf_mpeak = np.zeros(
                    (results['hist'][0].shape[0], \
                     results['hist'][0].shape[1], self.num_peaks * 2 + 3)
                )
                rebin_mpeak = np.zeros(
                    (results['hist'][0].shape[0], \
                     results['hist'][0].shape[1], self.num_peaks * 2 + 3)
                )
                for ii in range(results['hist'][0].shape[0]):
                    for jj in range(results['hist'][0].shape[1]):
                        (
                        d_mpeak[ii, jj],
                        cdf_mpeak[ii, jj],
                        rebin_mpeak[ii, jj],
                    ) = self.rebin_hist(results['hist'][kk][ii, jj])
                lq_comb = np.concatenate([d_mpeak, cdf_mpeak, rebin_mpeak], axis = 2)
                results['lq'].append(lq_comb)    
        else:
            raise ValueError()
        
        results['scale'] = self.scale
        
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'scale={self.scale}, '
                     f'temp_res={self.temp_res}, '
                     f'dtof_sampler={self.dtof_sampler}, ')

        return repr_str
