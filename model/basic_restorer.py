# Copyright (c) Meta Platforms, Inc. and affiliates.
import numbers
import numpy as np
import os.path as osp

import mmcv
from mmcv.runner import auto_fp16

from .base import BaseModel
from .builder import build_backbone, build_loss
from .registry import MODELS

@MODELS.register_module()
class BasicRestorer(BaseModel):
    """
    Args:
        generator (dict): Config for the generator structure.
        pixel_loss (dict): Config for pixel-wise loss.
        train_cfg (dict): Config for training. Default: None.
        test_cfg (dict): Config for testing. Default: None.
        pretrained (str): Path for pretrained model. Default: None.
    """

    def __init__(self,
                 generator,
                 pixel_loss,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super().__init__()

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        # support fp16
        self.fp16_enabled = False

        # generator
        self.generator = build_backbone(generator)
        self.init_weights(pretrained)

        # loss
        self.pixel_loss = build_loss(pixel_loss)
        self.current_iters = 0

    def init_weights(self, pretrained=None):
        """Init weights for models.

        Args:
            pretrained (str, optional): Path for pretrained weights. If given
                None, pretrained weights will not be loaded. Defaults to None.
        """
        self.generator.init_weights(pretrained)

    @auto_fp16(apply_to=('lq', ))
    def forward(self, lq, guide, gt=None, test_mode=False, **kwargs):
        """Forward function.

        Args:
            guide (Tensor): Input guide (RGB) images
            lq (Tensor): Input lq depth data.
            gt (Tensor): Ground-truth depth map. Default: None.
            test_mode (bool): Whether in test mode or not. Default: False.
            kwargs (dict): Other arguments.
        """

        if test_mode:
            return self.forward_test(lq, guide, gt, **kwargs)

        return self.forward_train(lq, guide, gt)

    def forward_train(self, lq, guide, gt):
        """Training forward function.

        Args:
            lq (Tensor): LQ Tensor with shape (n, c1, h/s, w/s).
            guide (Tensor): Guide Tensor with shape (n, c2, h, w).
            gt (Tensor): GT Tensor with shape (n, c1, h, w).

        Returns:
            Tensor: Output tensor.
        """
        losses = dict()
        output, intermed = self.generator(lq, guide)
        loss_pix = self.pixel_loss(output, gt)
        if self.current_iters <= 50000:
            loss_pix += 0.1 * self.pixel_loss(intermed['d_depth'], gt)
            loss_pix += 0.1 * self.pixel_loss(intermed['rgb_depth'], gt)
        losses['loss_pix'] = loss_pix
        outputs = dict(
            losses=losses,
            num_samples=len(gt.data),
            results=dict(lq=lq.cpu(), guide=guide.cpu(), gt=gt.cpu(), output=output.cpu()))
        self.current_iters += 1
        return outputs

    def evaluate(self, output, gt):
        """Evaluation function.

        Args:
            output (Tensor): Model output with shape (n, c1, h, w).
            gt (Tensor): GT Tensor with shape (n, c1, h, w).

        Returns:
            dict: Evaluation results.
        """
        crop_border = self.test_cfg.crop_border

        ## TODO
        eval_result = dict()
        eval_result['MAE_masked'] = torch.mean(torch.abs(output - gt))
        return eval_result

    def forward_test(self,
                     lq,
                     guide,
                     gt=None,
                     meta=None,
                     save_pred=False,
                     save_path=None,
                     iteration=None):
        """Testing forward function.

        Args:
            lq (Tensor): LQ Tensor with shape (n, c1, h/s, w/s).
            guide (Tensor): Guide Tensor with shape (n, c2, h, w).
            gt (Tensor): GT Tensor with shape (n, c1, h, w). Default: None.
            save_pred (bool): Whether to save predictions. Default: False.
            save_path (str): Path to save image. Default: None.
            iteration (int): Iteration for the saving image name.
                Default: None.

        Returns:
            dict: Output results.
        """
        output, intermed = self.generator(lq, guide)
        if self.test_cfg is not None and self.test_cfg.get('metrics', None):
            assert gt is not None, (
                'evaluation with metrics must have gt images.')
            results = dict(eval_result=self.evaluate(output, gt))
        else:
            results = dict(lq=lq.cpu(), guide = guide.cpu(), output=output.cpu())
            if gt is not None:
                results['gt'] = gt.cpu()

        # save predictions
        if save_pred:
            lq_path = meta[0]['lq_path']
            folder_name = osp.splitext(osp.basename(lq_path))[0]
            if isinstance(iteration, numbers.Number):
                save_path = osp.join(save_path, folder_name,
                                     f'{folder_name}-{iteration + 1:06d}.npy')
            elif iteration is None:
                save_path = osp.join(save_path, f'{folder_name}.npy')
            else:
                raise ValueError('iteration should be number or None, '
                                 f'but got {type(iteration)}')
            np.save(save_path, output.detach().cpu().numpy())

        return results

    ## TODO
    def forward_dummy(self, lq, guide):
        """Used for computing network FLOPs.

        Args:
            lq (Tensor): LQ Tensor with shape (n, c1, h/s, w/s).
            guide (Tensor): Guide Tensor with shape (n, c2, h, w).

        Returns:
            Tensor: Output image.
        """
        out, intermed = self.generator(lq, guide)
        return out

    def train_step(self, data_batch, optimizer):
        """Train step.

        Args:
            data_batch (dict): A batch of data.
            optimizer (obj): Optimizer.

        Returns:
            dict: Returned output.
        """
        outputs = self(**data_batch, test_mode=False)
        loss, log_vars = self.parse_losses(outputs.pop('losses'))

        # optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        outputs.update({'log_vars': log_vars})
        return outputs

    def val_step(self, data_batch, **kwargs):
        """Validation step.

        Args:
            data_batch (dict): A batch of data.
            kwargs (dict): Other arguments for ``val_step``.

        Returns:
            dict: Returned output.
        """
        output = self.forward_test(**data_batch, **kwargs)
        return output
