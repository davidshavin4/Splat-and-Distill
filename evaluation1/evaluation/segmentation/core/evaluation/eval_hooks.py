# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import warnings

import torch.distributed as dist
from mmcv.runner import DistEvalHook as _DistEvalHook
from mmcv.runner import EvalHook as _EvalHook
from torch.nn.modules.batchnorm import _BatchNorm


class EvalHook(_EvalHook):
    """Single GPU EvalHook, with efficient test support.

    Args:
        by_epoch (bool): Determine perform evaluation by epoch or by iteration.
            If set to True, it will perform by epoch. Otherwise, by iteration.
            Default: False.
        pre_eval (bool): Whether to use progressive mode to evaluate model.
            Default: False.
    Returns:
        list: The prediction results.
    """

    metric = [
        "a1",
        "a2",
        "a3",        
        "rmse"
    ]
    # greater_keys = ['mIoU', 'mAcc', 'aAcc']

    def __init__(self, *args, by_epoch=False, pre_eval=False, **kwargs):
        super().__init__(*args, by_epoch=by_epoch, **kwargs)
        self.pre_eval = pre_eval

    def _do_evaluate(self, runner):
        """perform evaluation and save ckpt."""
        if not self._should_evaluate(runner):
            return

        from evaluation.segmentation.apis import single_gpu_test
        import os
        import numpy as np
        from skimage.transform import resize
        from imageio import imwrite
        from ...hooks.visualization_hook import render_segmentation
        num_classes = len(self.dataloader.dataset.CLASSES)
        results, logs = single_gpu_test(
            runner.model, self.dataloader, pre_eval=self.pre_eval
        )
        eval_img_dir = os.path.join(runner.work_dir, f'eval_imgs_{runner.iter+1}')
        os.makedirs(eval_img_dir, exist_ok=True)
        for log_data in logs:
            img = log_data['img']
            filename = log_data['img_metas']['ori_filename'].replace('/', '_')
            ori_size = log_data['img_metas']['ori_shape']
            img = resize(img, ori_size, preserve_range=True, anti_aliasing=True).astype(np.uint8)
            gt = render_segmentation(log_data['ground_truth'], num_classes)
            pred = render_segmentation(log_data['pred'], num_classes)
            gt = (0.75 * gt + 0.25 * img).astype(np.uint8)
            pred = (0.75 * pred + 0.25 * img).astype(np.uint8)
            
            dis_img = np.concatenate([img, gt, pred], axis=1)
            imwrite(os.path.join(eval_img_dir, filename), dis_img)

        runner.log_buffer.clear()
        runner.log_buffer.output["eval_iter_num"] = len(self.dataloader)                
        key_score = self.evaluate(runner, results)
        if self.save_best:
            self._save_ckpt(runner, key_score)


class DistEvalHook(_DistEvalHook):
    """Distributed EvalHook, with efficient test support.

    Args:
        by_epoch (bool): Determine perform evaluation by epoch or by iteration.
            If set to True, it will perform by epoch. Otherwise, by iteration.
            Default: False.
        pre_eval (bool): Whether to use progressive mode to evaluate model.
            Default: False.
    Returns:
        list: The prediction results.
    """

    metric = [
        "a1",
        "a2",
        "a3",        
        "rmse",
    ]
    # greater_keys = ['mIoU', 'mAcc', 'aAcc']

    def __init__(self, *args, by_epoch=False, pre_eval=False, **kwargs):
        super().__init__(*args, by_epoch=by_epoch, **kwargs)
        self.pre_eval = pre_eval

    def _do_evaluate(self, runner):
        """perform evaluation and save ckpt."""
        # Synchronization of BatchNorm's buffer (running_mean
        # and running_var) is not supported in the DDP of pytorch,
        # which may cause the inconsistent performance of models in
        # different ranks, so we broadcast BatchNorm's buffers
        # of rank 0 to other ranks to avoid this.
        if self.broadcast_bn_buffer:
            model = runner.model
            for name, module in model.named_modules():
                if isinstance(module, _BatchNorm) and module.track_running_stats:
                    dist.broadcast(module.running_var, 0)
                    dist.broadcast(module.running_mean, 0)

        if not self._should_evaluate(runner):
            return

        tmpdir = self.tmpdir
        if tmpdir is None:
            tmpdir = osp.join(runner.work_dir, ".eval_hook")

        from evaluation.segmentation.apis import multi_gpu_test

        results = multi_gpu_test(
            runner.model,
            self.dataloader,
            tmpdir=tmpdir,
            gpu_collect=self.gpu_collect,
            pre_eval=self.pre_eval,
        )

        runner.log_buffer.clear()

        if runner.rank == 0:
            print("\n")
            runner.log_buffer.output["eval_iter_num"] = len(self.dataloader)
            key_score = self.evaluate(runner, results)

            if self.save_best:
                self._save_ckpt(runner, key_score)
