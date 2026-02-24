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
        "abs_rel",
        "rmse",
        "log_10",
        "rmse_log",
        "silog",
        "sq_rel",
    ]
    # greater_keys = ['mIoU', 'mAcc', 'aAcc']

    def __init__(self, *args, by_epoch=False, pre_eval=False, **kwargs):
        super().__init__(*args, by_epoch=by_epoch, **kwargs)
        self.pre_eval = pre_eval

    def _do_evaluate(self, runner):
        """perform evaluation and save ckpt."""
        if not self._should_evaluate(runner):
            return

        from evaluation.depth.apis import single_gpu_test
        import os
        import numpy as np
        from skimage.transform import resize
        from imageio import imwrite
        from ...engine.hooks.visualization_hook import render_depth
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
            gt = render_depth(log_data['ground_truth'])
            pred = render_depth(log_data['pred'])  


            def _resize_to_hw(x, hw):
                x = np.asarray(x)
                if x.ndim == 2:
                    x = resize(
                        x, hw, preserve_range=True, anti_aliasing=True
                    )
                elif x.ndim == 3:
                    x = resize(
                        x,
                        (hw[0], hw[1], x.shape[2]),
                        preserve_range=True,
                        anti_aliasing=True,
                    )
                else:
                    raise ValueError(f"Unexpected shape {x.shape}")
                return x

            gt = _resize_to_hw(gt, ori_size).astype(np.uint8)
            if pred.ndim == 4 and pred.shape[2] == 1:
                pred = pred.squeeze(2)  # (H, W, 4)
            pred = _resize_to_hw(pred[...,:3], ori_size).astype(np.uint8)

            
            dis_img = np.concatenate([img, gt, pred], axis=1)
            imwrite(os.path.join(eval_img_dir, filename), dis_img)
            
        #TODO: add the logging of the iamges to the work dir, modify single_gpu_test to return some images for logging
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
        "abs_rel",
        "rmse",
        "log_10",
        "rmse_log",
        "silog",
        "sq_rel",
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

        from evaluation.depth.apis import multi_gpu_test

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
