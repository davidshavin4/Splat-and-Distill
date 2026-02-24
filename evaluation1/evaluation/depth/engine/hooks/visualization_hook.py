from mmcv.runner import Hook, HOOKS
import os
import numpy as np
from PIL import Image
from mmcv.visualization import color_val
import logging
import cv2

def render_depth(values, colormap="magma_r"):
    import matplotlib.pyplot as plt
    vmin, vmax = values.min(), values.max()
    normed = (values - vmin) / (vmax - vmin + 1e-8)
    cmap = plt.get_cmap(colormap)
    colored = cmap(normed)[:, :, :3]
    return (colored * 255).astype(np.uint8)

@HOOKS.register_module()
class DepthVisualizationHook(Hook):
    def __init__(self, interval=1000, out_dir='vis_depth', min_depth=0., max_depth=10.0):
        self.interval = interval
        self.out_dir = out_dir
        self.min_depth = min_depth
        self.max_depth = max_depth

    # def after_iter(self, runner):
    def after_iter(self, runner):
        iter = runner.iter
        if iter % self.interval != 0:
            return

        outputs = runner.outputs
        if 'log_imgs' not in outputs:
            return
        img = outputs['log_imgs']['decode.img_rgb'].transpose(1,2,0) # (1,H,W) or (H,W)
        pred = outputs['log_imgs']['decode.img_depth_pred'][0].cpu().numpy()
        gt = outputs['log_imgs']['decode.img_depth_gt'][0].cpu().numpy()

        # Resize pred to match gt shape
        if pred.shape != gt.shape:
            pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_LINEAR)

        pred = np.clip(pred, self.min_depth, self.max_depth)        
        gt = np.clip(gt, self.min_depth, self.max_depth)        

        pred_depth = render_depth(pred)
        gt_depth = render_depth(gt)
        # img = (img * 255).astype(np.uint8)

        vis = np.concatenate((img, gt_depth, pred_depth), axis=1)
        for handler in runner.logger.handlers:
            if isinstance(handler, logging.FileHandler):
                log_file_path = handler.baseFilename
        out_dir = os.path.join(os.path.dirname(log_file_path), self.out_dir)
        os.makedirs(out_dir, exist_ok=True)
        save_path = os.path.join(out_dir, f"iter_{iter}.png")
        Image.fromarray(vis).save(save_path)
