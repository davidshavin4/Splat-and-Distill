from mmcv.runner import HOOKS, Hook
import os
import numpy as np
from PIL import Image
import logging
import cv2

def render_segmentation(values, num_classes, colormap="tab20"):
    """
    Render a segmentation map (integer class labels) as a color image.

    Args:
        values (np.ndarray): 2D array of integer class labels.
        colormap (str): Matplotlib colormap name for discrete classes.

    Returns:
        np.ndarray: RGB image (uint8) visualizing the segmentation.
    """
    import matplotlib.pyplot as plt
    values = values.astype(np.int32)    
    cmap = plt.get_cmap(colormap, num_classes)
    colored = cmap(values)[:, :, :3]  # shape (H, W, 3), floats in [0,1]
    return (colored * 255).astype(np.uint8)

@HOOKS.register_module()
class SegmentationVisualizationHook(Hook):
    def __init__(self, interval=1000, out_dir='vis_segmentation'):
        self.interval = interval
        self.out_dir = out_dir

    # def after_iter(self, runner):
    def after_iter(self, runner):
        iter = runner.iter        
        if iter % self.interval != 0:
            return

        outputs = runner.outputs
        if 'log_imgs' not in outputs:
            return
        
        num_classes = outputs['log_imgs']['decode.img_seg_pred'].shape[0] 
        img = outputs['log_imgs']['decode.img_rgb'].transpose(1,2,0) # (1,H,W) or (H,W)
        pred = outputs['log_imgs']['decode.img_seg_pred'].argmax(axis=0).cpu().numpy()  # (N,H,W) -> (H,W)
        gt = outputs['log_imgs']['decode.img_seg_gt'][0].cpu().numpy()

        # Resize pred to match gt shape
        if pred.shape != gt.shape:
            pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)

        # pred = np.clip(pred, self.min_depth, self.max_depth)        
        # gt = np.clip(gt, self.min_depth, self.max_depth)        

        pred_img = render_segmentation(pred, num_classes=num_classes)
        gt_img = render_segmentation(gt, num_classes=num_classes)
        pred_img = (0.75 * pred_img + 0.25 * img).astype(np.uint8)
        gt_img = (0.75 * gt_img + 0.25 * img).astype(np.uint8)

            
        # img = (img * 255).astype(np.uint8)

        vis = np.concatenate((img, gt_img, pred_img), axis=1)
        for handler in runner.logger.handlers:
            if isinstance(handler, logging.FileHandler):
                log_file_path = handler.baseFilename
        out_dir = os.path.join(os.path.dirname(log_file_path), self.out_dir)
        os.makedirs(out_dir, exist_ok=True)
        save_path = os.path.join(out_dir, f"iter_{iter}.png")
        Image.fromarray(vis).save(save_path)
