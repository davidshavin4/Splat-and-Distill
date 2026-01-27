from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol, runtime_checkable
import random
import moviepy.editor as mpy
import torch
import torch.nn.functional as F
from jaxtyping import Float
from pytorch_lightning import LightningModule
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.utilities import rank_zero_only
from torch import Tensor, nn, optim
import numpy as np
import json
import matplotlib.pyplot as plt
import numpy as np
from ..dataset.data_module import get_data_shim
from ..loss import Loss
from ..misc.benchmarker import Benchmarker
from ..misc.step_tracker import StepTracker

from ..visualization.feature_vis import tensor_pca_normalized, tensor_kmean_overlay, masks_to_color

from .decoder.decoder import Decoder, DepthRenderingMode
from .encoder import Encoder
from .encoder.visualization.encoder_visualizer import EncoderVisualizer
from .vit import VisionTransformerBase
from ..geometry.warps import warp_source_to_target       
import matplotlib.pyplot as plt
from src.ludvig.ludvig import render as render_fn
import copy
from src.model.vit.utils import blend_with_mask
import os
from datetime import datetime
from src.model.vit.utils import mask_aware_bilinear_upsample_hr_mask
from einops import rearrange

@dataclass
class OptimizerCfg:
    lr: float
    warm_up_steps: int
    cosine_lr: bool


@dataclass
class TestCfg:
    output_path: Path
    compute_scores: bool
    save_image: bool
    save_video: bool
    eval_time_skip_steps: int


@dataclass
class TrainCfg:
    depth_mode: DepthRenderingMode | None
    extended_visualization: bool
    print_log_every_n_steps: int
    teacher_momentum: float = 0.999
    update_teacher_every_n_steps: int = 10  
    blending_coef: float = 0.5  
    ckpt_interval: int = 10000
    num_training_steps: int = 500000



@runtime_checkable
class TrajectoryFn(Protocol):
    def __call__(
        self,
        t: Float[Tensor, " t"],
    ) -> tuple[
        Float[Tensor, "batch view 4 4"],  # extrinsics
        Float[Tensor, "batch view 3 3"],  # intrinsics
    ]:
        pass


class ModelWrapper(LightningModule):
    logger: Optional[WandbLogger]
    encoder: nn.Module
    encoder_visualizer: Optional[EncoderVisualizer]
    decoder: Decoder
    vit: VisionTransformerBase
    losses: nn.ModuleList
    optimizer_cfg: OptimizerCfg
    test_cfg: TestCfg
    train_cfg: TrainCfg
    step_tracker: StepTracker | None

    def __init__(
        self,
        optimizer_cfg: OptimizerCfg,
        test_cfg: TestCfg,
        train_cfg: TrainCfg,
        encoder: Encoder,
        encoder_visualizer: Optional[EncoderVisualizer],
        decoder: Decoder,
        vit: VisionTransformerBase,
        losses: list[Loss],
        step_tracker: StepTracker | None,
    ) -> None:
        super().__init__()
        self.optimizer_cfg = optimizer_cfg
        self.test_cfg = test_cfg
        self.train_cfg = train_cfg
        self.step_tracker = step_tracker

        # Set up the model.
        self.encoder = encoder
        self.encoder_visualizer = encoder_visualizer
        self.decoder = decoder
        self.vit = vit

        
        self.data_shim = get_data_shim(self.encoder)
        self.losses = nn.ModuleList(losses)

        # This is used for testing.
        self.benchmarker = Benchmarker()
        self.eval_cnt = 0

        if self.test_cfg.compute_scores:
            self.test_step_outputs = {}
            self.time_skip_steps_dict = {"encoder": 0, "decoder": 0}

        
        self.teacher_momentum = train_cfg.teacher_momentum
        self.update_teacher_every_n_steps = train_cfg.update_teacher_every_n_steps    
        self.blending_coef = train_cfg.blending_coef
        self.ckpt_interval = train_cfg.ckpt_interval
        self.num_training_steps = train_cfg.num_training_steps

        

        for name, param in self.named_parameters():
            if 'student' not in name:
                param.requires_grad = False
        
        
        timestamp = datetime.now().strftime("%d%m%Y-%H%M")


        self.experiment = f"experiment/{vit.cfg.name}_{vit.cfg.type}/{timestamp}"        

        os.makedirs(self.experiment, exist_ok=True)
        
    def training_step(self, batch, batch_idx):
        b, v, _, h, w = batch["context"]["image"].shape
        # Run the model.
        gaussians = self.encoder(
            batch["context"], self.global_step, False, scene_names=batch["scene"]
        )
        output = self.decoder.forward(
            gaussians,
            batch["target"]["extrinsics"],
            batch["target"]["intrinsics"],
            batch["target"]["near"],
            batch["target"]["far"],
            (h, w),
            depth_mode=self.train_cfg.depth_mode,
        )
        total_loss = 0.
        
        batch = self.vit.extract_features(batch, keys=['target'], model='student') 

        with torch.no_grad():                
            batch = self.vit.extract_features(batch, keys=['context'], model='teacher')  

            # Now do mask-aware upsampling to H, W for each image in the batch
            upsampled = []
            for i in range(b*v):
                feats_lr = rearrange(batch['context']['teacher'], 'b v c h w -> (b v) c h w')[i]                
                mask_hr = rearrange(batch['context']['segmentation'], 'b v h w -> (b v) h w')[i]
                
                feats_hr = mask_aware_bilinear_upsample_hr_mask(feats_lr, mask_hr, h, w)  # (dim, H, W)
                upsampled.append(feats_hr.unsqueeze(0))
            upsampled = torch.cat(upsampled, dim=0)  # (b*v, dim, H, W)
            upsampled = rearrange(upsampled, '(b v) c h w -> b v c h w', b=b, v=v)
            batch['context']['teacher'] = upsampled

            
            lifted = rearrange(batch['context']['teacher'], "b v c h w -> b (v h w) c")
            ###         
            gaussians.features = lifted
            render = render_fn(gaussians, batch, image_shape=batch['target']['image'].shape[-2:], key='target')               
            render = blend_with_mask(render, batch['target']['segmentation'], alpha=self.blending_coef)            
            render = rearrange(render, "b v c h w -> (b v) c h w")
            render = F.interpolate(render, size=batch['target']['student'].shape[-2:], mode='bilinear')
            render = rearrange(render, "(b v) c h w -> b v c h w", b=b)       
            err_masks = (output.color==0).all(dim=2)[:,:, None, :, :]  # (b, v, 1, h, w) bool
            err_masks = rearrange(err_masks, "b v c h w -> (b v) c h w").float()
            err_masks = F.interpolate(err_masks, size=batch['target']['student'].shape[-2:], mode='nearest')
            err_masks = err_masks >= 0.5  # back to boolean
            err_masks = rearrange(err_masks, "(b v) c h w -> b v c h w", b=b)

        render = render.masked_fill(err_masks.expand_as(render), 0)
        batch['target']['student'] = batch['target']['student'].masked_fill(err_masks.expand_as(batch['target']['student']), 0)
        batch['target']['student'] = batch['target']['student'].masked_fill(err_masks.expand_as(batch['target']['student']), 0)

        feature_loss = self.vit.training_step(render, batch['target']['student'])

        total_loss = feature_loss + total_loss
        

        if (
            self.global_rank == 0
            and self.global_step % self.train_cfg.print_log_every_n_steps == 0
        ):
            print(
                f"train step {self.global_step}; "
                f"scene = {[x[:20] for x in batch['scene']]}; "
                f"context = {batch['context']['index'].tolist()}; "
                f"bound = [{batch['context']['near'].detach().cpu().numpy().mean()} "
                f"{batch['context']['far'].detach().cpu().numpy().mean()}]; "
                f"loss = {total_loss:.6f}"
                f"feature_loss = {feature_loss:.9f}"
            )
        self.log("info/near", batch["context"]["near"].detach().cpu().numpy().mean())
        self.log("info/far", batch["context"]["far"].detach().cpu().numpy().mean())
        self.log("info/global_step", self.global_step)  # hack for ckpt monitor



        if self.global_step%self.ckpt_interval==0 and self.global_step > 0:        
            # 3. Strip LoRA from the model and return the original
            self.vit.save_model(path=f'{self.experiment}/teacher_{self.global_step}.pth',key='teacher')
            self.vit.save_model(path=f'{self.experiment}/student_{self.global_step}.pth', key='student')        
            self.trainer.save_checkpoint(f"{self.experiment}/model_{self.global_step}.ckpt")
        # Tell the data loader processes about the current step.
        if self.step_tracker is not None:
            self.step_tracker.set_step(self.global_step)
        
        if self.global_step==self.num_training_steps:
            raise ValueError('stop training')


        return total_loss
    
    @torch.no_grad()
    def on_after_backward(self):                
        """Update the teacher model using EMA after backpropagation."""            
        if self.global_step%self.update_teacher_every_n_steps==0:                  
            self.vit.update_teacher(self.teacher_momentum)        
            self._updated_teacher_at_step = self.global_step
                    

    def configure_optimizers(self):                
        optimizer = optim.Adam(self.parameters(), lr=self.optimizer_cfg.lr)
        if self.optimizer_cfg.cosine_lr:
            warm_up = torch.optim.lr_scheduler.OneCycleLR(
                            optimizer, self.optimizer_cfg.lr,
                            self.trainer.max_steps + 10,
                            pct_start=0.01,
                            cycle_momentum=False,
                            anneal_strategy='cos',
                        )
        else:
            warm_up_steps = self.optimizer_cfg.warm_up_steps
            warm_up = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                1 / warm_up_steps,
                1,
                total_iters=warm_up_steps,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": warm_up,
                "interval": "step",
                "frequency": 1,
            },
        }
