

import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_
from torch.nn.utils import weight_norm
import torch.nn.functional as F
from .vision_transformer import VisionTransformerBase, VisionTransformerBaseCfg
from .utils import apply_lora, remove_lora
import copy
from functools import partial
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any
from einops import rearrange
import numpy as np
from .utils import CustomLoraConfig as LoraConfig

imagenet_mean = torch.tensor([0.485, 0.456, 0.406], device='cuda').view(1, 3, 1, 1)
imagenet_std = torch.tensor([0.229, 0.224, 0.225], device='cuda').view(1, 3, 1, 1)


################################################################################
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.
################################################################################

class DINOHead(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        patch_loss=False,
        use_bn=False,
        nlayers=3,
        hidden_dim=2048,
        bottleneck_dim=256,
        mlp_bias=True,
    ):
        super().__init__()
        self.patch_loss = patch_loss
        nlayers = max(nlayers, 1)
        self.mlp = _build_mlp(nlayers, in_dim, bottleneck_dim, hidden_dim=hidden_dim, use_bn=use_bn, bias=mlp_bias)
        self.apply(self._init_weights)
        if not self.patch_loss:
            self.last_layer = weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))        
            self.last_layer.weight_g.data.fill_(1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        eps = 1e-6 if x.dtype == torch.float16 else 1e-12
        x = nn.functional.normalize(x, dim=-1, p=2, eps=eps)
        if not self.patch_loss:
            x = self.last_layer(x)    
        return x




def _build_mlp(nlayers, in_dim, bottleneck_dim, hidden_dim=None, use_bn=False, bias=True):
    if nlayers == 1:
        return nn.Linear(in_dim, bottleneck_dim, bias=bias)
    else:
        layers = [nn.Linear(in_dim, hidden_dim, bias=bias)]
        if use_bn:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.GELU())
        for _ in range(nlayers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim, bias=bias))
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
        layers.append(nn.Linear(hidden_dim, bottleneck_dim, bias=bias))
        return nn.Sequential(*layers)
    
################################################################################

@dataclass
class DinoHeadConfig:
    head_n_prototypes: int
    head_bottleneck_dim: int
    head_nlayers: int
    head_hidden_dim: int
    
@dataclass
class StudentConfig:
    arch: str
    patch_size: int
    student_temp: float

@dataclass
class TeacherConfig:
    momentum_teacher: float
    final_momentum_teacher: float
    teacher_temp: float

@dataclass
class DinoV2Cfg(VisionTransformerBaseCfg):
    type: str
    dino: DinoHeadConfig = field(default_factory=DinoHeadConfig)
    lora: Optional[LoraConfig] = None
    student: StudentConfig = field(default_factory=StudentConfig)
    teacher: TeacherConfig = field(default_factory=TeacherConfig)

class DinoV2(VisionTransformerBase):
    def __init__(self, cfg: DinoV2Cfg):
        super().__init__()
        self.cfg = cfg

        
        embed_dim = self.cfg.embed_dim
        student_model_dict = dict()
        teacher_model_dict = dict()


        # student_backbone, teacher_backbone, embed_dim = build_model_from_cfg(cfg)
        student_backbone = torch.hub.load(f'facebookresearch/{cfg.name}:main', f'{cfg.name}_{cfg.type}{cfg.student.patch_size}')
        dino_backbone = torch.hub.load(f'facebookresearch/{cfg.name}:main', f'{cfg.name}_{cfg.type}{cfg.student.patch_size}')

        self.use_lora = False
        if not cfg.lora is None:
            self.use_lora = True
            student_backbone = apply_lora(student_backbone, **asdict(cfg.lora))
        teacher_backbone = copy.deepcopy(student_backbone)
        student_model_dict["backbone"] = student_backbone
        teacher_model_dict["backbone"] = teacher_backbone
        
        self.embed_dim = embed_dim

        dino_head = partial(
            DINOHead,
            in_dim=embed_dim,
            out_dim=cfg.dino.head_n_prototypes,
            hidden_dim=cfg.dino.head_hidden_dim,
            bottleneck_dim=cfg.dino.head_bottleneck_dim,
            nlayers=cfg.dino.head_nlayers,
        )

        student_model_dict["dino_head"] = dino_head()
        teacher_model_dict["dino_head"] = dino_head()
        teacher_model_dict["dino_head"].load_state_dict(student_model_dict["dino_head"].state_dict())


        
        self.student = nn.ModuleDict(student_model_dict)
        self.teacher = nn.ModuleDict(teacher_model_dict)        


        self.dino_backbone = dino_backbone
        self.feature_channels = self.embed_dim
        self.patch_size = cfg.student.patch_size

        # there is no backpropagation through the teacher, so no need for gradients
        for p in self.teacher.parameters():
            p.requires_grad = False        



    def extract_features(self, batch, keys=['target'], model='teacher', scale_factor=1):
        if model=='teacher':
            vit = self.teacher['backbone']
        elif model=='student':
            vit = self.student['backbone']
        elif model=='ori':
            vit = self.dino_backbone
        else:
            print('model has to be either student, teacher or ori')
            return

        for key in keys:
            images = batch[key]['image'].clone()
            b, _, _, h, w = images.shape            
            images = rearrange(images, 'b v c h w -> (b v) c h w')
            H, W = h-(h%14), w-(w%14)
            if not scale_factor is None:       
                H,W = H*scale_factor, W*scale_factor         
            images = F.interpolate(images, size=(H, W), mode='bilinear')
            images = (images - imagenet_mean) / imagenet_std
            output = vit.forward_features(images)
            local_tokens = output["x_norm_patchtokens"]  # shape: (1, num_tokens, dim)                                

            local_tokens = rearrange(local_tokens, "(b v) (h w) c -> b v c h w", b=b, h=int(H/14))                
            batch[key][model] = local_tokens
        return batch    

    def training_step(self, teacher_features, student_features):

        teacher_features = rearrange(teacher_features, 'b v c h w -> (b v h w) c') # teacher_features.flatten(0, -2)
        student_features = rearrange(student_features, 'b v c h w -> (b v h w) c') # student_features.flatten(0, -2)

        teacher_features = self.teacher['dino_head'](teacher_features)    # [B, P, H, W] teacher prototypes (no grad)
        student_features = self.student['dino_head'](student_features)    # [B, P, H, W] student prototypes
        # Compute probability distributions 
        teacher_features = teacher_features.float()
        student_features = student_features.float()
        # Teacher: softmax to get probabilities (detach to avoid grad)
        teacher_features = F.softmax(teacher_features, dim=1).detach()
        student_features = F.log_softmax(student_features, dim=1)
        
        # Loss: cross-entropy between teacher_probs and student probabilities at each spatial location
        # Compute per-pixel cross-entropy: -sum(p * log q) over prototype dimension
        loss_map = -(teacher_features * student_features).sum(dim=1)   # shape [B, H, W]
        loss = loss_map.mean()  # average over all pixels and batch
        return loss


    def update_teacher(self, new_momentum: float = None):
        """
        Update teacher model parameters as an EMA of the student parameters.
        new_momentum: if provided, override the current momentum for this update.
        """                
        m = new_momentum if new_momentum is not None else self.current_momentum
        # Clamp momentum between 0 and 1
        m = max(0, min(1, m))
        for key in self.teacher.keys():
            for param_t, param_s in zip(self.teacher[key].parameters(), self.student[key].parameters()):
                param_t.data = param_t.data * m + param_s.data * (1 - m)

    def save_model(self, path, key):

        # save student and teacher 
        if key=='student':
            model = self.student['backbone']
        elif key=='teacher':
            model = self.teacher['backbone']
        else:
            raise ValueError("key must be either 'student' or 'teacher'")
        model = remove_lora(model)
        torch.save(model.state_dict(), path)
    
