import torch
import torch.nn as nn
from torch import Tensor

from utils.model_utils import build_2d_model


class FiT3D(nn.Module):
    def __init__(
        self,
        backbone_type,    
    ):
        super().__init__()

        backbone_type_timm = backbone_type.replace('fit3d', 'timm')
        self.vit = build_2d_model(backbone_type_timm)
        
        self.finetuned_model = build_2d_model(backbone_type)
        self.feat_dim = self.vit.num_features + self.finetuned_model.num_features

        
    def forward(
        self,
        x: Tensor,
        out_indices=None,
        output_cls_token=True,
        norm=True,
    ) -> list: 
        if out_indices is None:               
            raise ValueError(f"out_indices should be a list of indices, got {out_indices}.")   
        with torch.no_grad():
            # Extract features and cls tokens from both models
            vit_outputs = self.vit.get_intermediate_layers(
                x,
                n=out_indices if isinstance(out_indices, list) else [out_indices],
                reshape=True,
                return_prefix_tokens=False,
                return_class_token=True,
                norm=norm,
            )  # List of tuples: (features, cls_token)
            finetuned_outputs = self.finetuned_model.get_intermediate_layers(
                x,
                n=out_indices if isinstance(out_indices, list) else [out_indices],
                reshape=True,
                return_prefix_tokens=False,
                return_class_token=True,
                norm=norm,
            )

            result = []
            for (feat1, cls1), (feat2, cls2) in zip(vit_outputs, finetuned_outputs):
                # Concatenate features along channel dimension
                feat_cat = torch.cat([feat1, feat2], dim=1)
                # # Concatenate cls tokens along channel dimension
                # cls_cat = torch.cat([cls1, cls2], dim=1)
                # We'll only pass the output of the first model to avoid redundant cls tokens
                cls_cat = cls1
                if output_cls_token:
                    result.append(tuple([feat_cat, cls_cat]))
                else:
                    result.append(feat_cat)
            
        return result