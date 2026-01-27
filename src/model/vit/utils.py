import copy
import torch
import torch.nn.functional as F
from peft import get_peft_model, LoraConfig, PeftModel
from dataclasses import dataclass
from typing import List, Optional, Dict, Any


@dataclass
class CustomLoraConfig:
    r: int
    lora_alpha: int
    target_modules: List[str]
    lora_dropout: float
    bias: str

def resize_pos_embed(pos_embed, old_grid_size, new_img_size, patch_size, num_extra_tokens=1):
    """
    Interpolate ViT positional embeddings to match a new image size.
    
    Args:
        pos_embed (torch.Tensor): Positional embedding of shape [1, N_old, C]
        old_grid_size (int or tuple): Grid size used during pretraining (e.g., 14 or (14,14) for 224x224 and patch_size=16)
        new_img_size (int or tuple): Target image size (e.g., 384 or (384,384))
        patch_size (int): Patch size used by the model (e.g., 16)
        num_extra_tokens (int): Usually 1 (e.g., [CLS]); set to 0 for no extra tokens

    Returns:
        pos_embed_resized (torch.Tensor): Resized positional embedding of shape [1, N_new, C]
    """
    # Separate extra tokens (e.g., class token)
    extra_tokens = pos_embed[:, :num_extra_tokens]
    pos_tokens = pos_embed[:, num_extra_tokens:]

    # Reshape to 2D grid
    H_old, W_old = old_grid_size if isinstance(old_grid_size, tuple) else (old_grid_size, old_grid_size)
    B, N, C = pos_tokens.shape
    pos_tokens = pos_tokens.reshape(B, H_old, W_old, C).permute(0, 3, 1, 2)  # [1, C, H_old, W_old]

    # Interpolate
    H_new, W_new = new_img_size if isinstance(new_img_size, tuple) else (new_img_size, new_img_size)
    new_grid_size = (H_new // patch_size, W_new // patch_size)
    pos_tokens_interp = F.interpolate(pos_tokens, size=new_grid_size, mode='bicubic', align_corners=False)

    # Flatten and concat with extra tokens
    pos_tokens_interp = pos_tokens_interp.permute(0, 2, 3, 1).reshape(B, -1, C)

    return torch.cat((extra_tokens, pos_tokens_interp), dim=1)  # [1, N_new, C]


def apply_lora(model, **kwargs):
    peft_config = LoraConfig(**kwargs)
    lora_model = get_peft_model(model, peft_config)
    return lora_model


def remove_lora(model):
    if isinstance(model, PeftModel):                    
        model = copy.deepcopy(model)
        return model.merge_and_unload()

    else:
        print("Warning: Model is not a PeftModel; nothing to remove.")
        return model
    

def blend_with_mask(features, masks, alpha=0.5):
    """
    For each mask value in masks[b, v], blend features[b, v] using:
        alpha * features + (1 - alpha) * mean(features in region)
    Args:
        features: [B, V, C, H, W]
        masks: [B, V, H, W] (each value in mask is a mask index)
        alpha: float
    Returns:
        blended: [B, V, C, H, W]
    """
    import torch
    import torch.nn.functional as F

    B, V, C, H, W = features.shape
    Hm, Wm = masks.shape[-2:]
    if (Hm, Wm) != (H, W):
        # Merge B and V for resizing
        masks_reshaped = masks.reshape(B * V, 1, Hm, Wm).float()
        masks_resized = F.interpolate(masks_reshaped, size=(H, W), mode='nearest')
        masks = masks_resized.long().reshape(B, V, H, W)

    blended = torch.zeros_like(features)
    for b in range(B):
        for v in range(V):
            mask = masks[b, v]  # [H, W], each value is a mask index (e.g., 0, 1, 2, ...)
            unique_vals = torch.unique(mask)
            feat = features[b, v]  # [C, H, W]
            out = torch.empty_like(feat)
            for val in unique_vals:
                region = (mask == val)  # [H, W]
                if region.sum() == 0:
                    continue
                mean_feat = feat[:, region].mean(dim=1, keepdim=True)  # [C, 1]
                out[:, region] = alpha * feat[:, region] + (1 - alpha) * mean_feat
            blended[b, v] = out
    return blended

def mask_aware_bilinear_upsample_hr_mask(feats_lr, mask_hr, out_h, out_w):
    """
    Args:
        feats_lr: (C, H_lr, W_lr) torch tensor of low-res features
        mask_hr: (out_h, out_w) torch tensor of high-res mask (int)
        out_h, out_w: output resolution

    Returns:
        (C, out_h, out_w) torch tensor
    """
    C, H_lr, W_lr = feats_lr.shape
    device = feats_lr.device

    xs = torch.linspace(0, W_lr - 1, out_w, device=device)
    ys = torch.linspace(0, H_lr - 1, out_h, device=device)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')  # (out_h, out_w)

    # Four neighbors (for bilinear)
    x0 = torch.floor(grid_x).long()
    x1 = torch.clamp(x0 + 1, max=W_lr - 1)
    y0 = torch.floor(grid_y).long()
    y1 = torch.clamp(y0 + 1, max=H_lr - 1)

    wx = grid_x - x0.float()
    wy = grid_y - y0.float()
    w00 = (1 - wx) * (1 - wy)
    w01 = wx * (1 - wy)
    w10 = (1 - wx) * wy
    w11 = wx * wy

    def gather_feats(y_idx, x_idx):
        return feats_lr[:, y_idx, x_idx]  # (C, out_h, out_w)

    # For each HR pixel, determine its segment label from the HR mask
    mask_c = mask_hr  # (out_h, out_w)

    # For each LR grid location, assign it a segment label by nearest upsampling of HR mask
    lr_ys = torch.linspace(0, out_h - 1, H_lr, device=device).round().long().clamp(0, out_h - 1)
    lr_xs = torch.linspace(0, out_w - 1, W_lr, device=device).round().long().clamp(0, out_w - 1)
    mask_lr = mask_hr[lr_ys][:, lr_xs]  # (H_lr, W_lr)

    mask00 = mask_lr[y0, x0]
    mask01 = mask_lr[y0, x1]
    mask10 = mask_lr[y1, x0]
    mask11 = mask_lr[y1, x1]

    valid00 = (mask00 == mask_c).float()
    valid01 = (mask01 == mask_c).float()
    valid10 = (mask10 == mask_c).float()
    valid11 = (mask11 == mask_c).float()

    w00 = w00 * valid00
    w01 = w01 * valid01
    w10 = w10 * valid10
    w11 = w11 * valid11

    w_sum = w00 + w01 + w10 + w11
    w_sum = torch.where(w_sum == 0, torch.ones_like(w_sum), w_sum)
    w00 /= w_sum
    w01 /= w_sum
    w10 /= w_sum
    w11 /= w_sum

    f00 = gather_feats(y0, x0)
    f01 = gather_feats(y0, x1)
    f10 = gather_feats(y1, x0)
    f11 = gather_feats(y1, x1)

    feats_hr = (f00 * w00.unsqueeze(0) +
                f01 * w01.unsqueeze(0) +
                f10 * w10.unsqueeze(0) +
                f11 * w11.unsqueeze(0))
    return feats_hr
