import torch
import torch.nn.functional as F
from einops import rearrange


def resize_batch(batch, size, keys):
    for key in keys:
        images = batch[key]['image']
        b = images.size(0)
        images = rearrange(images, 'b v c h w -> (b v) c h w')
        images = F.interpolate(images, size=size)
        images = rearrange(images, '(b v) c h w -> b v c h w', b=b)
        batch[key]['image'] = images
    return batch



def expand_tensors(data, device):
    """
    Recursively iterates through a dictionary (or list) and expands the first dimension
    of any torch.Tensor found.
    
    Args:
        data (dict or list or any other type): The input data structure.
    
    Returns:
        dict or list: The processed structure with expanded tensors.
    """
    if isinstance(data, dict):
        return {key: expand_tensors(value, device=device) for key, value in data.items()}
    elif isinstance(data, list):
        return [expand_tensors(item, device=device) for item in data]
    elif isinstance(data, torch.Tensor):
        return data.unsqueeze(0).to(device)  # Expands first dimension
    else:
        return data  # Return as is if not a dict, list, or tensor
