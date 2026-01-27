import torch
import torch.nn.functional as F



import torch
import torch.nn.functional as F

def warp_source_to_target(data_dict):
    """
    Project 3D points with associated features onto a camera's sensor, selecting a unique point
    per pixel: the one that is closest (minimal depth), breaking ties by the point's order.
    
    Args:
        data_dict (dict): Contains:
            - extrinsics: Tensor of shape (B, 4, 4)
            - intrinsics: Tensor of shape (B, 3, 3)
            - features:   Tensor of shape (B, C, N)
            - means:      Tensor of shape (B, N, 3)
            - img_size:   Tuple (H, W)
    
    Returns:
        dict: with keys:
            - "warped_features": Tensor of shape (B, C, H, W)
            - "proj_xy":         Tensor of shape (B, 2, N)
            - "mask":            Tensor of shape (B, H, W) (integer mask)
    """
    extrinsics = data_dict["extrinsics"]   # (B, 4, 4)
    intrinsics = data_dict["intrinsics"]     # (B, 3, 3)
    features   = data_dict["features"]       # (B, C, N)
    means      = data_dict["means"]          # (B, N, 3)
    H, W = data_dict["img_size"]             # target image resolution

    B, C, N = features.shape
    eps = 1e-6

    # 1. Transform 3D points into camera space.
    ones = torch.ones((B, N, 1), device=means.device, dtype=means.dtype)
    points_homog = torch.cat([means, ones], dim=-1)  # (B, N, 4)
    extrinsics_inv = torch.inverse(extrinsics)         # (B, 4, 4)
    points_cam = torch.bmm(extrinsics_inv, points_homog.transpose(1, 2))[:, :3, :]  # (B, 3, N)

    # 2. Project points to pixel coordinates.
    scaled_intrinsics = intrinsics.clone()
    scaled_intrinsics[:, 0, :] *= float(W)
    scaled_intrinsics[:, 1, :] *= float(H)
    proj = torch.bmm(scaled_intrinsics, points_cam)       # (B, 3, N)
    proj_xy = proj[:, :2, :] / (proj[:, 2:3, :].clamp(min=eps))  # (B, 2, N)

    # 3. Determine valid points (in front of camera and within image bounds).
    x = proj_xy[:, 0, :]  # (B, N)
    y = proj_xy[:, 1, :]  # (B, N)
    depth = points_cam[:, 2, :]  # (B, N)
    valid = (x >= 0) & (x <= (W - 1)) & (y >= 0) & (y <= (H - 1)) & (depth > eps)  # (B, N)

    # 4. Compute nearest pixel indices.
    nn_x = torch.round(x).long().clamp(0, W - 1)  # (B, N)
    nn_y = torch.round(y).long().clamp(0, H - 1)    # (B, N)
    nn_idx = nn_y * W + nn_x                        # (B, N)

    # 5. For each point, form a combined metric:
    #    - Use effective_depth = depth for valid points (and inf for invalid)
    effective_depth = torch.where(valid, depth, torch.full_like(depth, float('inf')))
    #    - Compute a per-point order to break ties.
    order = torch.arange(N, device=features.device).unsqueeze(0).expand(B, N)
    # Combine depth and order (the order contribution is tiny compared to depth).
    combined_metric = effective_depth + order.float() * 1e-6

    # 6. For each pixel, use scatter_reduce ("amin") over combined_metric to find the best point.
    min_metric = torch.full((B, H * W), float('inf'), device=depth.device, dtype=depth.dtype)
    min_metric = min_metric.scatter_reduce(1, nn_idx, combined_metric, reduce="amin", include_self=True)
    # Gather, for each point, the minimal metric at its assigned pixel.
    min_metric_per_point = min_metric.gather(1, nn_idx)
    # Select the unique point per pixel: it must match the minimal metric.
    selected = (combined_metric == min_metric_per_point)  # (B, N)
    # Ensure only valid points contribute.
    selected = selected & valid

    # 7. Scatter the features of the uniquely selected points into the output image.
    selected_features = features * selected.unsqueeze(1).float()  # (B, C, N)
    warped_features = torch.zeros((B, C, H * W), device=features.device, dtype=features.dtype)
    warped_features = warped_features.scatter_add(2, nn_idx.unsqueeze(1).expand(B, C, N), selected_features)

    # 8. Build a binary mask (1 if a point was assigned).
    mask_flat = torch.zeros((B, H * W), device=features.device, dtype=torch.float32)
    mask_flat = mask_flat.scatter_add(1, nn_idx, selected.float())
    mask_flat = (mask_flat > 0).float()

    # 9. Reshape the outputs.
    warped_features = warped_features.view(B, C, H, W)
    mask = mask_flat.view(B, H, W).int()

    return {"warped_features": warped_features, "proj_xy": proj_xy, "mask": mask}

def warp_source_to_target_slow(data_dict):
    """
    Project 3D points with associated features onto a camera's sensor.
    Instead of accumulating contributions via bilinear splatting, this version
    assigns each pixel the feature from the point that is closest (i.e. with minimal depth).

    Args:
        data_dict (dict): A dictionary with the following keys:
            - extrinsics: Tensor of shape (B, 4, 4) representing camera-to-world poses.
            - intrinsics: Tensor of shape (B, 3, 3) representing normalized camera intrinsics.
            - features:   Tensor of shape (B, C, N) representing point features.
            - means:      Tensor of shape (B, N, 3) containing 3D world coordinates.
            - img_size:   Tuple (H, W) representing the target image resolution.

    Returns:
        A dictionary with:
            - warped_features: Tensor of shape (B, C, H, W) where each pixel has the feature of the closest 3D point.
            - proj_xy:         Tensor of shape (B, 2, N) of the projected 2D pixel coordinates.
            - mask:            Tensor of shape (B, H, W) indicating pixels with an assigned point.
    """
    extrinsics = data_dict["extrinsics"]   # (B, 4, 4)
    intrinsics = data_dict["intrinsics"]     # (B, 3, 3)
    features   = data_dict["features"]       # (B, C, N)
    means      = data_dict["means"]          # (B, N, 3)
    H, W = data_dict["img_size"]             # target image resolution

    B, C, N = features.shape
    eps = 1e-6

    # Convert 3D points to homogeneous coordinates, then transform into camera space.
    ones = torch.ones((B, N, 1), device=means.device, dtype=means.dtype)
    points_homog = torch.cat([means, ones], dim=-1)           # (B, N, 4)
    extrinsics_inv = torch.inverse(extrinsics)                # (B, 4, 4)
    points_cam = torch.bmm(extrinsics_inv, points_homog.transpose(1, 2))  # (B, 4, N)
    points_cam = points_cam[:, :3, :]                          # (B, 3, N)

    # Scale intrinsics from normalized to pixel coordinates.
    scaled_intrinsics = intrinsics.clone()
    scaled_intrinsics[:, 0, :] *= float(W)
    scaled_intrinsics[:, 1, :] *= float(H)

    # Project points from camera coordinates to pixel coordinates.
    proj = torch.bmm(scaled_intrinsics, points_cam)            # (B, 3, N)
    proj_xy = proj[:, :2, :] / (proj[:, 2:3, :].clamp(min=eps))  # (B, 2, N)

    # Determine valid points: in front of camera and within image bounds.
    x = proj_xy[:, 0, :]  # (B, N)
    y = proj_xy[:, 1, :]  # (B, N)
    depth = points_cam[:, 2, :]  # (B, N)
    valid = (x >= 0) & (x <= (W - 1)) & (y >= 0) & (y <= (H - 1)) & (depth > eps)  # (B, N)

    # Compute nearest pixel by rounding.
    nn_x = torch.round(x).long().clamp(0, W - 1)  # (B, N)
    nn_y = torch.round(y).long().clamp(0, H - 1)  # (B, N)
    nn_idx = nn_y * W + nn_x  # (B, N)

    # Initialize the output feature map and a depth buffer.
    warped_features = torch.zeros((B, C, H * W), device=features.device, dtype=features.dtype)
    depth_buffer = torch.full((B, H * W), float('inf'), device=features.device, dtype=depth.dtype)
    mask_flat = torch.zeros((B, H * W), device=features.device, dtype=torch.float32)

    # Loop over each batch and point: assign a point's feature if it is valid and is closer than previously assigned ones.
    for b in range(B):
        for n in range(N):
            if not valid[b, n]:
                continue
            idx = nn_idx[b, n].item()  # convert tensor scalar to Python int
            if depth[b, n] < depth_buffer[b, idx]:
                depth_buffer[b, idx] = depth[b, n]
                warped_features[b, :, idx] = features[b, :, n]
                mask_flat[b, idx] = 1.0

    warped_features = warped_features.view(B, C, H, W)
    mask = mask_flat.view(B, H, W).int()

    return {"warped_features": warped_features, "proj_xy": proj_xy, "mask": mask}


# def warp_source_to_target(data_dict):
#     """
#     Project 3D points with associated features onto a camera's sensor.
#     Additionally, returns a mask that indicates which pixels received contributions.

#     Instead of assuming the features are a dense image grid, we are given N 3D points
#     (with features of shape (B, C, N) and 3D coordinates of shape (B, N, 3)).
#     The function projects these points to image space (using the provided extrinsics and
#     normalized intrinsics, scaled by img_size) and then performs bilinear splatting to
#     accumulate into an output image. Points not within the frustum (or behind
#     the camera) are ignored.

#     Args:
#         data_dict (dict): A dictionary with the following keys:
#             - extrinsics: Tensor of shape (B, 4, 4) representing camera-to-world poses.
#             - intrinsics: Tensor of shape (B, 3, 3) representing normalized camera intrinsics.
#             - features:   Tensor of shape (B, C, N) representing point features.
#             - means:      Tensor of shape (B, N, 3) containing 3D world coordinates.
#             - img_size:   Tuple (H, W) representing the target image resolution.

#     Returns:
#         A dictionary with:
#             - warped_features: Tensor of shape (B, C, H, W) representing the splatted features.
#             - proj_xy:         Tensor of shape (B, 2, N) of the projected 2D pixel coordinates.
#             - mask:            Tensor of shape (B, C, H, W) indicating pixels with contributions.
#     """    

#     extrinsics = data_dict["extrinsics"]   # (B, 4, 4)
#     intrinsics = data_dict["intrinsics"]     # (B, 3, 3)
#     features   = data_dict["features"]       # (B, C, N)
#     means      = data_dict["means"]          # (B, N, 3)
#     H, W = data_dict["img_size"]             # target image resolution

#     B, C, N = features.shape
#     eps = 1e-6

#     # Convert 3D points to homogeneous coordinates, then transform into camera space.
#     ones = torch.ones((B, N, 1), device=means.device, dtype=means.dtype)
#     points_homog = torch.cat([means, ones], dim=-1)           # (B, N, 4)
#     extrinsics_inv = torch.inverse(extrinsics)                # (B, 4, 4)
#     points_cam = torch.bmm(extrinsics_inv, points_homog.transpose(1, 2))  # (B, 4, N)
#     points_cam = points_cam[:, :3, :]                          # (B, 3, N)

#     # Scale intrinsics from normalized to pixel coordinates.
#     scaled_intrinsics = intrinsics.clone()
#     scaled_intrinsics[:, 0, :] *= float(W)
#     scaled_intrinsics[:, 1, :] *= float(H)

#     # Project points from camera coordinates to pixel coordinates.
#     proj = torch.bmm(scaled_intrinsics, points_cam)            # (B, 3, N)
#     proj_xy = proj[:, :2, :] / (proj[:, 2:3, :].clamp(min=eps))  # (B, 2, N)

#     # Determine valid points: in front of camera and within image bounds.
#     x = proj_xy[:, 0, :]  # (B, N)
#     y = proj_xy[:, 1, :]  # (B, N)
#     depth = points_cam[:, 2, :]  # (B, N)
#     valid = (x >= 0) & (x <= (W - 1)) & (y >= 0) & (y <= (H - 1)) & (depth > eps)  # (B, N)

#     # Prepare an output image of zeros for features and mask.
#     warped_features = torch.zeros((B, C, H, W), device=features.device, dtype=features.dtype)
#     mask_accum = torch.zeros((B, 1, H * W), device=features.device, dtype=features.dtype)

#     # For bilinear splatting, compute the integer pixel indices and interpolation weights.
#     x0 = torch.floor(x).long()  # (B, N)
#     y0 = torch.floor(y).long()  # (B, N)
#     x1 = x0 + 1
#     y1 = y0 + 1

#     # Clip indices to image boundaries.
#     x0 = x0.clamp(0, W - 1)
#     x1 = x1.clamp(0, W - 1)
#     y0 = y0.clamp(0, H - 1)
#     y1 = y1.clamp(0, H - 1)

#     # Compute fractional part.
#     dx = (x - x0.float()).clamp(0, 1)  # (B, N)
#     dy = (y - y0.float()).clamp(0, 1)  # (B, N)

#     # Bilinear weights.
#     w00 = (1 - dx) * (1 - dy)  # (B, N)
#     w01 = (1 - dx) * (dy)      # (B, N)
#     w10 = (dx)     * (1 - dy)  # (B, N)
#     w11 = (dx)     * (dy)      # (B, N)

#     # Zero out weights for invalid points.
#     valid = valid.float()  # (B, N)
#     w00 = w00 * valid
#     w01 = w01 * valid
#     w10 = w10 * valid
#     w11 = w11 * valid

#     # For scattering, compute linear indices for each neighbor.
#     def linear_idx(x_idx, y_idx):
#         return y_idx * W + x_idx  # (B, N)

#     idx00 = linear_idx(x0, y0)  # (B, N)
#     idx01 = linear_idx(x0, y1)
#     idx10 = linear_idx(x1, y0)
#     idx11 = linear_idx(x1, y1)

#     # Reshape warped_features to (B, C, H*W) for scatter addition.
#     warped_flat = warped_features.view(B, C, -1)  # (B, C, H*W)

#     # A helper function to scatter-add contributions for features.
#     def scatter_add(weights, idx):
#         # weights: (B, N), idx: (B, N)
#         contrib = features * weights.unsqueeze(1)  # (B, C, N)
#         idx_flat = idx.view(B, -1)  # (B, N)
#         warped_flat.scatter_add_(2, idx_flat.unsqueeze(1).expand(-1, C, -1), contrib)

#     scatter_add(w00, idx00)
#     scatter_add(w01, idx01)
#     scatter_add(w10, idx10)
#     scatter_add(w11, idx11)

#     # Scatter contributions for the mask.
#     # Here, each point contributes the corresponding weight.
#     def scatter_add_mask(weights, idx):
#         contrib = weights.unsqueeze(1)  # (B, 1, N)
#         idx_flat = idx.view(B, -1)  # (B, N)
#         mask_accum.scatter_add_(2, idx_flat.unsqueeze(1), contrib)

#     scatter_add_mask(w00, idx00)
#     scatter_add_mask(w01, idx01)
#     scatter_add_mask(w10, idx10)
#     scatter_add_mask(w11, idx11)

#     # Reshape the feature tensor back to (B, C, H, W).
#     warped_features = warped_flat.view(B, C, H, W)

#     # Build a binary mask: 1 if any contribution, else 0.
#     mask = (mask_accum > 0).float()  # (B, 1, H*W)
#     mask = mask.view(B, H, W).int()
#     # Expand mask to have the same number of channels as warped_features.
#     # mask = mask.expand(B, C, H, W)

#     return {"warped_features": warped_features, "proj_xy": proj_xy, "mask": mask}

import torch

def apply_mask_to_features(features: torch.Tensor, mask: torch.Tensor, mode: str = 'trim') -> torch.Tensor:
    """
    Applies a spatial mask to a feature tensor.

    Args:
        features (torch.Tensor): Tensor of shape (B, C, H, W).
        mask (torch.Tensor): Tensor of shape (B, C, H, W). Assumes that the mask is identical across channels.
        mode (str): 'trim' to use the minimum valid count across batches,
                    'pad' to pad each batch up to the maximum valid count. Default is 'trim'.

    Returns:
        torch.Tensor: Masked features of shape (B, C, n) where n is determined by the chosen mode.
    """
    B, C, H, W = features.shape
    # Flatten spatial dimensions.
    features_flat = features.view(B, C, -1)   # (B, C, H*W)
    # Use the first channel of the mask for selection.
    mask_flat = mask[:, 0, :, :].view(B, -1)    # (B, H*W)

    valid_indices = []
    for b in range(B):
        idx = (mask_flat[b] != 0).nonzero(as_tuple=False).squeeze(1)
        valid_indices.append(idx)

    if mode == 'trim':
        # Use the smallest valid count across batches.
        n_valid = min(idx.numel() for idx in valid_indices)
        trimmed = []
        for b in range(B):
            trimmed.append(features_flat[b, :, valid_indices[b][:n_valid]])
        return torch.stack(trimmed, dim=0)
    elif mode == 'pad':
        # Pad each sample to have the maximum valid count.
        n_valid = max(idx.numel() for idx in valid_indices)
        padded = []
        for b in range(B):
            idx = valid_indices[b]
            num = idx.numel()
            out = features_flat[b, :, idx]
            if num < n_valid:
                pad_tensor = torch.zeros(C, n_valid - num, device=features.device, dtype=features.dtype)
                out = torch.cat([out, pad_tensor], dim=1)
            padded.append(out)
        return torch.stack(padded, dim=0)
    else:
        raise ValueError("mode must be either 'trim' or 'pad'")