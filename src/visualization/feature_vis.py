import torch
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage.transform import resize



def masks_to_color(mask_tensor):
    """
    Convert a mask tensor of shape [b, v, h, w] with values in 0..255
    to a color tensor of shape [b, v, 3, h, w] with values in [0, 1],
    assigning a random color to each unique mask value (consistent per call).
    """
    import torch
    import numpy as np

    b, v, h, w = mask_tensor.shape
    mask_np = mask_tensor.cpu().numpy().astype(np.int32)
    unique_vals = np.unique(mask_np)
    rng = np.random.RandomState(42)
    color_map = rng.rand(unique_vals.max() + 1, 3)  # [num_classes, 3], values in [0,1]

    color_tensor = []
    for bi in range(b):
        color_views = []
        for vi in range(v):
            mask_img = mask_np[bi, vi]  # [h, w]
            color_img = color_map[mask_img]  # [h, w, 3]
            color_img = torch.from_numpy(color_img).permute(2, 0, 1).float()  # [3, h, w]
            color_views.append(color_img)
        color_tensor.append(torch.stack(color_views))  # [v, 3, h, w]
    color_tensor = torch.stack(color_tensor)  # [b, v, 3, h, w]
    return color_tensor.to(mask_tensor.device)


def tensor_pca_normalized(batch, global_pca=True):
    """
    Perform PCA on a batch of feature maps (B, C, H, W) and reduce to (B, 3, H, W).
    Normalizes the output between [0, 1].
    """
    B, C, H, W = batch.shape
    batch_np = batch.cpu().detach().numpy()  # Convert to NumPy for sklearn

    if global_pca:
        # Reshape entire batch to (B*H*W, C) for joint PCA.
        reshaped = batch_np.transpose(0, 2, 3, 1).reshape(-1, C)  # (B*H*W, C)
        pca = PCA(n_components=3)
        reduced = pca.fit_transform(reshaped)  # (B*H*W, 3)
        # Reshape back to (B, 3, H, W)
        reduced = reduced.reshape(B, H, W, 3).transpose(0, 3, 1, 2)
        reduced = (reduced - reduced.min()) / (reduced.max() - reduced.min())
    else:
        # Compute PCA separately for each sample.
        reduced_list = []
        for i in range(B):
            reshaped = batch_np[i].reshape(C, -1).T  # (H*W, C)
            pca = PCA(n_components=3)
            reduced = pca.fit_transform(reshaped)  # (H*W, 3)
            reduced = reduced.T.reshape(3, H, W)  # (3, H, W)
            reduced = (reduced - reduced.min()) / (reduced.max() - reduced.min())
            reduced_list.append(reduced)
        reduced = np.stack(reduced_list)
    return torch.tensor(reduced, dtype=batch.dtype, device=batch.device)

def add_weighted(image, overlay, alpha=0.8, beta=0.2, gamma=0):
    """
    Blend two images together.
    
    Args:
        image (np.ndarray): Source image, shape (H, W, 3), dtype uint8.
        overlay (np.ndarray): Overlay image, shape (H, W, 3), dtype uint8.
        alpha (float): Weight for the source image.
        beta (float): Weight for the overlay.
        gamma (float): Scalar added to the sum.
    
    Returns:
        np.ndarray: Blended image, dtype uint8.
    """
    blended = image.astype(np.float32) * alpha + overlay.astype(np.float32) * beta + gamma
    blended = np.clip(blended, 0, 255)
    return blended.astype(np.uint8)


def tensor_kmean_overlay(batch_features, batch_images, k, global_kmeans=False):
    """
    Perform k-means clustering on a batch of feature maps and overlay the clusters
    on the corresponding images. Optionally, perform k-means clustering globally
    across the entire batch. After blending the overlay with the original image,
    draws borders around each cluster.
    
    Args:
        batch_features (torch.Tensor): Tensor of shape (B, C, H, W).
        batch_images (torch.Tensor): Tensor of shape (B, 3, H, W).
        k (int): Number of clusters.
        global_kmeans (bool): If True, perform k-means clustering on all features in the batch jointly.
    
    Returns:
        list: List of numpy images with k-means clusters overlayed and bordered, each of shape (H, W, 3).
    """
    B, C, H, W = batch_features.shape
    batch_features_np = batch_features.cpu().detach().numpy()
    batch_images_np = (batch_images.cpu().detach().numpy().transpose(0, 2, 3, 1) * 255).astype(np.uint8)
    results = []

    if global_kmeans:
        # Reshape all features to (B*H*W, C)
        all_features = batch_features_np.transpose(0, 2, 3, 1).reshape(-1, C)
        kmeans = KMeans(n_clusters=k, random_state=0).fit(all_features)
        all_labels = kmeans.labels_.reshape(B, H, W)
        colors = np.random.randint(0, 255, size=(k, 3), dtype=np.uint8)
        for i in range(B):
            labels = all_labels[i]
            overlay = np.zeros((H, W, 3), dtype=np.uint8)
            for cluster in range(k):
                overlay[labels == cluster] = colors[cluster]
            image = batch_images_np[i]
            blended = add_weighted(image, overlay, alpha=0.8, beta=0.2)
            border_color = np.array([255, 255, 255], dtype=np.uint8)
            border_thickness = 1
            for cluster in range(k):
                mask = (labels == cluster).astype(np.uint8)
                eroded = ndimage.binary_erosion(mask, structure=np.ones((3, 3))).astype(np.uint8)
                border = mask - eroded
                if border_thickness > 1:
                    border = ndimage.binary_dilation(border, iterations=border_thickness - 1).astype(np.uint8)
                blended[border.astype(bool)] = border_color
            results.append(blended)
    else:
        for i in range(B):
            features = batch_features_np[i].reshape(C, -1).T  # (H*W, C)
            kmeans = KMeans(n_clusters=k, random_state=0).fit(features)
            labels = kmeans.labels_.reshape(H, W)
            overlay = np.zeros((H, W, 3), dtype=np.uint8)
            colors = np.random.randint(0, 255, size=(k, 3), dtype=np.uint8)
            for cluster in range(k):
                overlay[labels == cluster] = colors[cluster]
            image = batch_images_np[i]
            blended = add_weighted(image, overlay, alpha=0.8, beta=0.2)
            border_color = np.array([255, 255, 255], dtype=np.uint8)
            border_thickness = 1
            for cluster in range(k):
                mask = (labels == cluster).astype(np.uint8)
                eroded = ndimage.binary_erosion(mask, structure=np.ones((3, 3))).astype(np.uint8)
                border = mask - eroded
                if border_thickness > 1:
                    border = ndimage.binary_dilation(border, iterations=border_thickness - 1).astype(np.uint8)
                blended[border.astype(bool)] = border_color
            results.append(blended)

    return results
def compute_pca(feature_map):
    """
    Computes a 3-component PCA of a feature map.
    
    Args:
        feature_map (np.ndarray): Array of shape (C, H, W).
    
    Returns:
        np.ndarray: PCA image of shape (H, W, 3) with values between 0 and 1.
    """
    c, h, w = feature_map.shape
    reshaped = feature_map.transpose(1, 2, 0).reshape(-1, c)
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(reshaped)  # Shape: (h*w, 3)
    pca_img = pca_result.reshape(h, w, 3)
    # Normalize to [0, 1]
    min_val = pca_img.min()
    max_val = pca_img.max()
    pca_img = (pca_img - min_val) / (max_val - min_val + 1e-5)
    return pca_img

def visualize_batch(batch, key_list, batch_index):
    """
    For each key in key_list, extracts the dictionary from batch.
    For each of 'student', 'teacher', and 'dino' feature maps (of shape (v, C, H, W))
    at the given batch_index, computes PCA to reduce to 3 channels, resizes the PCA image
    to match the corresponding 'image' shape, and vertically concatenates the original image
    with the three PCA outputs.
    
    Args:
        batch (dict): Batched data.
        key_list (list): List of keys to process.
        batch_index (int): Index for the batch dimension.
    
    Returns:
        dict: Mapping from key to a list of concatenated frames (each of shape (4*h, w, 3)).
    """
    result = {}
    for key in key_list:
        data = batch[key]
        # Feature maps for student, teacher, and dino are expected to have shape (v, C, H, W)
        student_feat = data['student'][batch_index]
        teacher_feat = data['teacher'][batch_index]
        dino_feat = data['dino'][batch_index]
        # Images are assumed to be of shape (v, H, W, 3) in [0, 1]
        images = data['image'][batch_index]
        
        v = student_feat.shape[0]
        frames = []
        for i in range(v):
            pca_student = compute_pca(student_feat[i])
            pca_teacher = compute_pca(teacher_feat[i])
            pca_dino = compute_pca(dino_feat[i])
            
            target_h, target_w = images[i].shape[:2]
            # Resize PCA images using skimage.transform.resize and preserve range.
            pca_student_rs = resize(pca_student, (target_h, target_w), anti_aliasing=True, preserve_range=True).astype(np.float32)
            pca_teacher_rs = resize(pca_teacher, (target_h, target_w), anti_aliasing=True, preserve_range=True).astype(np.float32)
            pca_dino_rs = resize(pca_dino, (target_h, target_w), anti_aliasing=True, preserve_range=True).astype(np.float32)
            
            # Concatenate vertically: original image on top, followed by the PCA visualizations.
            concat_img = np.concatenate([images[i], pca_student_rs, pca_teacher_rs, pca_dino_rs], axis=0)
            frames.append(concat_img)
        
        result[key] = frames
    return result