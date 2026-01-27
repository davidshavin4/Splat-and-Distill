import math
import numpy as np
import torch

from tqdm import tqdm
from src.geometry.projection import get_fov, homogenize_points
from src.model.decoder.cuda_splatting import get_projection_matrix
from src.model.encoder.common.gaussian_adapter import Gaussians

from src.model.encoder.common.gaussian_adapter import Gaussians

from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)
from einops import rearrange

C0 = 0.28209479177387814
C1 = 0.4886025119029199
C2 = [
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396
]
C3 = [
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435
]
C4 = [
    2.5033429417967046,
    -1.7701307697799304,
    0.9461746957575601,
    -0.6690465435572892,
    0.10578554691520431,
    -0.6690465435572892,
    0.47308734787878004,
    -1.7701307697799304,
    0.6258357354491761,
]   



def is_valid_rotation_matrix(R):
    Rt = R.transpose(-2, -1)
    should_be_identity = torch.matmul(Rt, R)
    I = torch.eye(3, device=R.device).expand_as(should_be_identity)
    error = (should_be_identity - I).abs().max(dim=-1)[0].max(dim=-1)[0]
    det = R.det()
    return (error < 1e-3) & (det > 0)


def rotation_matrix_to_quaternion(R):
    # R is expected to have shape (*, 3, 3)
    batch_shape = R.shape[:-2]
    R_orig_flat = R.reshape(-1, 3, 3) # Original R_flat for shape and full iteration
    num_matrices = R_orig_flat.shape[0]

    # Initialize output quaternions with a default (identity [1,0,0,0] for w,x,y,z)
    all_quats = torch.zeros((num_matrices, 4), dtype=R.dtype, device=R.device)
    if num_matrices > 0:
        all_quats[:, 0] = 1.0 

    valid_mask = is_valid_rotation_matrix(R_orig_flat)

    # If there are no valid matrices to process, return the default quaternions
    if not valid_mask.any():
        # Optional: print a message if debugging this scenario
        # if num_matrices > 0:
        #     print("[INFO] No valid rotation matrices found in rotation_matrix_to_quaternion. Returning default quaternions.")
        return all_quats.reshape(*batch_shape, 4)

    # Process only the valid rotation matrices
    R_flat_valid = R_orig_flat[valid_mask]
    
    # Create a temporary tensor for computed quaternions for valid matrices
    # This tensor will be placed into all_quats at valid_mask indices
    quats_for_valid_Rs = torch.empty((R_flat_valid.shape[0], 4), dtype=R.dtype, device=R.device)

    trace = R_flat_valid[:, 0, 0] + R_flat_valid[:, 1, 1] + R_flat_valid[:, 2, 2]
    eps = 1e-6  # Small value to avoid sqrt(negative) or division by zero

    # Case 1: trace > 0
    # This mask is relative to R_flat_valid
    mask_trace_pos = trace > 0.0 
    if mask_trace_pos.any():
        R_subset = R_flat_valid[mask_trace_pos]
        trace_subset = trace[mask_trace_pos]
        
        t = torch.sqrt(torch.clamp(trace_subset + 1.0, min=eps))
        # Ensure t is not zero for division by creating a safe_t
        safe_t = t.where(t > eps, torch.full_like(t, eps))

        quats_for_valid_Rs[mask_trace_pos, 0] = 0.5 * t
        quats_for_valid_Rs[mask_trace_pos, 1] = (R_subset[:, 2, 1] - R_subset[:, 1, 2]) / (2.0 * safe_t)
        quats_for_valid_Rs[mask_trace_pos, 2] = (R_subset[:, 0, 2] - R_subset[:, 2, 0]) / (2.0 * safe_t)
        quats_for_valid_Rs[mask_trace_pos, 3] = (R_subset[:, 1, 0] - R_subset[:, 0, 1]) / (2.0 * safe_t)

    # Case 2: trace <= 0
    # This mask is relative to R_flat_valid
    mask_trace_non_pos = ~mask_trace_pos 
    if mask_trace_non_pos.any():
        # m is the subset of R_flat_valid where trace <= 0
        m = R_flat_valid[mask_trace_non_pos] 
        
        # Temporary tensor for quaternions computed in this branch, matching m's size
        # This will be assigned to the corresponding slice of quats_for_valid_Rs
        quats_for_m_branch = torch.empty((m.shape[0], 4), dtype=R.dtype, device=R.device)

        cond0 = (m[..., 0, 0] >= m[..., 1, 1]) & (m[..., 0, 0] >= m[..., 2, 2])
        cond1 = ~cond0 & (m[..., 1, 1] >= m[..., 2, 2])
        cond2 = ~(cond0 | cond1)

        # m00 largest
        if cond0.any():
            R_c0 = m[cond0]
            t0 = torch.sqrt(torch.clamp(1.0 + R_c0[:, 0, 0] - R_c0[:, 1, 1] - R_c0[:, 2, 2], min=eps))
            safe_t0 = t0.where(t0 > eps, torch.full_like(t0, eps))
            quats_for_m_branch[cond0, 0] = (R_c0[:, 2, 1] - R_c0[:, 1, 2]) / (2.0 * safe_t0)
            quats_for_m_branch[cond0, 1] = 0.5 * t0
            quats_for_m_branch[cond0, 2] = (R_c0[:, 0, 1] + R_c0[:, 1, 0]) / (2.0 * safe_t0)
            quats_for_m_branch[cond0, 3] = (R_c0[:, 0, 2] + R_c0[:, 2, 0]) / (2.0 * safe_t0)

        # m11 largest
        if cond1.any():
            R_c1 = m[cond1]
            t1 = torch.sqrt(torch.clamp(1.0 + R_c1[:, 1, 1] - R_c1[:, 0, 0] - R_c1[:, 2, 2], min=eps))
            safe_t1 = t1.where(t1 > eps, torch.full_like(t1, eps))
            quats_for_m_branch[cond1, 0] = (R_c1[:, 0, 2] - R_c1[:, 2, 0]) / (2.0 * safe_t1)
            quats_for_m_branch[cond1, 1] = (R_c1[:, 0, 1] + R_c1[:, 1, 0]) / (2.0 * safe_t1)
            quats_for_m_branch[cond1, 2] = 0.5 * t1
            quats_for_m_branch[cond1, 3] = (R_c1[:, 1, 2] + R_c1[:, 2, 1]) / (2.0 * safe_t1)

        # m22 largest
        if cond2.any():
            R_c2 = m[cond2]
            t2 = torch.sqrt(torch.clamp(1.0 + R_c2[:, 2, 2] - R_c2[:, 0, 0] - R_c2[:, 1, 1], min=eps))
            safe_t2 = t2.where(t2 > eps, torch.full_like(t2, eps))
            quats_for_m_branch[cond2, 0] = (R_c2[:, 1, 0] - R_c2[:, 0, 1]) / (2.0 * safe_t2)
            quats_for_m_branch[cond2, 1] = (R_c2[:, 0, 2] + R_c2[:, 2, 0]) / (2.0 * safe_t2)
            quats_for_m_branch[cond2, 2] = (R_c2[:, 1, 2] + R_c2[:, 2, 1]) / (2.0 * safe_t2)
            quats_for_m_branch[cond2, 3] = 0.5 * t2
        
        # Place the results from this branch into the correct slice of quats_for_valid_Rs
        quats_for_valid_Rs[mask_trace_non_pos] = quats_for_m_branch

    # Normalize the computed quaternions (only those for valid R matrices)
    norms = quats_for_valid_Rs.norm(dim=-1, keepdim=True)
    
    # Check for NaNs before normalization (could come from R_flat_valid or calculations)
    # if torch.isnan(quats_for_valid_Rs).any():
    #     print("[NaN DETECTED] NaN found in quaternions before normalization.")

    quats_for_valid_Rs_normalized = quats_for_valid_Rs / (norms + eps) # Safe division

    # Check for NaNs after normalization (e.g. if norm was zero and eps was not enough, or input was already NaN)
    nan_mask_after_norm = torch.isnan(quats_for_valid_Rs_normalized)
    if nan_mask_after_norm.any():
        # print(f"[NaN DETECTED] {nan_mask_after_norm.sum().item()//4} quaternions contained NaN after normalization. Replacing with identity.")
        # Replace rows containing NaNs with identity quaternion
        rows_with_nans = nan_mask_after_norm.any(dim=-1)
        quats_for_valid_Rs_normalized[rows_with_nans, 0] = 1.0
        quats_for_valid_Rs_normalized[rows_with_nans, 1:] = 0.0
        # Re-normalize these identities (though already normalized) just to be safe if needed,
        # but simple assignment is usually fine for identity.

    # Place the processed valid quaternions into the full output tensor
    all_quats[valid_mask] = quats_for_valid_Rs_normalized
    
    return all_quats.reshape(*batch_shape, 4)


def extract_scales_and_rotations(gaussians):
    # Assume gaussians.covariances has shape [..., 3, 3]
    scales, rotation_matrices = torch.linalg.eigh(gaussians.covariances)
    quaternions = rotation_matrix_to_quaternion(rotation_matrices)
    return scales, quaternions





def local_render(
    intrinsics,
    extrinsics,
    near,
    far,
    image_shape,
    gaussian,
    pipe,
    bg_color: torch.Tensor,
    scaling_modifier=1.0,
    override_color=None,
):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """
    height, width = image_shape
    fov_x, fov_y = get_fov(intrinsics[None]).unbind(dim=-1)
    tan_fov_x = (0.5 * fov_x).tan()
    tan_fov_y = (0.5 * fov_y).tan()

    projection_matrix = get_projection_matrix(near[None], far[None], fov_x, fov_y)
    projection_matrix = rearrange(projection_matrix, "b i j -> b j i")
    view_matrix = rearrange(extrinsics[None].inverse(), "b i j -> b j i")
    full_projection = view_matrix @ projection_matrix

    active_sh_degree = 0
    max_sh_degree = 0

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = (
        torch.zeros_like(
            gaussian.means, dtype=gaussian.means.dtype, requires_grad=True, device="cuda"
        )
        + 0
    )
    try:
        screenspace_points.retain_grad()
    except:
        pass

    
    raster_settings = GaussianRasterizationSettings(
        image_height=int(height),
        image_width=int(width),
        tanfovx=tan_fov_x,
        tanfovy=tan_fov_y,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=view_matrix,
        projmatrix=full_projection,
        sh_degree=active_sh_degree,
        campos=extrinsics[:3, 3],
        prefiltered=False,
        debug=False,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = gaussian.means
    means2D = screenspace_points
    opacity = gaussian.opacities

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        print('not supported, there is the covariance matrix if we need it')
        cov3D_precomp = gaussian.get_covariance(scaling_modifier)
    else:
        scales = gaussian.scales
        rotations = gaussian.rotations

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = gaussian.features.transpose(1, 2).view(
                -1, 3, (max_sh_degree + 1) ** 2
            )
            camera_center = get_camera_center(extrinsics)
            dir_pp = gaussian.means - camera_center.repeat(
                gaussian.features.shape[0], 1
            )
            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = gaussian.get_features

        shs = shs.float()
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    # import pdb; pdb.set_trace()
    rendered_image, radii, depth = rasterizer(
        means3D=means3D.float(),
        means2D=means2D.float(),
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacity.float(),
        scales=scales.float(),
        rotations=rotations.float(),
        cov3D_precomp=cov3D_precomp,
    )

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter": radii > 0,
        "radii": radii,
        "depth_3dgs": depth,
    }

from dataclasses import dataclass

@dataclass
class PipelineConfig:
    convert_SHs_python: bool = False
    compute_cov3D_python: bool = False
    debug: bool = False

# @torch.no_grad()
def render(gaussian, batch, image_shape, key='context'):    
    """Render 2D feature map (D, H, W) based on 3D feature (N, D) and camera."""

    ###########
    import numpy as np


    batch_size, _ ,n_feat = gaussian.features.shape
    _, num_views, _, h, w = batch[key]["image"].shape
    # image_shape = (h, w)
    # image_shape = (h//14, w//14)


    # Then, instead of using a dictionary, create an instance:
    pipe = PipelineConfig()
    
    background_tensor = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device="cuda")    

    sem_list = []
    for b in range(batch_size):
        curr_sem_list = []
        scales, rotations = extract_scales_and_rotations(gaussian)
        curr_gaussian = Gaussians(
             gaussian.means[b],
             gaussian.covariances[b],
             scales[b],
             rotations[b],
             gaussian.harmonics[b],
             gaussian.opacities[b],             
        )
        curr_gaussian.features = gaussian.features[b]
        curr_gaussian.opacities = curr_gaussian.opacities[:,None]

        curr_features = curr_gaussian.features
        curr_intrinsics = batch[key]["intrinsics"][b]
        curr_extrinsics = batch[key]["extrinsics"][b]
        curr_near = batch[key]["near"][b]
        curr_far = batch[key]["far"][b]
        for v in range(num_views):
            counts = torch.zeros(n_feat, dtype=torch.float32, device="cuda")
            sem = None
            for j in np.arange(0, n_feat, 3):
                _j = min(j, n_feat - 3)
                _j = max(_j, 0)
                _semantic_map = local_render(
                    curr_intrinsics[v],
                    curr_extrinsics[v],
                    curr_near[v],
                    curr_far[v],
                    image_shape,
                    curr_gaussian,
                    pipe,
                    background_tensor,
                    override_color=curr_features[:, _j : _j + 3],
                )["render"]

                if sem is None:
                    sem = torch.zeros(
                        (n_feat, *_semantic_map.shape[1:]),
                        dtype=torch.float32,
                        device="cuda",
                    )
                dj = min(3, len(sem) - _j)
                sem[_j : _j + dj] += _semantic_map[:dj]
                counts[_j : _j + dj] += 1
            sem /= counts[:, None, None]
            curr_sem_list.append(sem)
        curr_sem_list = torch.stack(curr_sem_list, dim=0)
        sem_list.append(curr_sem_list)
    sem_list = torch.stack(sem_list, dim=0)


    return sem_list
