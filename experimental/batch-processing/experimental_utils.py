"""
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch.ao.quantization import HistogramObserver
from tqdm import tqdm

def linear_quantization(data, quantizations:int=6):
    """"""
    unique_quantization = np.round(data/255 * (quantizations-1)) * (255 / (quantizations-1))
    unique_quantization = unique_quantization.astype(np.uint8)
    return unique_quantization

def geometric_quantization(data, quantizations:int=8):
    """"""


def max_intensity_coordinates(data,axis:int=1):
    """"""
    depth,height,width = data.shape
    depth_indices,width_indices = np.indices((depth,width))
    height_indices = data.argmax(axis=axis)
    indices = (depth_indices,height_indices,width_indices)
    points = np.column_stack((depth_indices.ravel(),height_indices.ravel(),width_indices.ravel()))
    print(len(indices),len(indices[0]),points.shape)
    return indices,points

##########################################################################################

def perlin_3d_deterministic(shape, res, seed, device="cuda"):
    """
    3D Perlin noise with normalized coordinates. 
    Scaling is now frequency-based relative to the total volume size.
    """
    D, H, W = shape
    res_d, res_h, res_w = res
    generator = torch.Generator(device=device).manual_seed(seed)

    # 1. Normalize coordinates: base_freq=1.0 now covers the whole volume
    z = torch.linspace(0, res_d, D, device=device)
    y = torch.linspace(0, res_h, H, device=device)
    x = torch.linspace(0, res_w, W, device=device)
    grid = torch.stack(torch.meshgrid(z, y, x, indexing='ij'), dim=-1)

    p = grid.floor().long()
    f = grid - p
    fade_f = f * f * f * (f * (f * 6 - 15) + 10) # 5th degree smoothstep

    def get_grad(p_coords):
        # Generate deterministic 3D unit vectors
        phi = torch.rand(p_coords.shape[:-1], generator=generator, device=device) * 6.283185
        theta = torch.acos(2 * torch.rand(p_coords.shape[:-1], generator=generator, device=device) - 1)
        return torch.stack([theta.sin() * phi.cos(), theta.sin() * phi.sin(), theta.cos()], dim=-1)

    # 2. Vectorized 8-corner dot products
    dots = []
    for i in [0, 1]:
        for j in [0, 1]:
            for k in [0, 1]:
                offset = torch.tensor([i, j, k], device=device)
                grad = get_grad(p + offset)
                dots.append((grad * (f - offset)).sum(-1))

    # 3. Trilinear Interpolation via Lerp
    z00 = torch.lerp(dots[0], dots[4], fade_f[..., 0])
    z01 = torch.lerp(dots[1], dots[5], fade_f[..., 0])
    z10 = torch.lerp(dots[2], dots[6], fade_f[..., 0])
    z11 = torch.lerp(dots[3], dots[7], fade_f[..., 0])
    y0 = torch.lerp(z00, z10, fade_f[..., 1])
    y1 = torch.lerp(z01, z11, fade_f[..., 1])
    return torch.lerp(y0, y1, fade_f[..., 2])

# def get_3d_context_stats_via_indices(cpu_image, cpu_mask, margin_size=12, device="cuda"):
#     """
#     Extracts stats by calculating a bounding box around the mask hole 
#     and expanding it to find the surrounding context margin.
#     """
#     # 1. Find the spatial indices of the hole
#     # coords shape: [N, 3] -> (depth, height, width)
#     coords = torch.nonzero(cpu_mask)
    
#     if coords.shape[0] == 0:
#         return cpu_image.mean().to(device), cpu_image.std().to(device)

#     # 2. Calculate the bounding box of the hole
#     mins = coords.min(dim=0).values
#     maxs = coords.max(dim=0).values

#     # 3. Expand the boundaries by the margin size
#     # Clamp to ensure we stay within the volume dimensions
#     d_max, h_max, w_max = cpu_image.shape
    
#     z_start = max(0, mins[0] - margin_size)
#     z_end   = min(d_max, maxs[0] + margin_size + 1)
    
#     y_start = max(0, mins[1] - margin_size)
#     y_end   = min(h_max, maxs[1] + margin_size + 1)
    
#     x_start = max(0, mins[2] - margin_size)
#     x_end   = min(w_max, maxs[2] + margin_size + 1)

#     # 4. Extract the sub-volume (The 'Margin Box')
#     margin_patch = cpu_image[z_start:z_end, y_start:y_end, x_start:x_end]
#     mask_patch   = cpu_mask[z_start:z_end, y_start:y_end, x_start:x_end]

#     # 5. Context pixels are those in the patch NOT covered by the mask
#     context_pixels = margin_patch[mask_patch == 0]

#     if context_pixels.numel() == 0:
#         return cpu_image.mean().to(device), cpu_image.std().to(device)

#     # Upload only the scalars to GPU
#     return context_pixels.mean().to(device), context_pixels.std().to(device)

def generate_targeted_3d_fill_optimized(
    cpu_image, cpu_mask, seed=42, perlin_weight=0.7, 
    context_margin=24, base_freq=1.5, octaves=4, 
    persistence=0.5, lacunarity=2.0, device="cuda"
):
    # 1. Targeted Sub-Volume Logic
    coords = torch.nonzero(cpu_mask)
    if coords.shape[0] == 0: return cpu_image
    
    y_start = max(0, coords.min(dim=0).values[1] - context_margin)
    y_end = min(cpu_image.shape[1], coords.max(dim=0).values[1] + context_margin + 1)
    
    gpu_sub_vol = cpu_image[:, y_start:y_end, :].to(device)
    gpu_sub_mask = cpu_mask[:, y_start:y_end, :].to(device).float()

    # 2. GPU Contextual Stats
    valid_px = gpu_sub_vol[gpu_sub_mask < 0.5]
    mean, std = valid_px.mean(), valid_px.std()

    # 3. Procedural Generation
    noise = torch.zeros(gpu_sub_vol.shape, device=device)
    amp, freq = 1.0, base_freq
    for i in range(octaves):
        noise.add_(perlin_3d_deterministic(gpu_sub_vol.shape, (freq, freq, freq), seed + i, device), alpha=amp)
        amp *= persistence
        freq *= lacunarity
    
    # 4. Corrected High-Frequency Noise & Normalization
    gen = torch.Generator(device=device).manual_seed(seed)
    # Fix: randn() with shape + generator avoids the randn_like keyword error
    fine_noise = torch.randn(noise.shape, generator=gen, device=device)
    noise.mul_(perlin_weight).add_(fine_noise, alpha=1 - perlin_weight)
    noise.sub_(noise.mean()).div_(noise.std() + 1e-6).mul_(std).add_(mean)

    # 5. Volumetric Feathering & Re-graft
    soft_mask = F.avg_pool3d(gpu_sub_mask.unsqueeze(0).unsqueeze(0), 
                             kernel_size=context_margin*2+1, stride=1, padding=context_margin).squeeze()
    
    gpu_sub_vol.mul_(1 - soft_mask).add_(noise.mul_(soft_mask))
    cpu_image[:, y_start:y_end, :] = gpu_sub_vol.cpu()
    
    return cpu_image
    
#######

def iterative_3d_inpaint_with_grid(volume, mask, iterations=10, verbose:bool=False):
    orig_shape = volume.shape
    v = volume.float() if volume.ndim == 5 else volume.unsqueeze(0).unsqueeze(0).float()
    m = mask.bool() if mask.ndim == 5 else mask.unsqueeze(0).unsqueeze(0).bool()
    
    n, c, d, h, w = v.shape
    device = v.device

    if verbose:
        print(f'device: {device}')

    # 1. Identity Grid
    z = torch.linspace(-1, 1, d, device=device)
    y = torch.linspace(-1, 1, h, device=device)
    x = torch.linspace(-1, 1, w, device=device)
    mesh_z, mesh_y, mesh_x = torch.meshgrid(z, y, x, indexing='ij')
    identity_grid = torch.stack((mesh_x, mesh_y, mesh_z), dim=-1).unsqueeze(0).expand(n, -1, -1, -1, -1)

    curr_vol = v.clone()
    curr_mask = m.clone()

    for _ in tqdm(range(iterations),desc="Inpainting:"):
        if not curr_mask.any(): break

        # 2. Identify the boundary
        inverted_mask = 1.0 - curr_mask.float()
        dilated_valid = F.max_pool3d(inverted_mask, kernel_size=3, stride=1, padding=1)
        boundary = curr_mask & (dilated_valid > 0.5)

        if not boundary.any(): break

        # 3. Create a GRADIENT to find the "pull" direction
        # smooth_mask goes from 1 (valid) to 0 (hole)
        smooth_mask = F.avg_pool3d(inverted_mask, kernel_size=3, stride=1, padding=1)
        
        # Calculate gradients of the mask to see which way "valid" data is
        # dz, dy, dx
        grad_z = smooth_mask[:, :, 2:, 1:-1, 1:-1] - smooth_mask[:, :, :-2, 1:-1, 1:-1]
        grad_y = smooth_mask[:, :, 1:-1, 2:, 1:-1] - smooth_mask[:, :, 1:-1, :-2, 1:-1]
        grad_x = smooth_mask[:, :, 1:-1, 1:-1, 2:] - smooth_mask[:, :, 1:-1, 1:-1, :-2]
        
        # Pad gradients back to original size
        dz = F.pad(grad_z, (1, 1, 1, 1, 1, 1))
        dy = F.pad(grad_y, (1, 1, 1, 1, 1, 1))
        dx = F.pad(grad_x, (1, 1, 1, 1, 1, 1))

        # 4. Shift the grid ONLY at the boundary toward the valid data gradient
        # Scaling factor: push grid coordinates by ~1.5 voxels in the direction of data
        shift_strength = 1.5 * (2.0 / torch.tensor([w, h, d], device=device))
        
        pulling_grid = identity_grid.clone()
        pulling_grid[..., 0] += dx.squeeze(1) * shift_strength[0] # x
        pulling_grid[..., 1] += dy.squeeze(1) * shift_strength[1] # y
        pulling_grid[..., 2] += dz.squeeze(1) * shift_strength[2] # z

        # 5. Sample using the distorted grid
        # Now at the boundary, grid_sample looks at the neighboring valid pixels
        refined = F.grid_sample(curr_vol, pulling_grid, mode='bilinear', 
                               padding_mode='border', align_corners=True)

        # Update
        curr_vol[boundary.expand_as(curr_vol)] = refined[boundary.expand_as(refined)]
        curr_mask[boundary] = False
        
    return curr_vol.view(orig_shape)

def iterative_3d_inpaint_cropped_grid(volume, mask, iterations=10, verbose:bool = False):
    """
    Inpaints 3D holes by distorting the sampling grid toward valid data.
    Optimized with a bounding box crop based on the iteration count.
    """
    orig_shape = volume.shape
    v = volume.float() if volume.ndim == 5 else volume.unsqueeze(0).unsqueeze(0).float()
    m = mask.bool() if mask.ndim == 5 else mask.unsqueeze(0).unsqueeze(0).bool()
    
    # 1. Calculate Bounding Box
    coords = torch.nonzero(m)
    if coords.shape[0] == 0:
        return volume # Nothing to fill
    
    # Get bounds [N, C, D, H, W]
    z_min, z_max = coords[:, 2].min(), coords[:, 2].max()
    y_min, y_max = coords[:, 3].min(), coords[:, 3].max()
    x_min, x_max = coords[:, 4].min(), coords[:, 4].max()
    
    # Context margin based on iterations
    margin = iterations + 2
    z_s, z_e = max(0, z_min - margin), min(v.shape[2], z_max + margin + 1)
    y_s, y_e = max(0, y_min - margin), min(v.shape[3], y_max + margin + 1)
    x_s, x_e = max(0, x_min - margin), min(v.shape[4], x_max + margin + 1)
    
    # Crop
    curr_vol = v[:, :, z_s:z_e, y_s:y_e, x_s:x_e].clone()
    curr_mask = m[:, :, z_s:z_e, y_s:y_e, x_s:x_e].clone()
    
    n, c, d, h, w = curr_vol.shape
    device = curr_vol.device

    if verbose:
        print(f"device: {device}")

    # 2. Identity Grid for the crop
    z = torch.linspace(-1, 1, d, device=device)
    y = torch.linspace(-1, 1, h, device=device)
    x = torch.linspace(-1, 1, w, device=device)
    mesh_z, mesh_y, mesh_x = torch.meshgrid(z, y, x, indexing='ij')
    identity_grid = torch.stack((mesh_x, mesh_y, mesh_z), dim=-1).unsqueeze(0).expand(n, -1, -1, -1, -1)

    # 3. Iterative Refinement
    for _ in tqdm(range(iterations),desc="Inpainting"):
        if not curr_mask.any(): break

        # Find boundary
        inverted_mask = 1.0 - curr_mask.float()
        dilated_valid = F.max_pool3d(inverted_mask, kernel_size=3, stride=1, padding=1)
        boundary = curr_mask & (dilated_valid > 0.5)

        if not boundary.any(): break

        # Calculate gradient toward valid data
        smooth_mask = F.avg_pool3d(inverted_mask, kernel_size=3, stride=1, padding=1)
        
        # Finite differences for gradients (direction of 'validity')
        dz = F.pad(smooth_mask[:, :, 2:, 1:-1, 1:-1] - smooth_mask[:, :, :-2, 1:-1, 1:-1], (1,1,1,1,1,1))
        dy = F.pad(smooth_mask[:, :, 1:-1, 2:, 1:-1] - smooth_mask[:, :, 1:-1, :-2, 1:-1], (1,1,1,1,1,1))
        dx = F.pad(smooth_mask[:, :, 1:-1, 1:-1, 2:] - smooth_mask[:, :, 1:-1, 1:-1, :-2], (1,1,1,1,1,1))

        # Distort grid: Shift coordinates by ~1.5 voxels toward valid area
        # Grid scale is [-1, 1], so 1 voxel = 2.0 / size
        shift_vec = torch.tensor([2.0/w, 2.0/h, 2.0/d], device=device) * 1.5
        
        pulling_grid = identity_grid.clone()
        pulling_grid[..., 0] += dx.squeeze(1) * shift_vec[0]
        pulling_grid[..., 1] += dy.squeeze(1) * shift_vec[1]
        pulling_grid[..., 2] += dz.squeeze(1) * shift_vec[2]

        # Sample from shifted coordinates
        refined = F.grid_sample(curr_vol, pulling_grid, mode='bilinear', 
                               padding_mode='border', align_corners=True)

        # Update boundary voxels
        curr_vol[boundary.expand_as(curr_vol)] = refined[boundary.expand_as(refined)]
        curr_mask[boundary] = False

    # 4. Paste results back to original volume
    out_vol = v.clone()
    out_vol[:, :, z_s:z_e, y_s:y_e, x_s:x_e] = curr_vol
    return out_vol.view(orig_shape)

def iterative_3d_inpaint_hybrid(volume, mask, iterations=10, device='cuda', verbose:bool = False):
    """
    Hybrid CPU/GPU Inpainting:
    - Bounding box and full volume stay on CPU.
    - Active hole area (sub-volume) is processed on GPU.
    """
    orig_shape = volume.shape
    # Ensure inputs are at least 5D for standardization, keep on CPU initially
    v_cpu = volume.float() if volume.ndim == 5 else volume.unsqueeze(0).unsqueeze(0).float()
    m_cpu = mask.bool() if mask.ndim == 5 else mask.unsqueeze(0).unsqueeze(0).bool()
    
    # 1. Bounding Box Calculation (CPU)
    coords = torch.nonzero(m_cpu)
    if coords.shape[0] == 0:
        return volume
    
    z_min, z_max = coords[:, 2].min().item(), coords[:, 2].max().item()
    y_min, y_max = coords[:, 3].min().item(), coords[:, 3].max().item()
    x_min, x_max = coords[:, 4].min().item(), coords[:, 4].max().item()
    
    margin = iterations + 2
    z_s, z_e = max(0, z_min - margin), min(v_cpu.shape[2], z_max + margin + 1)
    y_s, y_e = max(0, y_min - margin), min(v_cpu.shape[3], y_max + margin + 1)
    x_s, x_e = max(0, x_min - margin), min(v_cpu.shape[4], x_max + margin + 1)
    
    # 2. Transfer only the CROP to GPU
    curr_vol = v_cpu[:, :, z_s:z_e, y_s:y_e, x_s:x_e].to(device).clone()
    curr_mask = m_cpu[:, :, z_s:z_e, y_s:y_e, x_s:x_e].to(device).clone()
    
    n, c, d, h, w = curr_vol.shape

    # 3. GPU Grid Setup
    z = torch.linspace(-1, 1, d, device=device)
    y = torch.linspace(-1, 1, h, device=device)
    x = torch.linspace(-1, 1, w, device=device)
    mesh_z, mesh_y, mesh_x = torch.meshgrid(z, y, x, indexing='ij')
    identity_grid = torch.stack((mesh_x, mesh_y, mesh_z), dim=-1).unsqueeze(0).expand(n, -1, -1, -1, -1)

    # 4. Iterative Loop (Fully on GPU)
    shift_scale = torch.tensor([2.0/w, 2.0/h, 2.0/d], device=device) * 1.5

    for _ in tqdm(range(iterations),desc="Inpainting:"):
        if not curr_mask.any(): break

        inv_mask = 1.0 - curr_mask.float()
        dilated_valid = F.max_pool3d(inv_mask, kernel_size=3, stride=1, padding=1)
        boundary = curr_mask & (dilated_valid > 0.5)

        if not boundary.any(): break

        # Gradient pull logic
        smooth = F.avg_pool3d(inv_mask, kernel_size=3, stride=1, padding=1)
        dz = F.pad(smooth[:, :, 2:, 1:-1, 1:-1] - smooth[:, :, :-2, 1:-1, 1:-1], (1,1,1,1,1,1))
        dy = F.pad(smooth[:, :, 1:-1, 2:, 1:-1] - smooth[:, :, 1:-1, :-2, 1:-1], (1,1,1,1,1,1))
        dx = F.pad(smooth[:, :, 1:-1, 1:-1, 2:] - smooth[:, :, 1:-1, 1:-1, :-2], (1,1,1,1,1,1))

        pulling_grid = identity_grid.clone()
        pulling_grid[..., 0] += dx.squeeze(1) * shift_scale[0]
        pulling_grid[..., 1] += dy.squeeze(1) * shift_scale[1]
        pulling_grid[..., 2] += dz.squeeze(1) * shift_scale[2]

        refined = F.grid_sample(curr_vol, pulling_grid, mode='bilinear', 
                               padding_mode='border', align_corners=True)

        curr_vol[boundary.expand_as(curr_vol)] = refined[boundary.expand_as(refined)]
        curr_mask[boundary] = False

    # 5. Transfer result back to CPU and assign to original
    # We clone the original volume to avoid modifying the input tensor in-place
    output_volume = volume.clone().float()
    # Handle the fact that volume might have been 3D, 4D or 5D
    if output_volume.ndim == 3:
        output_volume[z_s:z_e, y_s:y_e, x_s:x_e] = curr_vol.squeeze().cpu()
    elif output_volume.ndim == 4:
        output_volume[:, z_s:z_e, y_s:y_e, x_s:x_e] = curr_vol.squeeze(0).cpu()
    else:
        output_volume[:, :, z_s:z_e, y_s:y_e, x_s:x_e] = curr_vol.cpu()
        
    return output_volume.view(orig_shape)

# OG works well
# def iterative_3d_inpaint_with_noise(
#     volume: torch.Tensor, 
#     mask: torch.Tensor, 
#     iterations: int = 10, 
#     device: str | torch.device = 'cuda', 
#     noise_std: float = 0.01, 
#     clump_ratio: float = 0.7, 
#     clump_size: int | list[int] = 5,
#     verbose: bool = False,
# ) -> torch.Tensor:
#     """
#     Final Refinement:
#     - Anisotropic noise scaling for height-axis variation.
#     - Iterative Gaussian smoothing on the 'fill-front' to eliminate meeting seams.
#     """
#     orig_shape = volume.shape
#     v_cpu = volume.float() if volume.ndim == 5 else volume.unsqueeze(0).unsqueeze(0).float()
#     m_cpu = mask.bool() if mask.ndim == 5 else mask.unsqueeze(0).unsqueeze(0).bool()
    
#     # 1. Bounding Box Optimization (CPU)
#     coords = torch.nonzero(m_cpu)
#     if coords.shape[0] == 0: return volume
    
#     margin = iterations + 5
#     z_s, z_e = max(0, coords[:, 2].min().item() - margin), min(v_cpu.shape[2], coords[:, 2].max().item() + margin + 1)
#     y_s, y_e = max(0, coords[:, 3].min().item() - margin), min(v_cpu.shape[3], coords[:, 3].max().item() + margin + 1)
#     x_s, x_e = max(0, coords[:, 4].min().item() - margin), min(v_cpu.shape[4], coords[:, 4].max().item() + iterations + 5)
    
#     # 2. Transfer to GPU
#     curr_vol = v_cpu[:, :, z_s:z_e, y_s:y_e, x_s:x_e].to(device).clone()
#     curr_mask = m_cpu[:, :, z_s:z_e, y_s:y_e, x_s:x_e].to(device).clone()
    
#     valid_data = curr_vol[~curr_mask.expand_as(curr_vol)]
#     l_min, l_max = (valid_data.min(), valid_data.max()) if valid_data.numel() > 0 else (0.0, 1.0)
#     n, c, d, h, w = curr_vol.shape
    
#     # 3. Anisotropic Noise Setup
#     # Force higher resolution along height (h) to increase vertical variation
#     if isinstance(clump_size, int):
#         # We divide H by a smaller number (e.g., 2) to make the noise seed denser vertically
#         noise_res = (max(1, d // clump_size), max(1, h // 2), max(1, w // clump_size))
#     else:
#         noise_res = (max(1, d // clump_size[0]), max(1, h // clump_size[1]), max(1, w // clump_size[2]))

#     clump_seed = torch.randn((n, c, *noise_res), device=device)
#     clump_base = F.interpolate(clump_seed, size=(d, h, w), mode='trilinear', align_corners=True)
#     clump_base = (clump_base - clump_base.mean()) / (clump_base.std() + 1e-6)
#     static_noise = ((clump_base * clump_ratio) + (torch.randn_like(curr_vol) * (1.0 - clump_ratio))) * noise_std

#     # Identity Grid
#     grid_z = torch.linspace(-1, 1, d, device=device)
#     grid_y = torch.linspace(-1, 1, h, device=device)
#     grid_x = torch.linspace(-1, 1, w, device=device)
#     mesh_z, mesh_y, mesh_x = torch.meshgrid(grid_z, grid_y, grid_x, indexing='ij')
#     id_grid = torch.stack((mesh_x, mesh_y, mesh_z), dim=-1).unsqueeze(0).expand(n, -1, -1, -1, -1)

#     # 4. Iterative Loop
#     sx, sy, sz = 2.0/w, 2.0/h, 2.0/d

#     for _ in range(iterations):
#         if not curr_mask.any(): break
        
#         inv_mask = 1.0 - curr_mask.float()
#         dilated_v = F.max_pool3d(inv_mask, kernel_size=3, stride=1, padding=1)
#         boundary = curr_mask & (dilated_v > 0.5)
#         if not boundary.any(): break

#         # Gradient Pull
#         smooth_m = F.avg_pool3d(inv_mask, kernel_size=3, stride=1, padding=1)
#         dz = F.pad(smooth_m[:, :, 2:, 1:-1, 1:-1] - smooth_m[:, :, :-2, 1:-1, 1:-1], (1,1,1,1,1,1))
#         dy = F.pad(smooth_m[:, :, 1:-1, 2:, 1:-1] - smooth_m[:, :, 1:-1, :-2, 1:-1], (1,1,1,1,1,1))
#         dx = F.pad(smooth_m[:, :, 1:-1, 1:-1, 2:] - smooth_m[:, :, 1:-1, 1:-1, :-2], (1,1,1,1,1,1))
#         pull = F.avg_pool3d(torch.cat([dx, dy, dz], dim=1), kernel_size=3, stride=1, padding=1)
        
#         p_grid = id_grid.clone()
#         p_grid[..., 0] += pull[:, 0] * (sx * 2.8) 
#         p_grid[..., 1] += pull[:, 1] * (sy * 2.8)
#         p_grid[..., 2] += pull[:, 2] * (sz * 2.8)
        
#         # Stochastic Jitter
#         jitter = (torch.rand_like(p_grid) * 2 - 1)
#         jitter[..., 0] *= (sx * 0.9); jitter[..., 1] *= (sy * 0.9); jitter[..., 2] *= (sz * 0.9)
#         p_grid += jitter

#         # Sample Image & Noise
#         refined = F.grid_sample(curr_vol, p_grid, mode='bilinear', padding_mode='border', align_corners=True)
#         weights = F.grid_sample(inv_mask, p_grid, mode='bilinear', padding_mode='border', align_corners=True)
#         refined = refined / (weights + 1e-5)
        
#         w_noise = F.grid_sample(static_noise, p_grid, mode='bilinear', padding_mode='border', align_corners=True)
#         res = torch.clamp(refined + w_noise, l_min, l_max)

#         # --- FIX: LAPLACIAN SMOOTHING AT SEAMS ---
#         # Before updating, we blur the 'refined' area to help it blend with neighbors
#         blurred_res = F.avg_pool3d(res, kernel_size=3, stride=1, padding=1, count_include_pad=False)
#         # Mix 50% blurred and 50% sharp to maintain texture while hiding seams
#         res = 0.5 * res + 0.5 * blurred_res

#         curr_vol[boundary.expand_as(curr_vol)] = res[boundary.expand_as(res)]
#         curr_mask[boundary] = False

#     # 5. Reconstruction
#     out_vol = volume.clone().float()
#     res_cpu = curr_vol.cpu()
#     slc = (slice(None), slice(None), slice(z_s, z_e), slice(y_s, y_e), slice(x_s, x_e))
#     if out_vol.ndim == 3: out_vol[slc[2:]] = res_cpu.squeeze()
#     elif out_vol.ndim == 4: out_vol[slc[1:]] = res_cpu.squeeze(0)
#     else: out_vol[slc] = res_cpu
        
#     return out_vol.view(orig_shape).to(volume.device)

def iterative_3d_inpaint_with_noise(
    volume: torch.Tensor, 
    mask: torch.Tensor, 
    iterations: int = 10, 
    device: str | torch.device = 'cuda', 
    noise_std: float = 0.01, 
    clump_ratio: float = 0.7, 
    clump_size: int = 5,
    intensity_offset: float = 0.1,
    sharpness: float = 0.1,
    verbose: bool = False,
) -> torch.Tensor:
    """
    Args:
        intensity_offset: Constant added to infilled voxels (e.g. 0.05 to brighten).
        sharpness: 0.0 to 1.0. Boosts high-frequency edges within the fill.
    """
    orig_shape = volume.shape
    v_cpu = volume.float() if volume.ndim == 5 else volume.unsqueeze(0).unsqueeze(0).float()
    m_cpu = mask.bool() if mask.ndim == 5 else mask.unsqueeze(0).unsqueeze(0).bool()
    
    # 1. Bounding Box Optimization
    coords = torch.nonzero(m_cpu)
    if coords.shape == 0: return volume
    
    margin = iterations + 5
    z_s, z_e = max(0, coords[:, 2].min().item() - margin), min(v_cpu.shape[2], coords[:, 2].max().item() + margin + 1)
    y_s, y_e = max(0, coords[:, 3].min().item() - margin), min(v_cpu.shape[3], coords[:, 3].max().item() + margin + 1)
    x_s, x_e = max(0, coords[:, 4].min().item() - margin), min(v_cpu.shape[4], coords[:, 4].max().item() + margin + 1)
    
    curr_vol = v_cpu[:, :, z_s:z_e, y_s:y_e, x_s:x_e].to(device).clone()
    curr_mask = m_cpu[:, :, z_s:z_e, y_s:y_e, x_s:x_e].to(device).clone()
    
    # Identify local range
    valid_data = curr_vol[~curr_mask.expand_as(curr_vol)]
    l_min, l_max = (valid_data.min(), valid_data.max()) if valid_data.numel() > 0 else (0.0, 1.0)
    n, c, d, h, w = curr_vol.shape
    
    # 2. Anisotropic Noise and Grid
    noise_res = (max(1, d // clump_size), max(1, h // 2), max(1, w // clump_size))
    clump_seed = torch.randn((n, c, *noise_res), device=device)
    clump_base = F.interpolate(clump_seed, size=(d, h, w), mode='trilinear', align_corners=True)
    clump_base = (clump_base - clump_base.mean()) / (clump_base.std() + 1e-6)
    static_noise = ((clump_base * clump_ratio) + (torch.randn_like(curr_vol) * (1.0 - clump_ratio))) * noise_std

    grid_coords = [torch.linspace(-1, 1, s, device=device) for s in [d, h, w]]
    mesh_z, mesh_y, mesh_x = torch.meshgrid(grid_coords[0], grid_coords[1], grid_coords[2], indexing='ij')
    id_grid = torch.stack((mesh_x, mesh_y, mesh_z), dim=-1).unsqueeze(0).expand(n, -1, -1, -1, -1)

    # 3. Iterative Loop
    sx, sy, sz = 2.0/w, 2.0/h, 2.0/d

    for _ in range(iterations):
        if not curr_mask.any(): break
        inv_mask = 1.0 - curr_mask.float()
        dilated_v = F.max_pool3d(inv_mask, kernel_size=3, stride=1, padding=1)
        boundary = curr_mask & (dilated_v > 0.5)
        if not boundary.any(): break

        # Gradient Pull Logic
        smooth_m = F.avg_pool3d(inv_mask, kernel_size=3, stride=1, padding=1)
        dz = F.pad(smooth_m[:, :, 2:, 1:-1, 1:-1] - smooth_m[:, :, :-2, 1:-1, 1:-1], (1,1,1,1,1,1))
        dy = F.pad(smooth_m[:, :, 1:-1, 2:, 1:-1] - smooth_m[:, :, 1:-1, :-2, 1:-1], (1,1,1,1,1,1))
        dx = F.pad(smooth_m[:, :, 1:-1, 1:-1, 2:] - smooth_m[:, :, 1:-1, 1:-1, :-2], (1,1,1,1,1,1))
        pull = F.avg_pool3d(torch.cat([dx, dy, dz], dim=1), kernel_size=3, stride=1, padding=1)
        
        p_grid = id_grid.clone()
        p_grid[..., 0] += pull[:, 0] * (sx * 2.8) 
        p_grid[..., 1] += pull[:, 1] * (sy * 2.8)
        p_grid[..., 2] += pull[:, 2] * (sz * 2.8)
        p_grid += (torch.rand_like(p_grid) * 2 - 1) * torch.tensor([sx, sy, sz], device=device) * 0.9

        # Sampling and Normalization
        refined = F.grid_sample(curr_vol, p_grid, mode='bilinear', padding_mode='border', align_corners=True)
        weights = F.grid_sample(inv_mask, p_grid, mode='bilinear', padding_mode='border', align_corners=True)
        refined = refined / (weights + 1e-5)
        
        w_noise = F.grid_sample(static_noise, p_grid, mode='bilinear', padding_mode='border', align_corners=True)
        
        # Apply Infill + Offset
        res = refined + w_noise + intensity_offset

        # --- NEW: SHARPNESS & SEAM BLENDING ---
        # 1. Local average for both blending and unsharp masking
        local_avg = F.avg_pool3d(res, kernel_size=3, stride=1, padding=1, count_include_pad=False)
        
        # 2. Unsharp Masking: (Original + (Original - Blurred) * Strength)
        if sharpness > 0:
            res = res + (res - local_avg) * (sharpness * 2.0)
            
        # 3. Soften meeting seams (Blend result with its own average)
        res = 0.5 * res + 0.5 * local_avg

        # Update and Clamp
        res = torch.clamp(res, l_min, l_max)
        curr_vol[boundary.expand_as(curr_vol)] = res[boundary.expand_as(res)]
        curr_mask[boundary] = False

    # 4. Reconstruction
    out_vol = volume.clone().float()
    res_cpu = curr_vol.cpu()
    slc = (slice(None), slice(None), slice(z_s, z_e), slice(y_s, y_e), slice(x_s, x_e))
    if out_vol.ndim == 3: out_vol[slc[2:]] = res_cpu.squeeze()
    elif out_vol.ndim == 4: out_vol[slc[1:]] = res_cpu.squeeze(0)
    else: out_vol[slc] = res_cpu
        
    return out_vol.view(orig_shape).to(volume.device)

##########################################################################################

def quantize_intensity_tensor_to_8bit(input_intensity_tensor, use_symmetric_range=False, convert_to_numpy=False):
    """
    Quantizes n-dimensional intensity data (e.g., MRI/CT scans) using 
    entropy-based histogram weights to minimize information loss.
    """
    # 1. Define constants for 8-bit unsigned storage
    TARGET_QUANTIZED_DTYPE = torch.quint8
    
    # 2. Determine the Quantization Scheme (qscheme)
    # Affine: Allows a zero-point shift to maximize precision for positive scans.
    # Symmetric: Centers the range at zero (preferred for gradient/difference data).
    if use_symmetric_range:
        active_quantization_scheme = torch.per_tensor_symmetric
    else:
        active_quantization_scheme = torch.per_tensor_affine

    # 3. Initialize the Entropy-Based Histogram Observer
    # The observer analyzes the distribution to find an optimal clipping range
    # that minimizes Kullback-Leibler (KL) Divergence.
    entropy_weight_observer = HistogramObserver(
        dtype=TARGET_QUANTIZED_DTYPE, 
        qscheme=active_quantization_scheme
    )

    # 4. Calibration: Calculate optimal Scale and Zero-Point
    # We detach the tensor to ensure no gradient tracking occurs during calibration.
    entropy_weight_observer(input_intensity_tensor.detach())
    optimal_scale, optimal_zero_point = entropy_weight_observer.calculate_qparams()

    # 5. Apply Quantization
    # This transforms the N-dimensional float tensor into an 8-bit Quantized Tensor.
    quantized_intensity_output = torch.quantize_per_tensor(
        input_intensity_tensor, 
        scale=optimal_scale.item(), 
        zero_point=optimal_zero_point.item(), 
        dtype=TARGET_QUANTIZED_DTYPE
    )

    # 6. Final Data Formatting
    if convert_to_numpy:
        # We extract the raw integer representation (0-255) for NumPy compatibility.
        # Direct .numpy() calls on quantized tensors are unsupported in 2026.
        return quantized_intensity_output.int_repr().cpu().numpy()
    
    return quantized_intensity_output

##########################################################################################

def grab_center_window(
    input_data: np.ndarray | torch.Tensor, 
    window_shape: int | tuple[int, ...], 
    return_numpy: bool = True
) -> np.ndarray | torch.Tensor:
    """Slices an N-dimensional center window from a NumPy array or Torch tensor.

    Standardizes input using torch.as_tensor to leverage zero-copy memory 
    sharing and provides a unified interface for N-dimensional indexing.

    Args:
        input_data: The input multi-dimensional data source.
        window_shape: The target size of the output window. If an integer is 
            provided, it is applied uniformly to all dimensions.
        return_numpy: If True, returns a numpy.ndarray. If False, returns 
             a torch.Tensor.

    Returns:
        A view (or copy if return_numpy=True) of the center sub-window.
    """
    # Standardize input to a tensor without unnecessary copying
    # [torch.as_tensor](https://pytorch.org)
    target_tensor = torch.as_tensor(input_data)
    number_of_dimensions = target_tensor.ndim
    
    # Normalize window_shape to a tuple matching the input dimensionality
    if isinstance(window_shape, int):
        window_shape = (window_shape,) * number_of_dimensions
        
    # Calculate center coordinates and half-widths for all dimensions
    # Using floor division to determine the anchor point
    center_coordinates = np.array(target_tensor.shape) // 2
    half_widths = np.array(window_shape) // 2
    
    # Construct a list of slice objects for dynamic N-dimensional indexing
    dimension_slices = []
    for dimension_index in range(number_of_dimensions):
        start_index = max(0, center_coordinates[dimension_index] - half_widths[dimension_index])
        stop_index = min(
            target_tensor.shape[dimension_index], 
            start_index + window_shape[dimension_index]
        )
        dimension_slices.append(slice(start_index, stop_index))
        
    # Perform the extraction using a tuple of slice objects
    window_result = target_tensor[tuple(dimension_slices)]
    
    if return_numpy:
        # Move tensor to CPU before [numpy conversion](https://pytorch.org)
        return window_result.detach().cpu().numpy()
    
    return window_result

def grab_center_ellipsoid_optimized(
    input_data: np.ndarray | torch.Tensor, 
    neighbor_count: int | tuple[int, ...] = 1,
    return_numpy: bool = True
) -> np.ndarray | torch.Tensor:
    """Optimized N-D ellipsoid extraction with dynamic broadcast shape handling."""
    target_tensor = torch.as_tensor(input_data)
    input_shape = target_tensor.shape
    number_of_dimensions = target_tensor.ndim
    
    if isinstance(neighbor_count, int):
        neighbor_count = (neighbor_count,) * number_of_dimensions

    dimension_slices = []
    dimension_radii = []

    for dimension_index, dimension_size in enumerate(input_shape):
        neighbors = neighbor_count[dimension_index]
        if dimension_size % 2 != 0:
            center = dimension_size // 2
            start, stop, radius = center - neighbors, center + neighbors + 1, neighbors + 0.5
        else:
            center_low = (dimension_size // 2) - 1
            start, stop, radius = center_low - neighbors, center_low + neighbors + 2, neighbors + 1.0
            
        dimension_slices.append(slice(max(0, int(start)), min(dimension_size, int(stop))))
        dimension_radii.append(float(radius))

    # Efficient memory view
    sub_window = target_tensor[tuple(dimension_slices)]
    
    # Initialize the sum
    squared_distance_sum = torch.tensor(0.0, device=target_tensor.device, dtype=torch.float32)

    for dimension_index, dimension_size in enumerate(sub_window.shape):
        coordinates = torch.arange(dimension_size, device=target_tensor.device, dtype=torch.float32)
        center = (dimension_size - 1) / 2.0
        
        # Build broadcastable view shape: (1, 1, ..., dim_size, ..., 1)
        view_shape = [1] * number_of_dimensions
        view_shape[dimension_index] = dimension_size
        
        # Normalized squared distance for this axis
        component = ((coordinates.view(view_shape) - center) ** 2) / (dimension_radii[dimension_index] ** 2)
        
        # Use out-of-place addition to allow the broadcast shape to expand
        # [Broadcasting Semantics](https://pytorch.org)
        squared_distance_sum = squared_distance_sum + component

    # Create binary mask and apply to sub-window
    ellipsoid_mask = squared_distance_sum <= 1.0
    result = sub_window * ellipsoid_mask

    if return_numpy:
        return result.detach().cpu().numpy()
        
    return result

##########################################################################################

##########################################################################################
from kan import KAN
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from scipy import stats

def normalize_to_range(
    data: np.ndarray | torch.Tensor, 
    stats: dict[str, float] = None, 
    target_range: tuple[float, float] = (1.0, 2.0)
) -> tuple[np.ndarray | torch.Tensor, dict[str, float]]:
    """
    Normalizes data to a custom range to avoid numerical singularities in models.
    
    This is specifically useful for KANs to prevent 'UnboundLocalError: coef' 
    when using functions like log or 1/x, as shifting to [1.0, 2.0] ensures 
    the domain remains positive and away from zero.

    Args:
        data: The input data (e.g., binned 0-90 data). 
              Accepts both NumPy ndarrays and PyTorch Tensors.
        stats: Dictionary with 'min' and 'max' values. If None, calculates 
               them from the input.
        target_range: The desired output domain. Default is (1.0, 2.0).

    Returns:
        tuple containing:
            - The transformed data in the same type as input.
            - A dictionary containing 'min' and 'max' used for transformation.
    """
    a, b = target_range
    
    # 1. Capture or retrieve domain statistics
    if stats is None:
        d_min = data.min()
        d_max = data.max()
    else:
        d_min = stats['min']
        d_max = stats['max']
    
    # 2. Perform the Min-Max Shift
    # 1e-9 added to denominator to prevent ZeroDivision on identical inputs
    scale_factor = (b - a) / (d_max - d_min + 1e-9)
    transformed = a + (data - d_min) * scale_factor
    
    return transformed, {'min': d_min, 'max': d_max}

def denormalize_from_range(
    norm_data: np.ndarray | torch.Tensor, 
    stats: dict[str, float], 
    target_range: tuple[float, float] = (1.0, 2.0)
) -> np.ndarray | torch.Tensor:
    """
    Maps normalized predictions back to the original physical scale.

    Args:
        norm_data: The normalized output from the model.
        stats: The statistics dictionary returned by normalize_to_range.
        target_range: The range used during normalization. Default (1.0, 2.0).

    Returns:
        The data mapped back to the original scale.
    """
    a, b = target_range
    d_min, d_max = stats['min'], stats['max']
    
    # Inverse of the transformation formula to recover original units
    original = ((norm_data - a) * (d_max - d_min) / (b - a)) + d_min
    
    return original

def equidistant_mask_by_degree(
    shape:tuple, center:tuple|np.ndarray|None=None, scan_angle: float = 100, scan_angle_pad: int = 2
):
    """"""
    sampled_masks = []
    slow_axis, fast_axis = shape[:2]
    slow_coords, fast_coords = np.ogrid[0:slow_axis, 0:fast_axis]
    # get centroid
    # center_slow,center_fast = ndimage.center_of_mass(mask)
    if center is None:
        center_slow, center_fast = slow_axis // 2, fast_axis // 2
        distance_from_center = 0.0
    else:
        center_slow, center_fast = center
        distance_from_center = np.linalg.norm(np.asarray((center))-np.asarray((slow_axis//2,fast_axis//2)))

    # shift coordinates for the center
    slow_coords = slow_coords - center_slow
    fast_coords = fast_coords - center_fast
        
    # TODO add in support for rotations
    # ellipsoid adjustment parameters
    major_axis = slow_axis
    minor_axis = fast_axis
    # calculate distances
    # radii_map = np.sqrt(slow_coords**2 + fast_coords**2)
    elliptical_distance_squared = (slow_coords / major_axis) ** 2 + (
        fast_coords / minor_axis
    ) ** 2
    radii_map = np.sqrt(elliptical_distance_squared) * major_axis
    # define boundaries
    max_distance = np.max(radii_map)
    # num_rings = int(scan_angle/2.0)
    rings_per_pixel = scan_angle / slow_axis

    shift_padding = round(distance_from_center*rings_per_pixel)

    num_rings = round(rings_per_pixel * max_distance) - 2
    zone_boundaries = np.linspace(0, max_distance, num_rings + 1)

    sampled_masks = []
    for i in range((scan_angle // 2 + scan_angle_pad + shift_padding)):
        inner_radius = zone_boundaries[i]
        outer_radius = zone_boundaries[i + 1]
        # create mask for actively selected pixels
        ring_mask = (radii_map >= inner_radius) & (radii_map < outer_radius)
        sampled_masks.append(ring_mask)

    return sampled_masks

def aggregate_per_mask(
        data:np.ndarray, multi_mask:np.ndarray|list
):
    """"""
    sampled_pixels = []
    for mask in multi_mask:
        pixels_in_ring = data[mask]
        if pixels_in_ring.sum() > 0:
            nonzero_mask = pixels_in_ring > 0
            # sampled_pixels[f"Ring_{i+1}"] = pixels_in_ring[nonzero_mask].mean()
            sampled_pixels.append(pixels_in_ring[nonzero_mask].mean())
        else:
            # sampled_pixels.append(0.0)
            pass
    
    return sampled_pixels

def equidistant_pixel_error_by_degree(
    error_map: np.ndarray, scan_angle: float = 100, scan_angle_pad: int = 2
):
    """ """

    # replace nan values with zero
    data = np.nan_to_num(error_map)
    # get mask of nonzero values
    mask = data.astype(bool)
    # create coordinate grid
    slow_axis, fast_axis = data.shape[:2]
    slow_coords, fast_coords = np.ogrid[0:slow_axis, 0:fast_axis]
    # get centroid
    # center_slow,center_fast = ndimage.center_of_mass(mask)
    center_slow, center_fast = slow_axis // 2, fast_axis // 2
    # shift coordinates for the center
    slow_coords = slow_coords - center_slow
    fast_coords = fast_coords - center_fast
    # TODO add in support for rotations
    # ellipsoid adjustment parameters
    major_axis = slow_axis
    minor_axis = fast_axis
    # calculate distances
    # radii_map = np.sqrt(slow_coords**2 + fast_coords**2)
    elliptical_distance_squared = (slow_coords / major_axis) ** 2 + (
        fast_coords / minor_axis
    ) ** 2
    radii_map = np.sqrt(elliptical_distance_squared) * major_axis
    # define boundaries
    max_distance = np.max(radii_map)
    # num_rings = int(scan_angle/2.0)
    rings_per_pixel = scan_angle / slow_axis
    num_rings = round(rings_per_pixel * max_distance) - 2
    zone_boundaries = np.linspace(0, max_distance, num_rings + 1)
    # zone_boundaries = np.linspace(0,max_distance, num_rings+1)

    # initial_ring_mask = radii_map < zone_boundaries[0]
    # iniitial_pixels_in_ring = data[initial_ring_mask]

    sampled_pixels = []
    sampled_masks = []
    # sampled_pixels = {"Ring_0": data[int(center_slow),int(center_fast)]}
    # sampled_pixels = [data[int(center_slow),int(center_fast)]]
    # sampled_pixels = [iniitial_pixels_in_ring.mean()]

    # for i in range(num_rings):
    for i in range((scan_angle // 2 + scan_angle_pad)):
        inner_radius = zone_boundaries[i]
        outer_radius = zone_boundaries[i + 1]
        # create mask for actively selected pixels
        ring_mask = (radii_map >= inner_radius) & (radii_map < outer_radius)
        sampled_masks.append(ring_mask)
        # extract pixels
        pixels_in_ring = data[ring_mask]
        if pixels_in_ring.sum() > 0:
            nonzero_mask = pixels_in_ring > 0
            # sampled_pixels[f"Ring_{i+1}"] = pixels_in_ring[nonzero_mask].mean()
            sampled_pixels.append(pixels_in_ring[nonzero_mask].mean())
        else:
            # sampled_pixels.append(0.0)
            pass

    return sampled_pixels, sampled_masks

def linear_regress_and_plot(
            data: tuple[np.ndarray, np.ndarray],
            labels: tuple[str, str],
            sem_data: None | np.ndarray = None,
            title: str = "Generic Linear Regression Plot",
        ):
            """"""
            x, y = data
            x = x.ravel()
            y = y.ravel()
            slope, intercept, r_value, p_value, std_error = stats.linregress(x=x, y=y)
            print(f"p-value: {p_value}, std error: {std_error}")

            plt.figure(figsize=(10,6))

            if sem_data is not None:
                plt.errorbar(
                    x,
                    y,
                    sem_data,
                    fmt="o",
                    capsize=4,
                    label="Quantile Mean ± SEM",
                    # color="cyan",
                    color="black",
                    marker='s',
                    markersize=6,
                    zorder=1,
                )
            else:
                plt.scatter(x, y, label="Data",color="black",marker='s')
                # plt.scatter(x, y, label="Data",color="cyan",marker='s')

            # regression line / predictions
            regression_line = slope * x + intercept
            # residuals
            residuals = y - regression_line

            plt.plot(
                x,
                regression_line,
                color="red",
                label=f"Regression Line ({r_value:.2f})",
            )

            # check residuals
            plt.scatter(x,residuals,label="Residuals",color="green",marker='d')
            plt.axhline(0,color='magenta',linestyle='--')

            plt.title(f"{title} (Correlation: {r_value:.4f})")
            plt.xlabel(labels[0])
            plt.ylabel(labels[1])
            plt.legend()
            plt.show()

def kan_regress_and_plot(data: tuple[np.ndarray, np.ndarray],
            labels: tuple[str, str],
            #sem_data: None | np.ndarray = None,
            title: str = "Generic KAN Regression Plot",
            generate_symbolic_formula:bool=False
            ):
            """"""

            device = "cuda" #"cpu" #"cuda"

            q_angle_means, q_error_means = data

            # Test out KAN for fitting
            epsilon = 1e-6 #1e-6
            x_train = torch.tensor(q_angle_means,dtype=torch.float32).reshape(-1,1).to(device)
            # x_train = torch.tensor(q_angle_means,dtype=torch.float32).reshape(-1,1).to(device) + epsilon
            y_train = torch.tensor(q_error_means,dtype=torch.float32).reshape(-1,1).to(device)

            # normalize data
            x_train, x_train_norm_stats = normalize_to_range(x_train,stats=None,target_range=(1.0,2.0))
            y_train, y_train_norm_stats = normalize_to_range(y_train,stats=None,target_range=(1.0,2.0))

            # Init Model
            model = KAN(width=[1,5,1], grid=6, k=3, device=device)
            # model = KAN(width=[1,5,1], grid=6, k=3, device=device,grid_eps=0.0)

            # Train
            dataset = {
                "train_input": x_train,
                "train_label": y_train,
                "test_input": x_train,
                "test_label": y_train,
            }

            # Align grid with data points
            model.update_grid_from_samples(dataset["train_input"])

            # fit the model
            # Train with Sparsity Regularization (lamb)
            # This forces "useless" connection weights toward zero
            model.fit(dataset,steps=100,opt="Adam",lr=1e-3,lamb=0.0)
            model.fit(dataset,steps=20,opt="LBFGS",lamb=1e-4)
            # model.fit(dataset,steps=50,opt="LBFGS",lamb=0.001)

            # model.update_grid_from_samples(dataset["train_input"])
            # model.fit(dataset,steps=15,opt="LBFGS",lamb=0.0)

            # # increase spline resolution
            # model = model.refine(20)

            # # final finetune (refit w/out regularization)
            # model.fit(dataset,steps=15,opt="LBFGS",lamb=0.0)

            # plot network and extract formula
            model.to("cpu")
            model.device = "cpu"

            model.eval()
            # dummy input to synch activations
            dummy_input = x_train.detach().cpu()[:6]
            model(dummy_input)

            dataset = {key:val.cpu() for key,val in dataset.items()}
            
            # prune weak connections
            model.plot()
            model = model.prune()
            model.plot()

            # refit/polish
            model.update_grid_from_samples(dataset['train_input'])
            #model.fit(dataset,steps=100,opt="Adam",lr=1e-3,lamb=0.0)
            model.fit(dataset,steps=20,opt="LBFGS",lamb=0.0)

            if generate_symbolic_formula:
                model.train()
                with torch.no_grad():
                    _=model(dataset["train_input"].cpu())

                # extract formula
                print("Suggested Symbolics")
                depth = len(model.width)-1
                for layer in np.arange(depth):
                    in_nodes = model.width[layer][0]
                    out_nodes = model.width[layer+1][0]
                    print(f"Layer {layer} ({in_nodes} -> {out_nodes})")
                    for idx in np.arange(in_nodes):
                        for jdx in np.arange(out_nodes):
                            print(f"Edge ({layer,idx,jdx})")
                            model.suggest_symbolic(layer,idx,jdx)
                print("End of Suggested Symbolics")

                # model.auto_symbolic(lib=['x','x^2','exp','abs','sin','cos','tanh','gaussian'])
                # # model.auto_symbolic(lib=['x','x^2','x^0.5','log','exp','abs','sin','cos']) # log, x^0.5/sqrt, obviously 1/x do not work with values near hear
                # kan_formula = model.symbolic_formula()[0][0]
                model.fix_symbolic(0,0,0,'sin')
                model.fix_symbolic(0,0,1,'cos')
                model.fix_symbolic(1,0,0,'x^0.5')
                model.fix_symbolic(1,1,0,'exp')

                # polish
                model.update_grid_from_samples(dataset['train_input'])
                # model.fit(dataset,steps=50,opt="Adam",lr=1e-3,lamb=0.0)
                model.fit(dataset,steps=20,opt="LBFGS",lamb=0.0)
            

                # # Finetune post equation conversion It should be noted that this wasn't necessary when nearly everything was 'exp' function options were sin,cos,x^2, and exp
                
                # # Call this IMMEDIATELY after model.fix_symbolic(...)
                # model.update_grid_from_samples(dataset['train_input'])
                # # model.fit(dataset,steps=50,opt="Adam",lr=1e-3,lamb=0.0)

                # # polish
                # model.fit(dataset,steps=20,opt="LBFGS",lamb=0.0)

                # print formula
                kan_formula = model.symbolic_formula()[0][0]
                print(f"KAN formula: {kan_formula}")

            # smooth prediction range for non-linear curve
            q_angle_means_smooth = np.linspace(q_angle_means.min(),q_angle_means.max(),len(q_angle_means))
            x_tensor = torch.tensor(q_angle_means_smooth, dtype=torch.float32).reshape(-1,1)

            # normalize input
            x_tensor,x_ten_norm_stats = normalize_to_range(x_tensor,stats=None,target_range=(1.0,2.0))

            # generate predictions 
            with torch.inference_mode():
                kan_error_curve_pred = model(x_tensor)
                kan_error_curve_pred = kan_error_curve_pred.detach().squeeze().cpu()
                kan_error_pred = model(x_train.cpu())
                kan_error_pred = kan_error_pred.detach().squeeze().cpu()

            # denormalize data
            x_train_norm_stats = {key:val.cpu() for key,val in x_train_norm_stats.items()}
            y_train_norm_stats = {key:val.cpu() for key,val in y_train_norm_stats.items()}

            # Calculate Metrics
            residuals = y_train.detach().squeeze().cpu().numpy()-kan_error_pred.numpy()
            sum_of_squared_residuals = np.sum(residuals**2)
            total_sum_of_squares = np.sum((q_error_means-q_error_means.mean())**2)
            r_squared = 1 - (sum_of_squared_residuals/total_sum_of_squares)
            print(f"R^2: {r_squared}")

            # KAN trend line and preditctions
            kan_error_curve_pred = denormalize_from_range(kan_error_curve_pred,stats=y_train_norm_stats,target_range=(1.0,2.0)).numpy()
            kan_error_pred = denormalize_from_range(kan_error_pred,stats=y_train_norm_stats,target_range=(1.0,2.0)).numpy()

            # residuals
            residuals = q_error_means - kan_error_pred

            # Calculate Metrics
            sum_of_squared_residuals = np.sum(residuals**2)
            total_sum_of_squares = np.sum((q_error_means-q_error_means.mean())**2)
            r_squared = 1 - (sum_of_squared_residuals/total_sum_of_squares)
            print(f"R^2: {r_squared}")

            # Data vs Predicted Data
            plt.figure(figsize=(10,6))
            plt.scatter(q_angle_means,q_error_means, color='black',label='Quantile Binned Data',marker="s")
            plt.scatter(q_angle_means,kan_error_pred, color='cyan',label='Quantile Binned Predictions',marker='o')

            # KAN Optimized Non-Linear Fit
            plt.plot(q_angle_means_smooth,kan_error_curve_pred,color='crimson',linewidth=2.5,label=f"KAN Optimized Inference ({r_squared:.4f})")

            # check residuals
            plt.scatter(q_angle_means,residuals,c='green',label="Residuals",marker='d')
            plt.axhline(0,color='magenta',linestyle='--')

            plt.title(f"{title} (Correlation: {r_squared:.4f})")
            plt.xlabel(labels[0])
            plt.ylabel(labels[1])
            plt.legend()
            plt.show()

            return r_squared,residuals

##########################################################################################
###### plotting stuff #######
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List

def visualize_single_map(data_map, title="3D Surface Map"):
    """
    Displays a 2D array as an interactive 3D surface plot.
    """
    z_values = np.asarray(data_map)
    
    # 1. Create the Surface Trace
    # z is the 2D array; x and y are optional and default to indices
    fig = go.Figure(data=[go.Surface(z=z_values, colorscale='Viridis')])

    # 2. Refine Layout
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='X (Column Index)',
            yaxis_title='Y (Row Index)',
            zaxis_title='Height (Value)'
        ),
        autosize=True,
        width=800,
        height=800,
        margin=dict(l=65, r=50, b=65, t=90)
    )

    # 3. Render
    # 'browser' renderer is recommended for high-performance WebGL in 2026
    fig.show(renderer="browser")

def visualize_map_quad_dashboard(
    map_ref: np.ndarray | List[List[float]],
    map_comp: np.ndarray | List[List[float]],
    map_diff: np.ndarray | List[List[float]],
    map_error: np.ndarray | List[List[float]],
    title: str = "3D Multi-Metric Map Analysis"
) -> None:
    """
    Renders a synchronized 4-panel 3D dashboard using pre-computed map data.

    This function expects four distinct 2D datasets. It is optimized for 2026
    workflows where data cleaning (e.g., IQR outlier removal) and metric 
    generation are handled in a separate pipeline.

    Args:
        map_ref: Baseline 2D array for the top-left panel.
        map_comp: Comparison 2D array for the top-right panel.
        map_diff: Absolute or relative difference 2D array for the bottom-left panel.
        map_error: Percent or statistical error 2D array for the bottom-right panel.
        title: Main dashboard title.

    Raises:
        ValueError: If input maps are not 2D or have mismatched shapes.
    """
    # Standardize inputs to NumPy arrays
    maps = [np.asarray(m) for m in [map_ref, map_comp, map_diff, map_error]]
    
    # Validation: Ensure all maps are 2D and share the same dimensions
    shape = maps[0].shape
    for i, m in enumerate(maps):
        if m.ndim != 2:
            raise ValueError(f"Map at index {i} must be 2D.")
        if m.shape != shape:
            raise ValueError(f"Map at index {i} shape {m.shape} mismatch with reference {shape}.")

    # Initialize a 2x2 grid of 3D scenes
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Reference Map", "Comparison Map", 
            "Difference (Δ)", "Error (%)"
        ),
        specs=[[{'type': 'surface'}, {'type': 'surface'}],
               [{'type': 'surface'}, {'type': 'surface'}]],
        horizontal_spacing=0.05,
        vertical_spacing=0.1
    )

    # Panel Configuration: Data, Colormap, and specific Colorbar position
    # Colorbar positioning (x, y) prevents overlapping in the quad-view
    configs = [
        (maps[0], 'Viridis', -0.08, 0.8),  # Top-left
        (maps[1], 'Viridis', 1.05, 0.8),   # Top-right
        (maps[2], 'RdBu', -0.08, 0.2),      # Bottom-left (Diverging for diff)
        (maps[3], 'YlOrRd', 1.05, 0.2)     # Bottom-right (Sequential for error)
    ]

    for idx, (data, cmap, cb_x, cb_y) in enumerate(configs):
        row = (idx // 2) + 1
        col = (idx % 2) + 1
        fig.add_trace(
            go.Surface(z=data, colorscale=cmap, colorbar=dict(x=cb_x, y=cb_y, len=0.4)),
            row=row, col=col
        )

    # Sync camera scenes to ensure rotation in one moves all others
    common_scene = dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z')
    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=24)),
        height=950,
        width=1100,
        scene=common_scene, scene2=common_scene,
        scene3=common_scene, scene4=common_scene,
        showlegend=False
    )

    # Open in browser for full GPU-accelerated 3D performance
    fig.show(renderer="browser")

def visualize_cleaned_map_comparison(
    map_ref: np.ndarray | List[List[float]], 
    map_comp: np.ndarray | List[List[float]], 
    title: str = "3D Comparison: IQR Extreme Outlier Removal (3.0x)"
    ) -> None:
        """
        Renders a 4-panel 3D dashboard after removing extreme outliers (3.0 * IQR).
        Outliers are clipped to the upper boundary to preserve array shape for plotting.
        """
        m1 = np.asarray(map_ref)
        m2 = np.asarray(map_comp)

        def remove_extreme_outliers(data: np.ndarray) -> np.ndarray:
            """Applies 3.0 * IQR rule to clip extreme high-end outliers."""
            q1 = np.percentile(data, 25)
            q3 = np.percentile(data, 75)
            iqr = q3 - q1
            # Upper fence for extreme outliers
            upper_boundary = q3 + (3.0 * iqr)
            # Clip values to the boundary to maintain 2D grid integrity
            return np.clip(data, a_min=None, a_max=upper_boundary)

        # 1. Clean data using Interquartile approach before calculations
        m1_clean = remove_extreme_outliers(m1)
        m2_clean = remove_extreme_outliers(m2)

        # 2. Calculate Difference and Percent Error on cleaned data
        diff = m1_clean - m2_clean
        # Using 1e-9 epsilon to prevent division by zero
        percent_error = (np.abs(diff) / (np.abs(m1_clean) + 1e-9)) * 100

        # 3. Initialize Subplot Dashboard
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "Reference (Cleaned)", "Comparison (Cleaned)", 
                "Absolute Difference", "Percent Error (%)"
            ),
            specs=[[{'type': 'surface'}, {'type': 'surface'}],
                [{'type': 'surface'}, {'type': 'surface'}]]
        )

        # Add traces with specified color scales
        fig.add_trace(go.Surface(z=m1_clean, colorscale='Viridis', colorbar_x=-0.07), row=1, col=1)
        fig.add_trace(go.Surface(z=m2_clean, colorscale='Viridis', colorbar_x=0.45), row=1, col=2)
        fig.add_trace(go.Surface(z=diff, colorscale='RdBu', colorbar_x=-0.07, colorbar_y=0.2), row=2, col=1)
        fig.add_trace(go.Surface(z=percent_error, colorscale='YlOrRd', colorbar_x=0.45, colorbar_y=0.2), row=2, col=2)

        fig.update_layout(
            title=title,
            height=900, width=1100,
            scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'),
            scene2=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'),
            scene3=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'),
            scene4=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z')
        )

        # Force browser rendering for interactive WebGL stability
        fig.show(renderer="browser")

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Literal

def visualize_maximizable_quad_dashboard(
    map_ref: np.ndarray | List[List[float]],
    map_comp: np.ndarray | List[List[float]],
    map_diff: np.ndarray | List[List[float]],
    map_error: np.ndarray | List[List[float]],
    title: str = "Interactive 3D Comparison Dashboard"
) -> None:
    """
    Renders a 4-panel 3D dashboard where each view can be maximized 
    to take up the entire figure canvas (100% width and height).
    """
    maps = [np.asarray(m) for m in [map_ref, map_comp, map_diff, map_error]]
    titles = ["Reference", "Comparison", "Difference", "Error (%)"]
    
    # 1. Define coordinate domains
    # Grid: [start, end] from 0.0 to 1.0
    grid_domains = [
        {'x': [0.0, 0.48], 'y': [0.52, 1.0]},  # Top-left (Scene 1)
        {'x': [0.52, 1.0], 'y': [0.52, 1.0]},  # Top-right (Scene 2)
        {'x': [0.0, 0.48], 'y': [0.0, 0.48]},  # Bottom-left (Scene 3)
        {'x': [0.52, 1.0], 'y': [0.0, 0.48]}   # Bottom-right (Scene 4)
    ]
    full_canvas = {'x': [0.0, 1.0], 'y': [0.0, 1.0]}

    fig = make_subplots(
        rows=2, cols=2, subplot_titles=titles,
        specs=[[{'type': 'surface'}, {'type': 'surface'}],
               [{'type': 'surface'}, {'type': 'surface'}]]
    )

    scales = ['Viridis', 'Viridis', 'RdBu', 'YlOrRd']
    for idx, (data, scale) in enumerate(zip(maps, scales)):
        row, col = (idx // 2) + 1, (idx % 2) + 1
        fig.add_trace(go.Surface(z=data, colorscale=scale, name=titles[idx]), row=row, col=col)

    # 2. Build the Button logic
    buttons = []
    
    # Button: Show Grid (Reset all domains)
    grid_layout = {"visible": [True] * 4}
    for i in range(4):
        scene_key = f"scene{i+1 if i > 0 else ''}.domain"
        grid_layout[scene_key] = grid_domains[i]
    
    buttons.append(dict(label="Show Grid", method="update", args=[grid_layout, {"title": title}]))

    # Buttons: Maximize individual views
    for i in range(4):
        visibility = [False] * 4
        visibility[i] = True
        
        # Args to hide others and expand the chosen one to the full canvas domain
        max_layout = {"visible": visibility}
        scene_key = f"scene{i+1 if i > 0 else ''}.domain"
        max_layout[scene_key] = full_canvas
        
        buttons.append(dict(
            label=f"Full {titles[i]}",
            method="update",
            args=[max_layout, {"title": f"Maximized: {titles[i]}"}]
        ))

    # 3. Final Layout and Colorbar cleanup
    fig.update_layout(
        updatemenus=[dict(
            type="buttons", direction="right", x=0.5, y=1.12, 
            xanchor="center", buttons=buttons, bgcolor="white", bordercolor="#ddd"
        )],
        height=900, width=1100,
        margin=dict(t=120, l=10, r=10, b=10),
        # Clean up camera sync (optional)
        **{f"scene{i+1 if i > 0 else ''}": dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z') for i in range(4)}
    )

    # Note: Using the 'browser' renderer is essential for 3D/WebGL performance in 2026.
    fig.show(renderer="browser")

if __name__ == "__main__":
    # Example map data
    res = 60
    m = np.random.standard_normal((res, res)).cumsum(axis=0).cumsum(axis=1)
    visualize_maximizable_quad_dashboard(m, m*1.05, m-m.mean(), np.abs(m)/10)

##########################################################################################

def extract_brightest_pixel_along_axis(
    data: torch.Tensor | np.ndarray, 
    axis: int | str
) -> torch.Tensor:
    """
    Extracts 3D coordinates (N, 3) of maximum values along a specified axis.

    This function identifies the index of the brightest pixel along one axis and 
    pairs it with the corresponding grid coordinates of the other two axes. 
    It is optimized to avoid redundant memory allocations and uses stride 
    optimization for the argmax operation.

    Args:
        data: A 3D volume as a torch.Tensor or np.ndarray of shape (D, H, W).
        axis: The dimension to reduce. Accepts integers (0, 1, 2) or 
            strings ("depth", "height", "width").

    Returns:
        torch.Tensor: A tensor of shape (N, 3) containing the [d, h, w] 
            indices for each brightest pixel, where N = product of the 
            remaining two dimensions. Always returned on the CPU.

    Raises:
        ValueError: If 'data' is not 3D or 'axis' is invalid.
    """
    # 1. Standardize input to CPU Tensor
    # torch.as_tensor avoids copying data if already a CPU Tensor or ndarray
    vol = torch.as_tensor(data, device="cpu")
    
    if vol.ndim != 3:
        raise ValueError(f"Input must be 3D, but got {vol.ndim}D.")

    # 2. Resolve semantic axis mapping
    axis_map = {"depth": 0, "height": 1, "width": 2}
    reduced_axis = axis_map.get(axis.lower()) if isinstance(axis, str) else axis
    
    if reduced_axis not in (0, 1, 2):
        raise ValueError("Axis must be 0, 1, 2 or 'depth', 'height', 'width'.")

    # 3. Perform Optimized Argmax
    # Stride optimization: argmax is fastest on the last dimension
    if reduced_axis != 2:
        max_indices = torch.argmax(vol.transpose(reduced_axis, 2), dim=2)
    else:
        max_indices = torch.argmax(vol, dim=2)

    # 4. Define remaining axes dimensions
    other_axes = [d for d in range(3) if d != reduced_axis]
    dim_a, dim_b = other_axes
    size_a, size_b = vol.shape[dim_a], vol.shape[dim_b]

    # 5. Pre-allocate Output Buffer
    # Initializing a single (N, 3) buffer is faster than torch.stack()
    coords = torch.empty((size_a * size_b, 3), dtype=torch.long, device="cpu")

    # 6. Generate Grids using Broadcasting (Memory Efficient)
    # .expand() creates a view; it does not allocate new memory for the grid
    grid_a = torch.arange(size_a, device="cpu").view(-1, 1).expand(size_a, size_b)
    grid_b = torch.arange(size_b, device="cpu").view(1, -1).expand(size_a, size_b)

    # 7. Batch Assignment via Reshaping
    # .reshape(-1) provides a flattened view to fill the pre-allocated columns
    coords[:, reduced_axis] = max_indices.reshape(-1)
    coords[:, dim_a] = grid_a.reshape(-1)
    coords[:, dim_b] = grid_b.reshape(-1)

    return coords


def find_intensity_outliers(data, kernel_size:int = 3, std_threshold:float = 3.):
    """
    """
    F = torch.nn.functional

    # ensure data is torch tensor
    data = torch.as_tensor(data,dtype=torch.float)

    # Calculate residual index
    mean_filter = torch.ones((1,1,kernel_size,kernel_size,kernel_size)) / (kernel_size**3)
    local_mean = F.conv3d(data,mean_filter,padding=1)
    residual = torch.abs(data-local_mean)

    # Calculate gradient index
    dz = data[:,:,1:,:,:] - data[:,:,:-1,:,:]
    dh = data[:,:,:,1:,:] - data[:,:,:,:-1,:]
    dw = data[:,:,:,:,1:] - data[:,:,:,:,:-1]

    # Repad to original size
    grad_z = F.pad(dz, (0,0,0,0,0,1))
    grad_h = F.pad(dh, (0,0,0,1,0,0))
    grad_w = F.pad(dw, (0,1,0,0,0,0))

    gradient_idx = torch.sqrt(grad_z**2 + grad_h**2 + grad_w**2)

    # Dual-Thresholding
    residual_threshold = residual.mean() + (std_threshold * residual.std())
    gradient_threshold = gradient_idx.mean() + (std_threshold * gradient_idx.std())

    outliers = (residual > residual_threshold) & (gradient_idx > gradient_threshold)

    outliers = outliers.numpy().astype("uint8")

    return outliers, residual, gradient_idx

def find_spatial_outliers(
    binary_data: torch.Tensor, 
    kernel_size: int = 5, 
    std_threshold: float = 3.0
) -> tuple[np.ndarray, torch.Tensor, torch.Tensor]:
    """
    Detects isolated voxels in 3D volumes using density and gradient analysis.
    
    Identifies foreground voxels that are statistically likely to be noise based 
    on their isolation (low local density) and high surface contrast (gradients).

    Args:
        binary_data: Input tensor (3D, 4D, or 5D). Values > 0 are foreground.
        kernel_size: Size of the cubic neighborhood for density calculation.
        std_threshold: Sensitivity multiplier for the standard deviation cutoff.

    Returns:
        tuple containing:
            - outliers: Boolean NumPy array of the original input shape.
            - spatial_residual: Tensor of density deviations.
            - spatial_gradient: Tensor of local isolation magnitudes.
    """
    F = torch.nn.functional
    # ensure data is torch tensor
    binary_data = torch.as_tensor(binary_data>0,dtype=torch.bool)
    
    # 1. Dimension Standardization
    # Captures original shape to restore it before returning
    orig_shape = binary_data.shape
    if binary_data.ndim == 3:
        binary_data = binary_data.unsqueeze(0).unsqueeze(0)
    elif binary_data.ndim == 4:
        binary_data = binary_data.unsqueeze(0)

    device = binary_data.device
    binary_float = (binary_data > 0).float()

    # 2. Spatial Residual (Local Density Deviation)
    # Uses a 3D box filter to determine the average foreground density nearby.
    # Higher residual = voxel differs significantly from its neighbors.
    weight = torch.ones((1, 1, kernel_size, kernel_size, kernel_size), device=device)
    weight /= (kernel_size**3)
    
    local_density = F.conv3d(binary_float, weight, padding=kernel_size // 2)
    spatial_residual = torch.abs(binary_float - local_density)

    # 3. Spatial Gradient (Boundary Isolation)
    # Measures how 'disconnected' a voxel is by counting state changes (0 to 1).
    binary_bool = binary_float.bool()
    dz = binary_bool[:, :, 1:, :, :] ^ binary_bool[:, :, :-1, :, :]
    dh = binary_bool[:, :, :, 1:, :] ^ binary_bool[:, :, :, :-1, :]
    dw = binary_bool[:, :, :, :, 1:] ^ binary_bool[:, :, :, :, :-1]

    # Re-pad gradients to match original volume dimensions
    grad_z = F.pad(dz.float(), (0, 0, 0, 0, 0, 1))
    grad_h = F.pad(dh.float(), (0, 0, 0, 1, 0, 0))
    grad_w = F.pad(dw.float(), (0, 1, 0, 0, 0, 0))

    # Compute Euclidean magnitude of the gradient
    spatial_gradient = torch.sqrt(grad_z**2 + grad_h**2 + grad_w**2)

    # 4. Statistical Filtering
    # We evaluate outliers only within the existing foreground mask.
    mask = (binary_bool == True)
    
    if not mask.any():
        return np.zeros(orig_shape, dtype=bool), spatial_residual, spatial_gradient

    # Calculate dynamic thresholds using Z-score logic
    res_vals = spatial_residual[mask]
    grad_vals = spatial_gradient[mask]

    res_thresh = res_vals.mean() + (std_threshold * res_vals.std())
    grad_thresh = grad_vals.mean() + (std_threshold * grad_vals.std())

    # Logic: Voxel is an outlier if it is foreground AND exceeds both thresholds
    outliers_mask = mask & (spatial_residual > res_thresh) & (spatial_gradient > grad_thresh)

    # 5. Output Preparation
    # Reshape back to the user's input dimensions and move to CPU/NumPy
    return outliers_mask.view(orig_shape).cpu().numpy().astype(bool), spatial_residual, spatial_gradient

##########################################################################################

#### TODO Needs refinement
import numpy as np
from scipy.spatial import KDTree

def statistical_outlier_removal(bool_mask, k=20, std_mul=2.0):
    # 1. Extract coordinates of surface points (True values)
    coords = np.argwhere(bool_mask)
    
    # 2. Build KD-Tree for fast spatial lookup
    tree = KDTree(coords)
    
    # 3. Query distances to k-nearest neighbors for each point
    # k+1 because the query includes the point itself (distance 0)
    distances, _ = tree.query(coords, k=k+1)
    
    # Calculate mean distance for each point (excluding self at index 0)
    mean_distances = np.mean(distances[:, 1:], axis=1)
    
    # 4. Statistical thresholding
    global_mean = np.mean(mean_distances)
    global_std = np.std(mean_distances)
    threshold = global_mean + std_mul * global_std
    
    # 5. Identify outliers and create a mask
    is_outlier_list = mean_distances > threshold
    
    # Map back to original 3D volume shape
    outlier_mask = np.zeros_like(bool_mask, dtype=bool)
    outlier_mask[tuple(coords[is_outlier_list].T)] = True
    
    return outlier_mask

def identify_depth_outliers_torch(depth_map, kernel_size=5, threshold=3.0):
    """
    Identifies outliers in a 2D depth map using combined gradient and residual data.
    
    Args:
        depth_map (torch.Tensor): 2D tensor [H, W] or 3D tensor [1, H, W].
        kernel_size (int): Size of the median filter window for residual calculation.
        threshold (float): Z-score threshold for outlier detection (default 3.0).
        
    Returns:
        torch.Tensor: A boolean mask where True indicates an outlier.
    """
    F = torch.nn.functional

    if depth_map.dim() == 2:
        depth_map = depth_map.unsqueeze(0).unsqueeze(0)  # Shape [1, 1, H, W]
    elif depth_map.dim() == 3:
        depth_map = depth_map.unsqueeze(0)

    # 1. Compute Gradients (Local Changes)
    # Using central differences for gradient estimation
    dy, dx = torch.gradient(depth_map.squeeze())
    grad_mag = torch.sqrt(dx**2 + dy**2)

    # 2. Compute Residuals (Global Deviation)
    # Median filtering is robust for depth data outliers
    # We use unfold to create a sliding window for the median calculation
    pad = kernel_size // 2
    padded = F.pad(depth_map, (pad, pad, pad, pad), mode='replicate')
    patches = padded.unfold(2, kernel_size, 1).unfold(3, kernel_size, 1)
    # patches shape: [1, 1, H, W, kernel_size, kernel_size]
    median_surface = patches.contiguous().view(1, 1, depth_map.size(2), depth_map.size(3), -1).median(dim=-1)[0]
    residuals = torch.abs(depth_map - median_surface).squeeze()

    # 3. Combine and Identify Outliers
    # Standardize both metrics to Z-scores for a unified scale
    def z_score(x):
        return (x - x.mean()) / (x.std() + 1e-8)

    combined_score = z_score(grad_mag) + z_score(residuals)
    
    # Generate binary mask based on the combined threshold
    outlier_mask = combined_score > threshold
    
    return outlier_mask

##########################################################################################