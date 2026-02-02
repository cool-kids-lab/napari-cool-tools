""" """

import gc
from typing import Literal

import kornia.morphology as morph
import numpy as np
import torch
import torch.nn.functional as F


def validate_and_standardize_volume(
    input_volume: torch.Tensor | np.ndarray,
    target_standardized_dimensions: int = 4,
    minimum_allowable_dimensions: int = 3,
    device_name: str = "cpu",
) -> torch.Tensor:
    """Validates and standardizes input tensors to a specific rank.

    Standardizes inputs by converting to float32 and prepending singleton
    dimensions until the target dimension count is reached.

    Args:
        input_volume (Union[torch.Tensor, np.ndarray]): The input data.
        target_standardized_dimensions (int): The required rank (e.g., 4 for CDHW).
        minimum_allowable_dimensions (int): The floor for input rank validation.
        device_name (str): Target device (e.g., 'cuda', 'cpu').

    Returns:
        torch.Tensor: A tensor of rank 'target_standardized_dimensions' on the target device.

    Raises:
        TypeError: If input is not a torch.Tensor or np.ndarray.
        ValueError: If input rank is below minimum_allowable_dimensions or
                    above target_standardized_dimensions.
    """
    # 1. Type Conversion
    if isinstance(input_volume, np.ndarray):
        standardized_tensor = torch.from_numpy(input_volume).float()
    elif isinstance(input_volume, torch.Tensor):
        standardized_tensor = input_volume.float()
    else:
        raise TypeError(f"Input must be Tensor or ndarray, got {type(input_volume)}.")

    current_dimensions = standardized_tensor.ndim

    # 2. Rank Validation
    if current_dimensions < minimum_allowable_dimensions:
        raise ValueError(
            f"Input rank {current_dimensions} is below the minimum "
            f"allowed ({minimum_allowable_dimensions})."
        )

    if current_dimensions > target_standardized_dimensions:
        raise ValueError(
            f"Input rank {current_dimensions} exceeds the target "
            f"standardized rank ({target_standardized_dimensions})."
        )

    # 3. Prepend Dimensions to Match Target
    # We add singleton dimensions to the front (index 0) until we hit the target
    while standardized_tensor.ndim < target_standardized_dimensions:
        standardized_tensor = standardized_tensor.unsqueeze(0)

    # 4. Device Management
    return standardized_tensor.to(torch.device(device_name))


def generate_patch_maximum_intensity_projections(
    input_volume: torch.Tensor | np.ndarray,
    patch_height_size: int,
    stride_step: int = 1,
    use_symmetric_padding: bool = True,
    mini_batch_size: int = 16,
    device_name: str = "cpu",
    return_numpy: bool = False,
) -> torch.Tensor | np.ndarray:
    """Evaluates volume patches using standardized 4D input handling."""

    # Standardize to 4D (C, D, H, W) for this specific math logic
    processing_tensor = validate_and_standardize_volume(
        input_volume=input_volume,
        target_standardized_dimensions=4,
        minimum_allowable_dimensions=3,
        device_name="cpu",
    )

    channels, depth, original_height, width = processing_tensor.shape

    # Apply padding if requested
    if use_symmetric_padding:
        pad_val = patch_height_size - 1
        pad_at_start = pad_val // 2
        pad_at_end = pad_val - pad_at_start
        processing_tensor = F.pad(processing_tensor, (0, 0, pad_at_start, pad_at_end))

    # Generate sliding window view along Height (dim 2)
    volume_patch_view = processing_tensor.unfold(
        dimension=2, size=patch_height_size, step=stride_step
    )

    # Permute for iteration: (patch_count, channels, depth, width, patch_height_size)
    iteration_view = volume_patch_view.permute(2, 0, 1, 3, 4)
    total_patches_count = iteration_view.shape[0]

    current_batch_size = (
        total_patches_count if mini_batch_size == -1 else mini_batch_size
    )
    target_device = torch.device(device_name)
    processed_mip_list = []

    # Iterative GPU Processing
    for i in range(0, total_patches_count, current_batch_size):
        end_idx = min(i + current_batch_size, total_patches_count)
        gpu_patch_batch = iteration_view[i:end_idx].to(target_device)

        mip_batch_values, _ = torch.max(gpu_patch_batch, dim=-1)
        processed_mip_list.append(mip_batch_values.cpu())

        del gpu_patch_batch
        if target_device.type == "cuda":
            torch.cuda.empty_cache()

    # Reconstruction: (patch_count, C, D, W) -> (C, D, patch_count, W)
    reconstructed_patches = torch.cat(processed_mip_list, dim=0)
    final_projections = reconstructed_patches.permute(1, 2, 0, 3)
    final_output = final_projections.squeeze()

    # Final Cleanup
    del processing_tensor, volume_patch_view, iteration_view, reconstructed_patches
    gc.collect()

    return final_output.numpy() if return_numpy else final_output


def get_top_k_patch_intensity_indices(
    input_volume: torch.Tensor | np.ndarray,
    patch_height_size: int,
    intensity_threshold: float = 0.0,
    stride_step: int = 1,
    top_k_count: int = 5,
    use_symmetric_padding: bool = True,
    device_name: str = "cpu",
    return_numpy: bool = False,
) -> tuple[torch.Tensor | np.ndarray, torch.Tensor | np.ndarray]:
    """Calculates top-K patch intensities using modern Python 3.12+ type hints.

    Uses built-in tuple generics and memory-efficient summation to avoid
    modifying or cloning the input volume.

    Args:
        input_volume: 3D or 4D input data (ndarray or Tensor).
        patch_height_size: Sliding window size along the height axis.
        intensity_threshold: Voxel values below this are ignored.
        stride_step: Distance between consecutive window starts.
        top_k_count: Number of highest intensity patches to return.
        use_symmetric_padding: If True, pads height to maintain original length.
        device_name: Processing device (e.g., 'cuda', 'mps', 'cpu').
        return_numpy: Whether to return results as NumPy arrays.

    Returns:
        tuple: (top_k_values, top_k_indices) on CPU or as ndarrays.
    """
    # 1. Standardize to CPU initially to minimize VRAM usage for the reduction
    processing_tensor = validate_and_standardize_volume(
        input_volume=input_volume,
        target_standardized_dimensions=4,
        minimum_allowable_dimensions=3,
        device_name="cpu",
    )

    # 2. Memory-Efficient Thresholded Summation
    # torch.where creates a transient tensor that doesn't modify the input
    target_device = torch.device(device_name)

    # Reduction along Depth (1) and Width (3)
    spatial_sum = torch.sum(
        torch.where(processing_tensor > intensity_threshold, processing_tensor, 0.0),
        dim=(1, 3),
    ).to(target_device)

    # 3. Conditional Symmetric Padding
    if use_symmetric_padding:
        padding_val = patch_height_size - 1
        pad_at_start = padding_val // 2
        pad_at_end = padding_val - pad_at_start
        # Pad the height dimension (dim 1)
        spatial_sum = F.pad(spatial_sum, (pad_at_start, pad_at_end))

    # 4. Windowing and Averaging on 1D/2D profile
    # Resulting shape: (Channel, patch_count, patch_height_size)
    height_windows = spatial_sum.unfold(
        dimension=1, size=patch_height_size, step=stride_step
    )

    # Mean of intensities within each window
    patch_averages = torch.mean(height_windows, dim=-1).squeeze()

    # 5. Top-K Extraction
    actual_k = min(top_k_count, patch_averages.size(-1))
    top_k_values, top_k_indices = torch.topk(patch_averages, k=actual_k)

    # 6. Final Cleanup
    del processing_tensor, spatial_sum, height_windows
    gc.collect()
    if target_device.type == "cuda":
        torch.cuda.empty_cache()

    if return_numpy:
        return top_k_values.cpu().numpy(), top_k_indices.cpu().numpy()

    return top_k_values.cpu(), top_k_indices.cpu()


def get_top_k_slices_radial_symmetry(
    data: np.ndarray | torch.Tensor,
    k: int = 5,
    window_size: int = 3,
    stride: int = 1,
    batch_size: int = -1,
    device: str = "cuda",
    return_numpy: bool = True,
    angle_bins: int = 12,
    density_percentile: float = 50.0,
    generate_mask: bool = False,
    window_label: int = 10,
    center_label: int = 6,
) -> (
    tuple[np.ndarray | torch.Tensor, np.ndarray | torch.Tensor]
    | tuple[
        np.ndarray | torch.Tensor, np.ndarray | torch.Tensor, np.ndarray | torch.Tensor
    ]
):
    """Finds top k slices with distant, radially uniform clusters above a density cutoff.

    Scores are calculated based on mean radius and angular entropy. Windows are
    ignored if their total point count is below a calculated percentile threshold
    of all non-zero counts in the volume.

    Args:
        data: A 3D array or tensor of shape (depth, height, width).
        k: Number of top windows to return.
        window_size: Height-wise window size for averaging.
        stride: Stride along the height dimension.
        batch_size: Number of slices to process per GPU batch.
        device: Target device for computation.
        return_numpy: If True, returns results as numpy arrays.
        angle_bins: Number of bins for calculating angular entropy.
        density_percentile: Percentile (0-100) of non-zero counts used as a cutoff.
        generate_mask: If True, returns a uint8 mask of the top window.
        window_label: Label for the top window in the mask.
        center_label: Label for the center slice of the top window.

    Returns:
        If generate_mask is False: (top_k_values, top_k_indices)
        If generate_mask is True: (top_k_values, top_k_indices, uint8_mask)
    """
    tensor_cpu = torch.as_tensor(data).float()
    depth_dim, height_dim, width_dim = tensor_cpu.shape

    # 1. Calculate Density Cutoff: Determine percentile of non-zero counts per slice
    # Efficiently count non-zeros along the height dimension
    nonzero_counts = torch.count_nonzero(tensor_cpu, dim=(0, 2)).float()
    # Calculate the q-th quantile (percentile / 100)
    density_cutoff = torch.quantile(nonzero_counts, density_percentile / 100.0)

    geometric_center = torch.tensor([depth_dim / 2.0, width_dim / 2.0], device=device)
    slice_scores = torch.zeros(height_dim, device="cpu")

    effective_batch_size = height_dim if batch_size == -1 else batch_size

    for height_start in range(0, height_dim, effective_batch_size):
        height_end = min(height_start + effective_batch_size, height_dim)
        active_batch = tensor_cpu[:, height_start:height_end, :].to(device)

        current_batch_length = height_end - height_start
        for batch_index in range(current_batch_length):
            global_height_index = height_start + batch_index

            # Apply cutoff: Ignore slices with too few points
            if nonzero_counts[global_height_index] < density_cutoff:
                continue

            masked_coordinates = torch.nonzero(active_batch[:, batch_index, :]).float()

            if masked_coordinates.shape[0] > 1:
                centered_coords = masked_coordinates - geometric_center
                radial_distances = torch.linalg.norm(centered_coords, dim=1)
                mean_radius = torch.mean(radial_distances)

                angles = torch.atan2(centered_coords[:, 0], centered_coords[:, 1])
                angle_counts = torch.histc(
                    angles, bins=angle_bins, min=-np.pi, max=np.pi
                )
                probs = angle_counts / (angle_counts.sum() + 1e-9)
                angular_entropy = -torch.sum(probs * torch.log(probs + 1e-9))

                # New Score: radius * entropy (removed num_pixels multiplier)
                slice_scores[global_height_index] = mean_radius * angular_entropy

        del active_batch
        if "cuda" in device:
            torch.cuda.empty_cache()
            gc.collect()

    # 2. Window Averaging: Use unfold to calculate averages over windows
    score_windows = slice_scores.unfold(dimension=0, size=window_size, step=stride)
    window_averages = score_windows.mean(dim=-1)

    top_k_values, top_k_indices = torch.topk(window_averages, k, largest=True)

    # 3. Mask Generation: Labeled 3D mask of the highest rated window
    mask_tensor = None
    if generate_mask:
        mask_tensor = torch.zeros((depth_dim, height_dim, width_dim), dtype=torch.uint8)
        best_window_start = int(top_k_indices[0])
        best_window_end = best_window_start + window_size

        mask_tensor[:, best_window_start:best_window_end, :] = window_label
        mask_tensor[:, best_window_start + (window_size // 2), :] = center_label

    if return_numpy:
        results = (top_k_values.numpy(), top_k_indices.numpy())
        return (*results, mask_tensor.numpy()) if generate_mask else results

    return (
        (top_k_values, top_k_indices, mask_tensor)
        if generate_mask
        else (top_k_values, top_k_indices)
    )


def get_top_k_slices_perpendicular_radial(
    data: np.ndarray | torch.Tensor,
    k: int = 5,
    window_size: int = 3,
    stride: int = 1,
    batch_size: int = -1,
    device: str = "cpu",
    return_numpy: bool = True,
    angle_bins: int = 12,
    density_percentile: float = 0.1,
    generate_mask: bool = True,
    window_label: int = 10,
    center_label: int = 6,
    return_scores: bool = False,
) -> tuple:
    """Identifies clusters perpendicular to height with centered window scoring.

    The window average for slice 'i' is calculated by centering a window of size
    'window_size' on index 'i'. This ensures the returned scores array matches
    the input height (H), allowing top_k indices to match slice_scores.argmax().

    Args:
        data: A 3D array or tensor of shape (depth, height, width).
        k: Number of top windows to return.
        window_size: Size of the centered sliding window. Must be odd for symmetry.
        stride: Stride of the window. Use 1 to maintain output length H.
        batch_size: Number of height slices to process per batch.
        device: Target device for computation.
        return_numpy: If True, returns results as numpy arrays.
        angle_bins: Bins for calculating angular entropy (symmetry).
        density_percentile: Percentile cutoff for non-zero counts (default 0.1).
        generate_mask: If True, returns a 3D uint8 mask (default True).
        window_label: Label for the highest-rated window in the mask (default 10).
        center_label: Label for the center slice of that window (default 6).
        return_scores: If True, returns normalized per-slice scores.

    Returns:
        A tuple containing (top_k_values, top_k_indices).
        Includes [uint8_mask] if generate_mask is True.
        Includes [height_scores] if return_scores is True.
    """
    tensor_cpu = torch.as_tensor(data).float()
    depth_dim, height_dim, width_dim = tensor_cpu.shape

    # 1. Volume-wide Density Thresholding
    nonzero_counts = torch.count_nonzero(tensor_cpu, dim=(0, 2)).float()
    density_cutoff = torch.quantile(nonzero_counts, density_percentile / 100.0)

    geometric_center_dw = torch.tensor(
        [depth_dim / 2.0, width_dim / 2.0], device=device
    )
    slice_scores = torch.zeros(height_dim, device="cpu")

    effective_batch_size = height_dim if batch_size == -1 else batch_size

    # 2. Batch Processing for Core Metrics
    for height_start in range(0, height_dim, effective_batch_size):
        height_end = min(height_start + effective_batch_size, height_dim)
        active_batch = tensor_cpu[:, height_start:height_end, :].to(device)

        for batch_index in range(height_end - height_start):
            global_height_index = height_start + batch_index
            if nonzero_counts[global_height_index] < density_cutoff:
                continue

            masked_coordinates = torch.nonzero(active_batch[:, batch_index, :]).float()
            if masked_coordinates.shape[0] > 1:
                centered_dw = masked_coordinates - geometric_center_dw
                radial_distances = torch.linalg.norm(centered_dw, dim=1)
                mean_radius = torch.mean(radial_distances)

                angles = torch.atan2(centered_dw[:, 0], centered_dw[:, 1])
                angle_counts = torch.histc(
                    angles, bins=angle_bins, min=-np.pi, max=np.pi
                )
                probs = angle_counts / (angle_counts.sum() + 1e-9)
                angular_entropy = -torch.sum(probs * torch.log(probs + 1e-9))

                slice_scores[global_height_index] = mean_radius * angular_entropy

        del active_batch
        if "cuda" in device:
            torch.cuda.empty_cache()
            gc.collect()

    # 3. Normalization (Min-Max)
    s_min, s_max = slice_scores.min(), slice_scores.max()
    slice_scores = (slice_scores - s_min) / (s_max - s_min + 1e-9)

    # 4. Centered Sliding Window Averaging
    # Pad to maintain length H: (window_size // 2) on each side
    padding_val = window_size // 2
    # F.pad expects (left, right) for 1D tensors
    padded_scores = F.pad(
        slice_scores.unsqueeze(0), (padding_val, padding_val), mode="constant", value=0
    )

    # unfold creates windows centered on the original indices
    score_windows = padded_scores.unfold(
        dimension=1, size=window_size, step=stride
    ).squeeze(0)
    window_averages = score_windows.mean(dim=-1)

    # 5. Result Extraction
    # If stride=1, window_averages.shape[0] == height_dim
    top_k_values, top_k_indices = torch.topk(window_averages, k, largest=True)

    results = [
        top_k_values.numpy() if return_numpy else top_k_values,
        top_k_indices.numpy() if return_numpy else top_k_indices,
    ]

    if generate_mask:
        mask_tensor = torch.zeros((depth_dim, height_dim, width_dim), dtype=torch.uint8)
        best_center_idx = int(top_k_indices[0])
        # Define window boundaries centered on the best index
        win_start = max(0, best_center_idx - padding_val)
        win_end = min(height_dim, best_center_idx + padding_val + 1)

        mask_tensor[:, win_start:win_end, :] = window_label
        mask_tensor[:, best_center_idx, :] = center_label
        results.append(mask_tensor.numpy() if return_numpy else mask_tensor)

    if return_scores:
        # These scores are now one-to-one with the height axis
        results.append(window_averages.numpy() if return_numpy else window_averages)

    return tuple(results)


def clean_hardware_resources() -> None:
    """Triggers garbage collection and clears device-specific hardware caches.

    This ensures that non-CPU backends (CUDA or MPS) free unused memory before
    and after large tensor operations, critical for 24GB VRAM limits.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif hasattr(torch, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()


@torch.inference_mode()
def calculate_height_density_profile_og(
    voxel_grid: np.ndarray | torch.Tensor,
    window_height_voxels: int,
    return_as_numpy: bool = False,
    use_bfloat16: bool = True,
    enable_visualization: bool = False,
    window_highlight_value: int = 10,
    center_slice_value: int = 6,
) -> (
    torch.Tensor
    | np.ndarray
    | tuple[torch.Tensor | np.ndarray, torch.Tensor | np.ndarray]
):
    """Calculates density profiles using mixed precision (FP16 or BF16) along axis 1.

    The function uses a 3D convolution kernel to calculate the local occupancy ratio
    within a vertical window. It leverages BF16 on 2026 hardware for superior
    dynamic range during the summation of millions of voxels.

    Args:
        voxel_grid: 3D input grid of shape (Depth, Height, Width).
        window_height_voxels: The vertical size of the sliding window.
        return_as_numpy: If True, returns results as NumPy arrays.
        use_bfloat16: If True, uses BF16; otherwise uses FP16. (BF16 recommended for 3090+).
        enable_visualization: If True, returns a uint8 grid highlighting the max density region.
        window_highlight_value: uint8 value for the entire max density window.
        center_slice_value: uint8 value for the center slice of that window.

    Returns:
        A 1D density profile, or a tuple of (density profile, 3D visualization grid).
    """
    # 1. Tensor Conversion and Device Management
    input_tensor = torch.as_tensor(voxel_grid, dtype=torch.float32)
    target_device = input_tensor.device

    # Selection of 16-bit dtype (BF16 is native on RTX 3090)
    computation_dtype = torch.bfloat16 if use_bfloat16 else torch.float16

    if target_device.type != "cpu":
        clean_hardware_resources()

    # 2. Axis Alignment
    # Input: (Depth, Height, Width). Height is at axis 1.
    # Permute to (Height, Depth, Width) so index 0 is the Height axis for Conv3d.
    working_tensor = input_tensor.permute(1, 0, 2)
    current_height, current_depth, current_width = working_tensor.shape

    # Reshape for Conv3d: (Batch=1, Channel=1, Depth=Height, H=Depth, W=Width)
    working_volume = working_tensor.unsqueeze(0).unsqueeze(0)

    # 3. Convolutional Sliding Window with Autocast
    with torch.amp.autocast(device_type=target_device.type, dtype=computation_dtype):
        # Kernel size (window_height_voxels, 1, 1) sums only along the vertical axis.
        vertical_kernel = torch.ones(
            (1, 1, window_height_voxels, 1, 1),
            device=target_device,
            dtype=computation_dtype,
        )

        # Calculate sum of occupied voxels within the window.
        # Padding ensures the output height matches input height.
        density_sum_3d = F.conv3d(
            working_volume.to(computation_dtype),
            vertical_kernel,
            padding=(window_height_voxels // 2, 0, 0),
        )

    # Slice to match the exact original height and cast back to FP32 for normalization
    density_sum_3d = density_sum_3d[:, :, :current_height, :, :].to(torch.float32)

    # 4. Profile Reduction and Normalization
    # Average across the spatial dimensions (Depth and Width) to get a 1D profile
    height_profile_sum = density_sum_3d.sum(dim=(0, 1, 3, 4))
    normalization_factor = window_height_voxels * current_depth * current_width
    height_density_profile = height_profile_sum / (normalization_factor + 1e-9)

    # 5. Optional Visualization Generation
    visualization_grid = None
    if enable_visualization:
        # Find index of max density
        max_density_center_index = torch.argmax(height_density_profile).item()

        # Build 1D mask along height axis
        height_mask_1d = torch.zeros(
            current_height, dtype=torch.uint8, device=target_device
        )

        # Calculate window span
        half_window = window_height_voxels // 2
        window_start = max(0, max_density_center_index - half_window)
        window_end = min(current_height, max_density_center_index + half_window + 1)

        # Apply highlight values
        height_mask_1d[window_start:window_end] = window_highlight_value
        height_mask_1d[max_density_center_index] = center_slice_value

        # Project 1D mask back to 3D and restore original (Depth, Height, Width) orientation
        visualization_grid = height_mask_1d.view(current_height, 1, 1).expand(
            -1, current_depth, current_width
        )
        visualization_grid = visualization_grid.permute(1, 0, 2)

    # 6. Final Cleanup and Output Formatting
    if target_device.type != "cpu":
        del working_volume, working_tensor, density_sum_3d
    clean_hardware_resources()

    output_profile = (
        height_density_profile.cpu().numpy()
        if return_as_numpy
        else height_density_profile
    )

    if enable_visualization:
        output_viz = (
            visualization_grid.cpu().numpy() if return_as_numpy else visualization_grid
        )
        return output_profile, output_viz

    return output_profile


import torch
import torch.nn.functional as F
import numpy as np
import gc


def clean_hardware_resources() -> None:
    """Triggers garbage collection and clears device-specific hardware caches.

    Ensures that non-CPU backends (CUDA or MPS) free unused memory before
    and after large tensor operations, critical for 24GB VRAM limits.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif hasattr(torch, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()


@torch.inference_mode()
def calculate_height_density_profile_og2(
    voxel_grid: np.ndarray | torch.Tensor,
    window_height_voxels: int,
    return_as_numpy: bool = False,
    use_bfloat16: bool = True,
    enable_visualization: bool = False,
    window_highlight_value: int = 10,
    center_slice_value: int = 6,
) -> (
    torch.Tensor
    | np.ndarray
    | tuple[torch.Tensor | np.ndarray, torch.Tensor | np.ndarray]
):
    """Calculates density profiles using mixed precision (FP16 or BF16) along axis 1.

    The function uses a 3D convolution kernel to calculate the local occupancy ratio
    within a vertical window. It leverages BF16 on 2026 hardware for superior
    dynamic range during the summation of millions of voxels.

    Args:
        voxel_grid: 3D input grid of shape (Depth, Height, Width).
        window_height_voxels: The size of the vertical sliding window.
        return_as_numpy: If True, returns results as NumPy arrays.
        use_bfloat16: If True, uses BF16; otherwise uses FP16. (BF16 recommended for 3090+).
        enable_visualization: If True, returns a uint8 grid highlighting the max density region.
        window_highlight_value: uint8 value for the entire max density window.
        center_slice_value: uint8 value for the center slice of that window.

    Returns:
        A 1D density profile, or a tuple of (density profile, 3D visualization grid).
    """
    # 1. Tensor Conversion and Device Management
    input_tensor = torch.as_tensor(voxel_grid, dtype=torch.float32)
    target_device = input_tensor.device

    # Selection of 16-bit dtype (BF16 is native on RTX 3090)
    computation_dtype = torch.bfloat16 if use_bfloat16 else torch.float16

    if target_device.type != "cpu":
        clean_hardware_resources()

    # 2. Axis Alignment
    # Input: (Depth, Height, Width). Height is at axis 1.
    # Permute to (Height, Depth, Width) so index 0 is the Height axis for Conv3d.
    working_tensor = input_tensor.permute(1, 0, 2)
    current_height, current_depth, current_width = working_tensor.shape

    # Reshape for Conv3d: (Batch=1, Channel=1, Depth=Height, H=Depth, W=Width)
    working_volume = working_tensor.unsqueeze(0).unsqueeze(0)

    # 3. Convolutional Sliding Window with Autocast
    with torch.amp.autocast(device_type=target_device.type, dtype=computation_dtype):
        # Kernel size (window_height_voxels, 1, 1) sums only along the vertical axis.
        vertical_kernel = torch.ones(
            (1, 1, window_height_voxels, 1, 1),
            device=target_device,
            dtype=computation_dtype,
        )

        # Calculate sum of occupied voxels within the window.
        density_sum_3d = F.conv3d(
            working_volume.to(computation_dtype),
            vertical_kernel,
            padding=(window_height_voxels // 2, 0, 0),
        )

    # Cleanup kernel and input conversion immediately after math
    del vertical_kernel

    # Slice to match the exact original height and cast back to FP32 for normalization
    density_sum_3d = density_sum_3d[:, :, :current_height, :, :].to(torch.float32)

    # 4. Profile Reduction and Normalization
    # Average across the spatial dimensions (Depth and Width) to get a 1D profile
    height_profile_sum = density_sum_3d.sum(dim=(0, 1, 3, 4))
    normalization_factor = window_height_voxels * current_depth * current_width
    height_density_profile = height_profile_sum / (normalization_factor + 1e-9)

    # 5. Optional Visualization Generation
    visualization_grid = None
    if enable_visualization:
        # Find index of max density
        max_density_center_index = torch.argmax(height_density_profile).item()

        # Build 1D mask along height axis
        height_mask_1d = torch.zeros(
            current_height, dtype=torch.uint8, device=target_device
        )

        # Calculate window span
        half_window = window_height_voxels // 2
        window_start = max(0, max_density_center_index - half_window)
        window_end = min(current_height, max_density_center_index + half_window + 1)

        # Apply highlight values
        height_mask_1d[window_start:window_end] = window_highlight_value
        height_mask_1d[max_density_center_index] = center_slice_value

        # Project 1D mask back to 3D and restore original (Depth, Height, Width) orientation
        visualization_grid = height_mask_1d.view(current_height, 1, 1).expand(
            -1, current_depth, current_width
        )
        visualization_grid = visualization_grid.permute(1, 0, 2)
        del height_mask_1d

    # 6. Final Cleanup and Output Formatting
    # Delete massive intermediate volumes before final hardware cleanup
    if target_device.type != "cpu":
        del working_volume, working_tensor, density_sum_3d
        clean_hardware_resources()

    output_profile = (
        height_density_profile.cpu().numpy()
        if return_as_numpy
        else height_density_profile
    )

    if enable_visualization:
        output_viz = (
            visualization_grid.cpu().numpy() if return_as_numpy else visualization_grid
        )
        return output_profile, output_viz

    return output_profile


@torch.inference_mode()
def calculate_height_density_profile(
    voxel_grid: np.ndarray | torch.Tensor,
    window_height_voxels: int,
    use_tiling: bool = True,
    spatial_tile_size: tuple[int, int] = (420, 400),
    return_as_numpy: bool = False,
    use_bfloat16: bool = True,
    enable_visualization: bool = False,
    window_highlight_value: int = 10,
    center_slice_value: int = 6,
) -> (
    torch.Tensor
    | np.ndarray
    | tuple[torch.Tensor | np.ndarray, torch.Tensor | np.ndarray]
):
    """Calculates density profiles along axis 1 (Height) with CPU-bound output.

    Args:
        voxel_grid: 3D input grid of shape (Depth, Height, Width).
        window_height_voxels: Vertical sliding window size.
        use_tiling: If True, processes in spatial patches to save VRAM.
        spatial_tile_size: (Depth_Tile, Width_Tile) used if use_tiling is True.
        return_as_numpy: Returns results as NumPy arrays if True.
        use_bfloat16: Uses BF16 for math (recommended for RTX 3090+).
        enable_visualization: Returns a uint8 grid highlighting the max density region.
        window_highlight_value: uint8 value for the max density window.
        center_slice_value: uint8 value for the center slice of that window.

    Returns:
        A 1D density profile (CPU), or a tuple of (density profile, 3D visualization grid) (CPU).
    """
    input_tensor = torch.as_tensor(voxel_grid, dtype=torch.float32)
    target_device = input_tensor.device
    depth_full, height_full, width_full = input_tensor.shape
    computation_dtype = torch.bfloat16 if use_bfloat16 else torch.float16

    # Initialize global sum for 1D profile on target device
    global_profile_sum = torch.zeros(
        height_full, device=target_device, dtype=torch.float32
    )

    if target_device.type != "cpu":
        clean_hardware_resources()

    # 1. Processing Logic (Tiled vs. Full Volume)
    if not use_tiling:
        # Full volume calculation (Optimized for RTX 3090/24GB)
        working_tensor = input_tensor.permute(1, 0, 2)
        working_vol = working_tensor.unsqueeze(0).unsqueeze(0).to(computation_dtype)

        with torch.amp.autocast(
            device_type=target_device.type, dtype=computation_dtype
        ):
            vertical_kernel = torch.ones(
                (1, 1, window_height_voxels, 1, 1),
                device=target_device,
                dtype=computation_dtype,
            )
            density_sum_3d = F.conv3d(
                working_vol, vertical_kernel, padding=(window_height_voxels // 2, 0, 0)
            )

        density_sum_3d = density_sum_3d[:, :, :height_full, :, :].to(torch.float32)
        global_profile_sum = density_sum_3d.sum(dim=(0, 1, 3, 4))

        del working_tensor, working_vol, vertical_kernel, density_sum_3d
    else:
        # Tiled calculation (Optimized for 8GB VRAM cards)
        tile_depth_step, tile_width_step = spatial_tile_size
        for d_start in range(0, depth_full, tile_depth_step):
            d_end = min(d_start + tile_depth_step, depth_full)
            for w_start in range(0, width_full, tile_width_step):
                w_end = min(w_start + tile_width_step, width_full)

                tile_tensor = input_tensor[d_start:d_end, :, w_start:w_end]
                working_tile = tile_tensor.permute(1, 0, 2)
                current_h, current_td, current_tw = working_tile.shape
                working_vol = (
                    working_tile.unsqueeze(0).unsqueeze(0).to(computation_dtype)
                )

                with torch.amp.autocast(
                    device_type=target_device.type, dtype=computation_dtype
                ):
                    vertical_kernel = torch.ones(
                        (1, 1, window_height_voxels, 1, 1),
                        device=target_device,
                        dtype=computation_dtype,
                    )
                    tile_sum_3d = F.conv3d(
                        working_vol,
                        vertical_kernel,
                        padding=(window_height_voxels // 2, 0, 0),
                    )

                tile_sum_3d = tile_sum_3d[:, :, :current_h, :, :].to(torch.float32)
                global_profile_sum += tile_sum_3d.sum(dim=(0, 1, 3, 4))

                del tile_tensor, working_tile, working_vol, vertical_kernel, tile_sum_3d
                if target_device.type != "cpu":
                    clean_hardware_resources()

    # 2. Global Normalization
    normalization_factor = window_height_voxels * depth_full * width_full
    height_density_profile_gpu = global_profile_sum / (normalization_factor + 1e-9)

    # Move primary output to CPU
    height_density_profile = height_density_profile_gpu.cpu()
    del height_density_profile_gpu

    # 3. Optional Visualization Generation (Reconstructed on GPU then moved to CPU)
    visualization_grid = None
    if enable_visualization:
        # Calculate max density index (still on CPU-friendly scalar)
        max_idx = torch.argmax(height_density_profile).item()

        # Build 1D mask on device for fast expansion
        height_mask_1d = torch.zeros(
            height_full, dtype=torch.uint8, device=target_device
        )
        half_win = window_height_voxels // 2
        w_start, w_end = (
            max(0, max_idx - half_win),
            min(height_full, max_idx + half_win + 1),
        )

        height_mask_1d[w_start:w_end] = window_highlight_value
        height_mask_1d[max_idx] = center_slice_value

        # Expand and permute to (Depth, Height, Width)
        visualization_grid_gpu = height_mask_1d.view(1, height_full, 1).expand(
            depth_full, -1, width_full
        )

        # Move to CPU and release GPU handle
        visualization_grid = visualization_grid_gpu.cpu()

        del height_mask_1d, visualization_grid_gpu

    # 4. Final Cleanup of Large Tensors
    if target_device.type != "cpu":
        del global_profile_sum
        clean_hardware_resources()

    # 5. Return Formatting
    output_profile = (
        height_density_profile.numpy() if return_as_numpy else height_density_profile
    )

    if enable_visualization:
        output_viz = (
            visualization_grid.numpy() if return_as_numpy else visualization_grid
        )
        return output_profile, output_viz

    return output_profile


def apply_morphology(
    volume: torch.Tensor | np.ndarray,
    kernel: torch.Tensor | np.ndarray,
    batch_dimension: Literal["depth", "height", "width"] = "height",
    morphology_type: Literal[
        "dilation", "erosion", "opening", "closing", "gradient", "top_hat", "bottom_hat"
    ] = "dilation",
    precision: Literal["float32", "float16", "bfloat16"] = "bfloat16",
    inference: bool = True,
    return_numpy: bool = True,
    keep_on_gpu: bool = False,
    compute_device: torch.device | None = None,
) -> torch.Tensor | np.ndarray:
    """Processes a 3D volume using Kornia morphological operations.

    Args:
        volume: The 3D input volume (Depth, Height, Width).
        kernel: 2D structuring element for the operation.
        batch_dimension: The axis to treat as the batch dimension.
        morphology_type: The specific morphological operation to perform.
        precision: The numerical precision for computation. Defaults to "bfloat16".
        inference: If True, disables gradient tracking via inference_mode.
        return_numpy: If True, returns a NumPy array; otherwise returns a Tensor.
        keep_on_gpu: If True, prevents moving the output tensor to CPU.
        compute_device: Hardware target for computation (e.g., "cuda", "cpu").

    Returns:
        The processed 3D volume as a torch.Tensor or np.ndarray.
    """
    if compute_device is None:
        compute_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Map precision strings to torch dtypes
    PRECISION_MAP = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    target_dtype = PRECISION_MAP[precision]

    DIMENSION_ORDER = {"depth": (0, 1, 2), "height": (1, 0, 2), "width": (2, 0, 1)}
    REVERSE_ORDER = {"depth": (0, 1, 2), "height": (1, 0, 2), "width": (1, 2, 0)}
    OPERATIONS = {
        "dilation": morph.dilation,
        "erosion": morph.erosion,
        "opening": morph.opening,
        "closing": morph.closing,
        "gradient": morph.gradient,
        "top_hat": morph.top_hat,
        "bottom_hat": morph.bottom_hat,
    }

    with torch.inference_mode(mode=inference):
        input_tensor = torch.as_tensor(volume, device=compute_device)
        kernel_tensor = torch.as_tensor(kernel, device=compute_device)

        # Apply mixed-precision via autocast
        # Note: autocast is disabled for float32 to avoid conversion overhead
        use_autocast = precision != "float32"
        with torch.autocast(device_type=compute_device.type, dtype=target_dtype, enabled=use_autocast):
            permute_order = DIMENSION_ORDER[batch_dimension]
            batched_view = input_tensor.permute(*permute_order).unsqueeze(1)

            processed_view = OPERATIONS[morphology_type](batched_view, kernel_tensor)

            inverse_order = REVERSE_ORDER[batch_dimension]
            restored_volume = processed_view.squeeze(1).permute(*inverse_order)

        if return_numpy:
            output_data = restored_volume.detach().cpu().numpy()
        else:
            output_data = restored_volume if keep_on_gpu else restored_volume.cpu()

    # Explicit memory cleanup
    del input_tensor, kernel_tensor, batched_view, processed_view, restored_volume
    gc.collect()
    if compute_device.type == "cuda":
        torch.cuda.empty_cache()

    return output_data


def apply_patch_morphology(
    volume: torch.Tensor | np.ndarray,
    kernel: torch.Tensor | np.ndarray,
    patch_size: tuple[int, int, int] = (64, -1, 64),
    morphology_type: Literal["dilation", "erosion", "opening", "closing"] = "dilation",
    precision: Literal["float32", "float16", "bfloat16"] = "bfloat16",
    inference: bool = True,
    compute_device: torch.device | None = None
) -> np.ndarray:
    """Processes 3D volume using patches with mixed-precision support.

    Args:
        volume: Input 3D volume (Depth, Height, Width).
        kernel: 2D structuring element.
        patch_size: 3D patch dimensions (-1 for full axis).
        morphology_type: Kornia morphological operation.
        precision: Numerical precision for GPU operations. Defaults to 'bfloat16'.
        inference: If True, uses torch.inference_mode. Defaults to True.
        compute_device: Hardware target. Defaults to auto-select.

    Returns:
        np.ndarray: Reconstructed 3D volume on CPU.
    """
    if compute_device is None:
        compute_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Mapping precision string to torch dtypes
    DTYPE_MAP = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16
    }
    target_dtype = DTYPE_MAP[precision]

    # Resolve -1 indicators to full dimension sizes
    input_tensor = torch.as_tensor(volume, dtype=torch.float32)
    kernel_tensor = torch.as_tensor(kernel, device=compute_device, dtype=target_dtype)
    D, H, W = input_tensor.shape
    
    resolved_patches = [
        (patch_size[i] if patch_size[i] != -1 else input_tensor.shape[i])
        for i in range(3)
    ]
    pD, pH, pW = resolved_patches
    halo = [k // 2 for k in kernel_tensor.shape]
    hD, hW = halo, halo
    
    output = torch.zeros_like(input_tensor)
    operation = getattr(morph, morphology_type)

    # Use inference_mode and autocast for performance
    with torch.inference_mode(mode=inference):
        for d in range(0, D, pD):
            for h in range(0, H, pH):
                for w in range(0, W, pW):
                    
                    # Define extraction bounds with context halo
                    d_start, d_end = max(0, d - hD), min(D, d + pD + hD)
                    h_start, h_end = h, min(H, h + pH)
                    w_start, w_end = max(0, w - hW), min(W, w + pW + hW)

                    # Move patch and cast to target precision
                    patch = input_tensor[d_start:d_end, h_start:h_end, w_start:w_end]
                    patch = patch.to(device=compute_device, dtype=target_dtype)
                    
                    # Autocast handles the mixed-precision context
                    with torch.amp.autocast(device_type=compute_device.type, dtype=target_dtype):
                        # (Batch=Height, Channel=1, Depth, Width)
                        patch_batched = patch.permute(1, 0, 2).unsqueeze(1)
                        processed = operation(patch_batched, kernel_tensor)
                
                    # Reconstruct: Restore shape and remove halo
                    processed = processed.squeeze(1).permute(1, 0, 2)
                    crop_d = hD if d > 0 else 0
                    crop_w = hW if w > 0 else 0
                    
                    actual_d = min(pD, D - d)
                    actual_h = min(pH, H - h)
                    actual_w = min(pW, W - w)
                    
                    # Copy result back to host in float32
                    output[d:d+actual_d, h:h+actual_h, w:w+actual_w] = \
                        processed[crop_d:crop_d+actual_d, :, crop_w:crop_w+actual_w].cpu().float()

                    del patch, patch_batched, processed
                    if compute_device.type == "cuda":
                        torch.cuda.empty_cache()

    return output.numpy()