"""
Functional (Qt-free) module for bidirectional A-scan registration.

Contains the polynomial unwarp model, sharpness scoring functions, and the
coefficient search (``auto_find_coeffs``) used by
``Bidirectional_Ascan_Registration_Widget.autoFindCoeffs``. Kept independent
of Qt so it can be unit tested and reused outside of the widget.
"""

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import napari_cool_tools_io


def blur_score_vol_torch_frequency(img: torch.Tensor) -> torch.Tensor:
    """
    Edge magnitude via frequency-domain derivatives.
    img: (H, W) real tensor (any float dtype); returns a scalar score (sum of magnitudes).
    """
    assert img.ndim == 2, "Input must be 2D (H, W)."

    H, W = img.shape
    device, dtype = img.device, img.dtype

    # 1) FFT (no fftshift)
    Fimg = torch.fft.fft2(img)

    # 2) Frequency coordinates shaped (H, W)
    u = torch.fft.fftfreq(W, d=1.0, device=device, dtype=dtype)   # (W,)
    v = torch.fft.fftfreq(H, d=1.0, device=device, dtype=dtype)   # (H,)
    V, U = torch.meshgrid(v, u, indexing='ij')                    # both (H, W)

    # 3) Derivative filters: j*2*pi*f
    Hx = torch.complex(torch.zeros_like(U), 2 * torch.pi * U)     # (H, W) complex
    Hy = torch.complex(torch.zeros_like(V), 2 * torch.pi * V)

    # 4) Apply filters in frequency
    Fx = Fimg * Hx
    Fy = Fimg * Hy

    # 5) Inverse FFT to spatial gradients
    edge_x = torch.fft.ifft2(Fx).real
    edge_y = torch.fft.ifft2(Fy).real

    # 6) Gradient magnitude + a simple score
    edge_mag = torch.hypot(edge_x, edge_y)                        # sqrt(x^2 + y^2)
    score = edge_mag.abs().sum()                                  # torch scalar

    return score


def blur_score_vol_torch_spatial(x: torch.Tensor) -> torch.Tensor:
    """
    Sum of absolute Laplacian (higher => sharper).
    Returns a *torch scalar* (zero-dim tensor) on the same device.

    x: (H,W) or (D,H,W). For 3D, applies 2D Laplacian per slice.
    """
    # Shape to (N,1,H,W) for conv2d
    x4 = x.unsqueeze(0).unsqueeze(0) if x.ndim == 2 else x.unsqueeze(1)

    # 3x3 Laplacian kernel
    h = torch.tensor([[0., 1., 0.],
                      [1., -4., 1.],
                      [0., 1., 0.]], dtype=x.dtype, device=x.device).view(1, 1, 3, 3)

    # replicate-pad edges (nearest) then conv
    xpad = F.pad(x4, (1, 1, 1, 1), mode='replicate')
    L = F.conv2d(xpad, h)  # (N,1,H,W)

    # Sum of absolute Laplacian -> torch scalar
    score = L.abs().sum()
    return score


def unwarp_polynomial_offset_torch(
    frameData: torch.Tensor,               # (H,W) torch.Tensor
    coeffs: torch.Tensor,                  # 1D tensor [c0,c1,c2,c3]
    scales: torch.Tensor,                  # 1D tensor [sc0,sc1,sc2,sc3]
    mode: str = "bilinear",                # interpolation mode for grid_sample ("bilinear" or "nearest")
) -> torch.Tensor:
    """
    Shifts rows (depth axis) by a constant offset derived from c0, via grid_sample.
    """
    img = frameData
    dtype, device = img.dtype, img.device

    H, W = img.shape
    offset = 2 * scales[0] * coeffs[0] / H

    # ---- build sampling grid for grid_sample ----
    # grid_sample expects normalized coords in [-1, 1]
    # x: width axis, y: height axis
    x_norm = torch.linspace(-1.0, 1.0, W, device=device, dtype=dtype)
    y_norm = torch.linspace(-1.0, 1.0, H, device=device, dtype=dtype) + offset  # over depth

    # make (H, W) grid: rows repeat y, columns repeat x
    grid_x = x_norm.view(1, W).expand(H, W)
    grid_y = y_norm.view(H, 1).expand(H, W)
    grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)      # (1, H, W, 2)

    # ---- sample ----
    img_bchw = img.view(1, 1, H, W)
    out = F.grid_sample(
        img_bchw, grid,
        mode=mode,
        padding_mode="border",
        align_corners=True
    )
    result = out[0, 0]  # (H, W)
    return result


def unwarp_polynomial_linear_torch(
    frameData: torch.Tensor,               # (H,W) torch.Tensor
    coeffs: torch.Tensor,                  # 1D tensor [c0,c1,c2,c3]
    scales: torch.Tensor,                  # 1D tensor [sc0,sc1,sc2,sc3]
    mode: str = "bilinear",                # interpolation mode for grid_sample ("bilinear" or "nearest")
) -> torch.Tensor:
    """
    Applies a linear (depth) scaling derived from c1, via grid_sample.
    """
    img = frameData
    dtype, device = img.dtype, img.device

    H, W = img.shape

    linear_scale = (H + (coeffs[1] * scales[1])) / H

    y_input = torch.linspace(0.0, 1.0, H, device=device, dtype=dtype)
    y_warp = linear_scale * y_input
    y_warp = 2.0 * y_warp - 1.0

    # ---- build sampling grid for grid_sample ----
    y_norm = y_warp        # shape (H,)
    x_norm = 2.0 * torch.arange(W, device=device, dtype=dtype) / (W - 1) - 1.0  # shape (W,)

    # make (H, W) grid: rows repeat y, columns repeat x
    grid_x = x_norm.view(1, W).expand(H, W)
    grid_y = y_norm.view(H, 1).expand(H, W)
    grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)      # (1, H, W, 2)

    # ---- sample ----
    img_bchw = img.view(1, 1, H, W)
    out = F.grid_sample(
        img_bchw, grid,
        mode=mode,
        padding_mode="border",
        align_corners=True
    )
    result = out[0, 0]  # (H, W)

    return result


def unwarp_polynomial_unified_torch(
    frameData: torch.Tensor,               # (H,W) torch.Tensor
    coeffs: torch.Tensor,                  # 1D tensor [c0,c1,c2,c3]
    scales: torch.Tensor,                  # 1D tensor [sc0,sc1,sc2,sc3]
    mode: str = "bilinear",                # interpolation mode for grid_sample ("bilinear" or "nearest")
) -> torch.Tensor:
    """
    Applies the nonlinear (quadratic + cubic, c2 and c3) portion of the warp via grid_sample.
    Warps rows (y); columns (x) are identity.
    """
    img = frameData
    dtype, device = img.dtype, img.device

    H, W = img.shape

    y_input = torch.linspace(0.0, 1.0, H, device=device, dtype=dtype)
    # polynomial
    y_warp = (y_input
              + (coeffs[2] * scales[2]) * y_input * y_input.abs()
              + (coeffs[3] * scales[3]) * (y_input ** 3))

    # normalize back to [0..1] and map to pixel index [0..H-1]
    denom = (y_warp.max() - y_warp.min()).clamp_min(1e-12)
    y_warp_norm = (y_warp - y_warp.min()) / denom

    # ---- build sampling grid for grid_sample ----
    y_norm = 2.0 * y_warp_norm - 1.0          # shape (H,)
    x_norm = 2.0 * torch.arange(W, device=device, dtype=dtype) / (W - 1) - 1.0  # shape (W,)

    # make (H, W) grid: rows repeat y, columns repeat x
    grid_x = x_norm.view(1, W).expand(H, W)
    grid_y = y_norm.view(H, 1).expand(H, W)
    grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)      # (1, H, W, 2)

    # ---- sample ----
    img_bchw = img.view(1, 1, H, W)
    out = F.grid_sample(
        img_bchw, grid,
        mode=mode,
        padding_mode="border",
        align_corners=True
    )
    result = out[0, 0]  # (H, W)
    return result


def process_image_no_plot_torch(
    image: torch.Tensor,
    coeffs: torch.Tensor,
    scales: torch.Tensor,
    AA: int,
    BB: int,
    dual_edge: bool,
    mode: str = "bilinear",
) -> torch.Tensor:
    """
    Apply the offset/linear/unified polynomial unwarp to the AA-interleaved
    columns of ``image`` (and, if ``dual_edge``, to the BB-interleaved
    columns using the negated coefficients). ``image`` is returned unchanged
    when ``enable`` is False.
    """
    new_image_torch = image.clone()

    new_image_1 = new_image_torch[:, AA::2]
    new_image_1 = unwarp_polynomial_offset_torch(new_image_1, coeffs, scales, mode=mode)
    new_image_1 = unwarp_polynomial_linear_torch(new_image_1, coeffs, scales, mode=mode)
    new_image_1 = unwarp_polynomial_unified_torch(new_image_1, coeffs, scales, mode=mode)
    new_image_torch[:, AA::2] = new_image_1

    if dual_edge:
        new_image_2 = new_image_torch[:, BB::2]
        new_image_2 = unwarp_polynomial_offset_torch(new_image_2, -1.0 * coeffs, scales, mode=mode)
        new_image_2 = unwarp_polynomial_linear_torch(new_image_2, -1.0 * coeffs, scales, mode=mode)
        new_image_2 = unwarp_polynomial_unified_torch(new_image_2, -1.0 * coeffs, scales, mode=mode)
        new_image_torch[:, BB::2] = new_image_2

    return new_image_torch


def auto_bidirectional_ascan_registration(
    volume: np.ndarray | torch.Tensor, # the 3D volume to be registered (num_frames, height, width)
    reference_frame_index: int = 0, # the index of the reference frame in the volume to be registered against
    bmscan: int = 1, # whether the volume is a B-M scan (True) or a B-scan (False)
    init_coeffs: list[float] = [0.0, 0.0, 0.0, 0.0], # initial coefficients for the polynomial unwarp must be size of 4
    ranges: int = 20,
    step_size: float = 1.0,
    search_c0: bool = False,
    search_c1: bool = False,
    search_c2: bool = False,
    search_c3: bool = False,
    mode: str = "bilinear",
    dual_edge: bool = False,
    inverse: bool = False, #inverse the coefficient directions
    flipAB: bool = False, #flip the AA and BB interleave indices
    double_side: bool = False, #whether the volume is double-sided (alternating) B-M scan acquisition
    frequency_domain: bool = True
)-> np.ndarray:

    #volume must be a 3D torch tensor of shape (num_frames, height, width) and the first dim length must be at least 2. The function will register the first two frames of the volume against each other and return the best coefficients and the best score.

    # This is a standalone, Qt-free helper. The widget layer should pass in the
    # relevant values (volume, index, flags, etc.) instead of accessing self.*.
    if volume is None:
        raise ValueError("volume must not be None")
    if not isinstance(volume, (torch.Tensor, np.ndarray)):
        raise TypeError("volume must be a torch.Tensor or np.ndarray")
    if volume.ndim != 3:
        raise ValueError(f"volume must be 3D (num_frames, height, width), got shape {tuple(volume.shape)}")
    if volume.shape[0] < 2:
        raise ValueError("volume must contain at least 2 frames")

    if not 0 <= reference_frame_index < (volume.shape[0] - 1):
        raise IndexError(
            f"reference_frame_index={reference_frame_index} out of bounds for volume of length {volume.shape[0]} - 1"
        )

    idx1 = reference_frame_index
    idx2 = idx1 + 1

    bscan1 = volume[idx1, :, :]
    bscan2 = volume[idx2, :, :]

    if inverse:
        bscan1 = bscan1[::-1, :]
        bscan2 = bscan2[::-1, :]

    if isinstance(volume, np.ndarray):
        image1 = torch.from_numpy(bscan1.copy()).to(device=napari_cool_tools_io.device)
        image2 = torch.from_numpy(bscan2.copy()).to(device=napari_cool_tools_io.device)
    else:
        image1 = bscan1.clone().detach().to(device=napari_cool_tools_io.device)
        image2 = bscan2.clone().detach().to(device=napari_cool_tools_io.device)

    dtype = image1.dtype
    npdtype = np.dtype(image1.cpu().numpy().dtype)
    device = image1.device

    #if all search flags are False, raise an error
    if not (search_c0 or search_c1 or search_c2 or search_c3):
        raise ValueError("At least one of search_c0, search_c1, search_c2, or search_c3 must be True")

    c0_range = [0.0]
    c1_range = [0.0]
    c2_range = [0.0]
    c3_range = [0.0]

    # search grid for each coefficient. Any coefficient not marked for search stays
    # fixed at its initial value.
    if search_c0:
        c0_range = np.arange(-ranges, ranges, 1, dtype=np.float32) + init_coeffs[0]

    if search_c1:
        c1_range = np.arange(-ranges, ranges, 1, dtype=np.float32) + init_coeffs[1]

    if search_c2:
        c2_range = np.arange(-ranges, ranges, 1, dtype=np.float32) + init_coeffs[2]

    if search_c3:
        c3_range = np.arange(-ranges, ranges, 1, dtype=np.float32) + init_coeffs[3]

    total_iterations = len(c0_range) * len(c1_range) * len(c2_range) * len(c3_range)

    best_coeffs = torch.as_tensor(init_coeffs, dtype=dtype, device=device)

    #if step_size length is singular
    if isinstance(step_size, list) or isinstance(step_size, tuple) or isinstance(step_size, np.ndarray):
        if len(step_size) != 4:
            raise ValueError(f"step_size must be a singular float or a list, tuple, or numpy array of length 4, got length {len(step_size)}")
        
        scales = torch.as_tensor(step_size, dtype=dtype, device=device)
    else:
        scales = torch.as_tensor([step_size, step_size, step_size, step_size], dtype=dtype, device=device)

    AA, BB = (0, 1)

    if flipAB:
        AA, BB = np.flip((AA, BB))

    AA1, BB1 = (AA, BB)
    AA2, BB2 = (AA, BB)

    if double_side:#this means the AB is alternating
        #if the frame is odd, flip (if they are in the same group, they may be both flipped)
        cframe1 = int(np.floor(idx1/bmscan)) #this will handle bmscan
        if cframe1 % 2:
            AA1, BB1 = np.flip((AA, BB))
        cframe2 = int(np.floor(idx2/bmscan)) #this will handle bmscan
        if cframe2 % 2:
            AA2, BB2 = np.flip((AA, BB))


    blur_score_vol = blur_score_vol_torch_frequency if frequency_domain else blur_score_vol_torch_spatial

    new_image_torch1 = process_image_no_plot_torch(image1, best_coeffs, scales, AA=AA1, BB=BB1, dual_edge=dual_edge, mode=mode)
    new_image_torch2 = process_image_no_plot_torch(image2, best_coeffs, scales, AA=AA2, BB=BB2, dual_edge=dual_edge, mode=mode)

    best_score = blur_score_vol(new_image_torch1).item() + blur_score_vol(new_image_torch2).item()

    iteration = 0
    with tqdm(total=total_iterations, desc="Searching coeffs") as pbar:
        pbar.set_postfix(best_score=best_score,coeffs=best_coeffs)

        for c0 in c0_range:
            for c1 in c1_range:
                for c2 in c2_range:
                    for c3 in c3_range:
                        iteration += 1
                        coeffs = torch.as_tensor([c0, c1, c2, c3], dtype=dtype, device=device)

                        new_image_torch1 = process_image_no_plot_torch(image1,coeffs,scales,AA=AA1,BB=BB1,dual_edge=dual_edge,mode=mode)
                        new_image_torch2 = process_image_no_plot_torch(image2,coeffs,scales,AA=AA2,BB=BB2,dual_edge=dual_edge,mode=mode)

                        score = blur_score_vol(new_image_torch1).item() + blur_score_vol(new_image_torch2).item()
                        if score < best_score:
                            best_score = score
                            best_coeffs = torch.as_tensor([c0, c1, c2, c3], dtype=dtype, device=device)
                            pbar.set_postfix(
                                best_score=best_score,
                                coeffs=best_coeffs.cpu().numpy()
                            )

                        pbar.update(1)
    
    print(f"Best coeffs found: {best_coeffs} with score: {best_score}")

    #apply this coefficient to the entire volume and return the registered volume
    if isinstance(volume, np.ndarray):
        save_volume = np.zeros(volume.shape, dtype=volume.dtype)
    elif isinstance(volume, torch.Tensor):
        save_volume = np.zeros(volume.shape, dtype=npdtype)

    for current_idx, bscan in enumerate(volume):

        if inverse:
            bscan = bscan[::-1,:] #(2048,800)

        # Default
        AA, BB = (0, 1)

        if flipAB:
            AA, BB = np.flip((AA, BB))

        split = 2

        if double_side:#this means the AB is alternating
            cframe = int(np.floor(current_idx/bmscan)) #this will handle bmscan
            if cframe % 2:
                AA, BB = np.flip((AA, BB))

        # if self.splitModeComboBox.currentIndex() == 1:#TODO fix for split mode
        #     AA, BB = 2*AA, 2*BB
        #     split = 4
        if isinstance(bscan, np.ndarray):
            new_image_torch = torch.from_numpy(bscan.copy()).to(device=device, dtype=dtype)
        else:
            new_image_torch = bscan.clone().detach().to(device=device, dtype=dtype)

        new_image_1 = new_image_torch[:,AA::split]
        new_image_1 = unwarp_polynomial_offset_torch(new_image_1, best_coeffs, scales, mode=mode)
        new_image_1 = unwarp_polynomial_linear_torch(new_image_1, best_coeffs, scales, mode=mode)
        new_image_1 = unwarp_polynomial_unified_torch(new_image_1, best_coeffs, scales, mode=mode)
        new_image_torch[:,AA::split] = new_image_1

        # if self.splitModeComboBox.currentIndex() == 1:
        #     new_image_1 = new_image_torch[:,AA+1::split]
        #     new_image_1 = unwarp_polynomial_offset_torch(new_image_1, best_coeffs, scales, mode=mode)
        #     new_image_1 = unwarp_polynomial_linear_torch(new_image_1, best_coeffs, scales, mode=mode)
        #     new_image_1 = unwarp_polynomial_unified_torch(new_image_1, best_coeffs, scales, mode=mode)
        #     new_image_torch[:,AA+1::split] = new_image_1

        if dual_edge:
            new_image_2 = new_image_torch[:,BB::split]
            new_image_2 = unwarp_polynomial_offset_torch(new_image_2, -1.0*best_coeffs, scales, mode=mode)
            new_image_2 = unwarp_polynomial_linear_torch(new_image_2, -1.0*best_coeffs, scales,mode=mode)
            new_image_2 = unwarp_polynomial_unified_torch(new_image_2, -1.0*best_coeffs,scales, mode=mode)
            new_image_torch[:,BB::split] = new_image_2

            # if self.splitModeComboBox.currentIndex() == 1:
            #     new_image_2 = new_image_torch[:,BB+1::split]
            #     new_image_2 = unwarp_polynomial_offset_torch(new_image_2, -1.0*best_coeffs, scales, mode=mode)
            #     new_image_2 = unwarp_polynomial_linear_torch(new_image_2, -1.0*best_coeffs, scales,mode=mode)
            #     new_image_2 = unwarp_polynomial_unified_torch(new_image_2, -1.0*best_coeffs,scales, mode=mode)
            #     new_image_torch[:,BB+1::split] = new_image_2

        new_image = new_image_torch.cpu().numpy()

        if inverse:
            new_image = new_image[::-1,:]

        save_volume[current_idx] = new_image
    
    return save_volume
    

