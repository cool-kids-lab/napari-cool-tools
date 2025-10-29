import torch
import math
import torch.nn.functional as F
from napari_cool_tools_oct_preproc import Operation

def desine(frame: torch.Tensor, mode = "bilinear", transpose: bool = True) -> torch.Tensor:
    def desine_torch_2D(frame: torch.Tensor, mode = "bilinear", transpose: bool = True) -> torch.Tensor:
        """
        Regrid a 2D array from sine-spaced samples to uniform (linspace) along the chosen axis.
        Uses F.grid_sample with an analytically derived inverse mapping.
        
        Args:
            frame: 2D tensor [H, W], sine-sampled along axis (0 or 1).
            axis:  0 -> rows are sine-sampled; 1 -> columns are sine-sampled.
            device: 'cpu' or 'cuda'
        Returns:
            2D tensor [H, W] on the same device as requested.
        """
        # If sine sampling is along columns, we work with [H, W].
        # If along rows, transpose to treat rows as columns, then transpose back.
        x = frame
        if transpose:
            x = x.t()  # work on columns

        H, W = x.shape  # now sine-sampling is along the last dimension (width)

        # Prepare input for grid_sample: [N=1, C=1, H, W]
        x = x.unsqueeze(0).unsqueeze(0)

        # --- Build inverse mapping from uniform output index j -> sine-sampled source index n_src ---
        # grid_sample with align_corners=True interprets indices in [0..W-1] mapped to [-1..1] by:  g = 2*i/(W-1)-1
        Wm1 = float(W - 1)
        j = torch.linspace(0.0, Wm1, W, device=frame.device)                  # uniform target coords
        # inverse of: y_org = (Wm1/2) * sin(theta) + (Wm1/2), with theta = (pi/Wm1)*n - pi/2
        # Solve for n given y=j:
        arg = (j - Wm1 * 0.5) / (Wm1 * 0.5)                             # in [-1, 1]
        arg = torch.clamp(arg, -1.0, 1.0)                               # numeric safety
        theta = torch.arcsin(arg)                                       # [-pi/2, pi/2]
        n_src = (theta + math.pi * 0.5) * (Wm1 / math.pi)               # source index in [0..W-1]
        grid_x = (2.0 * n_src / Wm1) - 1.0                              # normalize to [-1, 1]

        # Tile across rows; y stays linear (identity)
        grid_x = grid_x.unsqueeze(0).repeat(H, 1)                       # [H, W]
        grid_y = torch.linspace(-1.0, 1.0, H, device=frame.device).unsqueeze(1).repeat(1, W)  # [H, W]
        grid = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0)       # [1, H, W, 2]

        # Sample (bilinear = linear along each axis; zeros outside)
        y = F.grid_sample(
            x, grid, mode=mode, padding_mode="zeros", align_corners=True
        ).squeeze(0).squeeze(0)  # [H, W]

        if transpose:
            y = y.t()

        return y

    def desine_torch_3d(frame: torch.Tensor, mode = "bilinear", transpose: bool = True) -> torch.Tensor:
        """
        Regrid a 3D volume from sine-spaced samples to uniform (linspace)
        along the chosen axis (0, 1, or 2).
        Uses F.grid_sample for trilinear interpolation.

        Args:
            frame: 3D tensor [D, H, W] (float32 or float64)
            axis:  which dimension is sine-sampled (0=depth, 1=height, 2=width)
            device: 'cpu' or 'cuda'
        Returns:
            3D tensor [D, H, W] after resampling to uniform spacing.
        """
        # move to float32 for F.grid_sample
        x = frame

        # permute so sine-axis becomes last (W)
        if transpose:
            x = x.permute(0, 2, 1)  # H,W,D

        D, H, W = x.shape  # now W is sine-sampled axis

        # add batch/channel dims: [N,C,D,H,W]
        x = x.unsqueeze(0).unsqueeze(0)

        Wm1 = float(W - 1)
        j = torch.linspace(0.0, Wm1, W, device=frame.device)
        arg = (j - Wm1 * 0.5) / (Wm1 * 0.5)
        arg = torch.clamp(arg, -1.0, 1.0)
        theta = torch.arcsin(arg)
        n_src = (theta + math.pi * 0.5) * (Wm1 / math.pi)
        grid_x = (2.0 * n_src / Wm1) - 1.0  # normalized [-1,1]

        # create full 3D grid (Z,Y,X)
        grid_z = torch.linspace(-1.0, 1.0, D, device=frame.device)
        grid_y = torch.linspace(-1.0, 1.0, H, device=frame.device)
        grid_z, grid_y, grid_x = torch.meshgrid(grid_z, grid_y, grid_x, indexing="ij")
        grid = torch.stack((grid_x, grid_y, grid_z), dim=-1).unsqueeze(0)  # [1,D,H,W,3]

        # trilinear sampling
        y = F.grid_sample(
            x, grid, mode="bilinear", padding_mode="zeros", align_corners=True
        ).squeeze(0).squeeze(0)  # [D,H,W]

        # undo permutation
        if transpose:
            y = y.permute(0, 2, 1)  # back to [D,H,W]

        return y

    if frame.dim() == 2:
        return desine_torch_2D(frame, mode, transpose)
    elif frame.dim() == 3:
        return desine_torch_3d(frame, mode, transpose)

from napari_cool_tools_io import device
from napari_cool_tools_oct_preproc import OCTACalc 
import numpy as np

def generate_octa(
    img: np.ndarray,
    mscans: int = 1,
    calc: OCTACalc = OCTACalc.STD,
    ) -> np.ndarray:

    """Generate OCTA volume from structural OCT data."""
    """All operation are done using torch for speed."""
    """but the output is in numpy format for napari compatibility."""

    m_img = torch.tensor(img).to(device)

    new_shape = (-1, mscans, img.shape[-2], img.shape[-1])
    m_img = m_img.reshape(new_shape)
    
    if calc == OCTACalc.STD:
        out_data = m_img.std(dim=1)

    elif calc == OCTACalc.VAR:
        out_data = m_img.var(dim=1)

    elif calc == OCTACalc.VAR2:
        out_data = m_img.var(dim=1)
        out_data = out_data**2

    elif calc == OCTACalc.ADA :
        #amplitude decorrelation        
        out_data = torch.zeros((m_img.shape[0],m_img.shape[-2],m_img.shape[-1]), device=device)
        for idx,pair in enumerate(m_img):
            for ii in range(0,mscans-1):
                frameA = pair[ii]
                frameB = pair[ii+1]

                ada = 1 - (frameA * frameB) / (0.5*frameA**2 + 0.5*frameB**2)
                out_data[idx] = out_data[idx]+ada

            #average ada
            out_data[idx] = out_data[idx]/(mscans-1)

    out_data_numpy = out_data.cpu().numpy()

    if device.type == "cuda":
        torch.cuda.empty_cache()

    return out_data_numpy

def reshuffle_vista_frames(ref_RawData: np.ndarray, nvista: int, numMScans: int) -> np.ndarray:
    """
    Reorders frames in ref_RawData for 'vista' grouping, matching the given MATLAB logic.

    Parameters
    ----------
    ref_RawData : np.ndarray
        3D array of shape (H, W, numBScans).
    nvista : int
        Number of vista views.
    numMScans : int
        Number of M-scans.

    Returns
    -------
    np.ndarray
        Reordered array with the same shape as ref_RawData.
    """
    numBScans  = ref_RawData.shape[0]
    if nvista <= 1:
        return ref_RawData.copy()

    block = nvista * numMScans
    if numBScans % block != 0:
        raise ValueError(
            f"numBScans ({numBScans}) must be divisible by nvista*numMScans ({block})."
        )

    # --- Vectorized permutation (fast) ---
    # Within each block of size (nvista x numMScans), MATLAB orders linear indices column-wise.
    # The MATLAB code remaps to row-wise order. Build that mapping:
    # perm[t] = source_index_within_block for target position t (0-based).
    perm = np.arange(block).reshape(nvista, numMScans, order='F').ravel(order='C')

    out = np.empty_like(ref_RawData)
    for start in range(0, numBScans, block):
        out[start:start+block,:,:] = ref_RawData[start + perm, :, :]

    return out


# def reshuffle_vista_frames(ref_RawData: np.ndarray, nvista: int, numMScans: int) -> np.ndarray:
#     H, W, numBScans = ref_RawData.shape
#     if nvista <= 1:
#         return ref_RawData.copy()

#     temp_bcans_size = numBScans // (nvista * numMScans)
#     if temp_bcans_size * nvista * numMScans != numBScans:
#         raise ValueError("numBScans must be divisible by nvista*numMScans.")

#     new_data = np.zeros_like(ref_RawData)

#     # Build A_idx (MATLAB 1-based -> convert to 0-based)
#     A_idx = np.arange(1, nvista * numMScans + 1).reshape(nvista, numMScans, order='F')

#     for k in range(1, temp_bcans_size + 1):
#         for i in range(1, nvista + 1):
#             for j in range(1, numMScans + 1):
#                 target_idx = nvista * numMScans * (k - 1) + ((i - 1) * numMScans) + j
#                 ori_idx    = A_idx[i - 1, j - 1] + nvista * numMScans * (k - 1)

#                 # Convert to 0-based indices for Python
#                 new_data[:, :, target_idx - 1] = ref_RawData[:, :, ori_idx - 1]

#     return new_data
