from napari.layers import Image
import torch
import numpy as np
import math
import torch.nn.functional as F
from napari_cool_tools_oct_preproc import Operation
from napari_cool_tools_img_proc._normalization_funcs import normalize_data_in_range_func

def auto_contrast(
    img: Image,
    lower_percentile: float = 1.0,
    upper_percentile: float = 99.0,
    num_averages: int = 1,
):
    vol = img.data
    n = num_averages

    if n == 0:
        n = 1
    
    if vol.ndim == 3:
        center = vol.shape[0] // 2

        half = n // 2
        if n % 2 == 1:  # odd
            temp_frame = vol[center-half : center+half+1]
        else:           # even
            temp_frame = vol[center-half : center+half]

        temp_frame = np.mean(temp_frame, axis=0)
        vmin, vmax = np.percentile(temp_frame, (lower_percentile, upper_percentile))
        img.contrast_limits = (float(vmin), float(vmax))

    elif vol.ndim == 2:
        vmin, vmax = np.percentile(vol, (lower_percentile, upper_percentile))
        img.contrast_limits = (float(vmin), float(vmax))
    else:
        show_error("Input image must be 2D or 3D.")
        return


def desine(frame: torch.Tensor, mode = "bilinear", align_corners: bool = False, transpose: bool = True, scale_fac: int = 2) -> torch.Tensor:
    def desine_torch_2D(frame: torch.Tensor, mode = "bilinear", align_corners: bool = False, transpose: bool = True) -> torch.Tensor:
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
            x, grid, mode=mode, padding_mode="zeros", align_corners=align_corners
        ).squeeze(0).squeeze(0)  # [H, W]

        if transpose:
            y = y.t()

        return y

    def desine_torch_3d(frame: torch.Tensor, mode = "bilinear", align_corners: bool = False, transpose: bool = True, scale_fac: int = 1) -> torch.Tensor:
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
        interp_mode = "bilinear"
        if mode == "nearest":
            interp_mode = "nearest-exact"
        else:
            interp_mode = mode

        print(f"Grid smaple mode: {mode} with interpolation mode: {interp_mode}")
                    # permute so sine-axis becomes last (W)
        if transpose:
            frame = frame.permute(0, 2, 1)  # H,W,D [800,800,1024] -> [800,1024,800]

        y = torch.zeros_like(frame)

        _,H, W = y.shape  # now W is sine-sampled axis
        W = W * scale_fac

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

        for i, x in enumerate(frame):

            # add batch/channel dims: [N,C,H,W]
            x = x.unsqueeze(0).unsqueeze(0)

            #up sample image
            if interp_mode not in ("area","nearest","nearest-exact"):
                x = torch.nn.functional.interpolate(
                    x,
                    scale_factor=(1.0, scale_fac),  # explicit output size
                    mode=interp_mode,
                    # mode='bilinear',
                    align_corners=False
                )
            else:
                x = torch.nn.functional.interpolate(
                    x,
                    scale_factor=(1.0, scale_fac),  # explicit output size
                    mode=interp_mode,
                    # mode='bilinear',
                )

            # Sample (bilinear = linear along each axis; zeros outside)
            x = F.grid_sample(
                x, grid, mode=mode, padding_mode="zeros", align_corners=align_corners
            )

            #down sample image
            if interp_mode not in ("area","nearest","nearest-exact"):
                x = torch.nn.functional.interpolate(
                    x,
                    scale_factor=(1.0, 1.0/scale_fac),  # explicit output size
                    mode=interp_mode,
                    # mode='bilinear',
                    align_corners=False
                )
            else:
                    x = torch.nn.functional.interpolate(
                    x,
                    scale_factor=(1.0, 1.0/scale_fac),  # explicit output size
                    mode=interp_mode,
                    # mode='bilinear',
                )

            y[i] = x.squeeze(0).squeeze(0)  # [D,H,W]


        # undo permutation
        if transpose:
            y = y.permute(0, 2, 1)  # back to [D,H,W]

        return y

    if frame.dim() == 2:
        return desine_torch_2D(frame, mode=mode, align_corners=align_corners, transpose=transpose)
    elif frame.dim() == 3:
        return desine_torch_3d(frame, mode=mode, align_corners=align_corners,transpose=transpose, scale_fac=scale_fac)

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

    elif calc == OCTACalc.ADAVAR2 :
        out_data = torch.zeros((m_img.shape[0],m_img.shape[-2],m_img.shape[-1]), device=device)
        for idx,pair in enumerate(m_img):
            for ii in range(0,mscans-1):
                frameA = pair[ii]
                frameB = pair[ii+1]

                ada = 1 - (frameA * frameB) / (0.5*frameA**2 + 0.5*frameB**2)
                out_data[idx] = out_data[idx]+ada

            #average ada
            out_data[idx] = out_data[idx]/(mscans-1)
            out_data[idx] = out_data[idx] * (m_img[idx].var(dim=0)**2)

    out_data_numpy = out_data.cpu().numpy()

    if device.type == "cuda":
        torch.cuda.empty_cache()

    return out_data_numpy
