"""
"""

import gc
import numpy as np
from napari.types import ImageData
from napari_cool_tools_io import torch, device


def adjust_log_func(data:ImageData, gain:float=1, inv:bool=False) -> ImageData:
    """Pass through function of skimage.exposure adjust_log function.
    
    Args:
        img (Image): Image to be adjusted.
        gain (float): constant multiplier.
        inv (bool): If True performs inverse log correction instead of log correction.
        
    Returns:
        Logarithm corrected output image with '_LC' suffix added to name."""
    
    from skimage.exposure import adjust_log
    from tqdm import tqdm

    #data = data.copy()

    if data.ndim != 2 and data.ndim != 3:
        raise RuntimeError(f"CLAHE only works for data of 2 or 3 dimensions")
        
    if data.ndim == 2:
        log_corrected = adjust_log(data,gain=gain,inv=inv)
    elif data.ndim == 3:
        log_corrected = np.empty_like(data)
        for i in tqdm(range(len(data)),desc="Log Correction"):
            log_corrected[i] = adjust_log(data[i],gain=gain,inv=inv)

    return log_corrected

def adjust_log_pt_func(data:ImageData, gain:float=1, inv:bool=False, clip_output:bool=True) -> ImageData:
    """Pass through function of kornia.enhance adjust_log function.
    
    Args:
        img (Image): Image to be adjusted.
        gain (float): constant multiplier.
        inv (bool): If True performs inverse log correction instead of log correction.
        clip_output (bool, optional) – Whether to clip the output image with range of [0, 1]
        
    Returns:
        Logarithm corrected output image with '_LC' suffix added to name."""
    
    from kornia.enhance import adjust_log

    data = data.copy()

    if data.ndim != 2 and data.ndim != 3:
        raise RuntimeError(f"CLAHE only works for data of 2 or 3 dimensions")
    
    pt_data = torch.tensor(data,device=device)

    log_corrected = adjust_log(pt_data,gain=gain,inv=inv)
    out_data = log_corrected.detach().cpu().numpy()
    
    del(
        pt_data,
        log_corrected
    )
    gc.collect()
    torch.cuda.empty_cache()

    gpu_mem_clear = (torch.cuda.memory_allocated() == torch.cuda.memory_reserved() == 0)
    print(f"GPU memory is clear: {gpu_mem_clear}\n")

    if not gpu_mem_clear:
        print(f"{torch.cuda.memory_summary()}\n")

    return out_data