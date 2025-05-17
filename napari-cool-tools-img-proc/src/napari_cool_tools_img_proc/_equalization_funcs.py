"""
"""
import gc
from enum import Enum

import numpy as np
from tqdm import tqdm
from napari.types import ImageData
from napari_cool_tools_img_proc._normalization_funcs import normalize_data_in_range_func, normalize_data_in_range_pt_func
from napari_cool_tools_io import torch, device

class DTYPE(Enum):
    NP_FLOAT = 'f4' #np.dtype(np.float64)
    NP_UINT8 = 'u1' #np.dtype(np.uint8)

#def init_bscan_preproc(img:ImageData,num_std:int=4,min_intensity:float=0.0,max_intensity:float=1.0,dtype:DTYPE=DTYPE.NP_FLOAT):
def init_bscan_preproc(img:ImageData,num_std:int=4,min_intensity:float=0.0,max_intensity:float=255.0,dtype:DTYPE=DTYPE.NP_UINT8):
    '''
    Args:
    Returns:
    Raises:
    '''
    out_img = background_removal_func(img)
    out_img = auto_brightness_adjust(out_img,num_std=num_std,min_intensity=min_intensity,max_intensity=max_intensity,dtype=dtype)

    return out_img


def background_removal_func(img:ImageData):
    '''
    Args:
    Returns:
    Raises:
    '''
    img_norm = normalize_data_in_range_func(img,min_val=0,max_val=1) #(img-img.min())/(img.max()-img.min())
    img_adjust = np.clip((img_norm-img_norm.mean()),0,1)
    output_norm = normalize_data_in_range_func(img_adjust,min_val=0,max_val=1) #(img_adjust-img_adjust.min())/(img_adjust.max()-img.min())
    return  output_norm

def auto_brightness_adjust(img:ImageData,num_std:int=4,min_intensity:float=0.0,max_intensity:float=1.0,dtype:DTYPE=DTYPE.NP_FLOAT,in_place:bool=True):
    '''
    Args:
    Returns:
    Raises:
    '''
    # this should typically be run following background removal
    # calc non_zero mean and std
    non_zero_mask = img > 0
    max_val = img.max()
    non_zero_mean,non_zero_std = img[non_zero_mask].mean(),img[non_zero_mask].std()
    non_zero_total = len(img[non_zero_mask].flatten())

    # calc samples within num_std stds
    desired_stds = non_zero_std*num_std
    new_max = non_zero_mean + desired_stds 
    desired_std_mask = img > new_max

    out_img = img

    out_img[desired_std_mask] = new_max
    out_img = normalize_data_in_range_func(out_img,min_val=min_intensity,max_val=max_intensity).astype(dtype.value)

    # non zero percentage
    non_zero_desired_std_mask = (img > 0) & (img < new_max)
    desired_std_nonzero = len(img[non_zero_desired_std_mask].flatten())
    non_zero_percentage = desired_std_nonzero/non_zero_total
    print(f"Nonzero mean,std: ({non_zero_mean},{non_zero_std}), {num_std} stds above the mean includes {non_zero_percentage} of all nonzero values.\n")
    print(f"New max intensity: {new_max} vs old max intensity: {max_val}\n")

    return out_img


def clahe_func(data:ImageData, kernel_size=None,clip_limit:float=0.01,nbins=256,norm_min=0,norm_max=1) -> ImageData:
    ''''''
    from skimage.exposure import equalize_adapthist

    if data.ndim != 2 and data.ndim != 3:
        raise RuntimeError(f"CLAHE only works for data of 2 or 3 dimensions")
    
    dtype_in = data.dtype
    norm_data = normalize_data_in_range_func(data,min_val=norm_min,max_val=norm_max)

    if data.ndim == 2:
        init_out = equalize_adapthist(norm_data,kernel_size=kernel_size,clip_limit=clip_limit,nbins=nbins)
        img_out = init_out.astype(dtype_in)
        
    elif data.ndim == 3:
        for i in tqdm(range(len(data)),desc="CLAHE"):
            norm_data[i] = equalize_adapthist(norm_data[i],kernel_size=kernel_size,clip_limit=clip_limit,nbins=nbins)
        
        img_out = norm_data.astype(dtype_in)

    return img_out

def clahe_pt_func(data:ImageData, kernel_size=None,clip_limit:float=40.0,nbins=256,norm_min=0,norm_max=1) -> ImageData:
    """"""

    from kornia.enhance import equalize_clahe

    if data.ndim != 2 and data.ndim != 3:
        raise RuntimeError(f"CLAHE only works for data of 2 or 3 dimensions")

    dtype_in = data.dtype
    norm_data = normalize_data_in_range_pt_func(data,min_val=norm_min,max_val=norm_max,numpy_out=False)
    #pt_data = torch.tensor(norm_data,device=device)
    pt_data = norm_data.to(device)

    if data.ndim == 2:
        equalized = equalize_clahe(pt_data,clip_limit)
        out_data = equalized.detach().cpu().numpy()
        del equalized
    elif data.ndim == 3:
        for i in tqdm(range(len(pt_data)),desc="CLAHE(PT)"):
            pt_data[i] = equalize_clahe(pt_data[i],clip_limit)
        
        out_data = pt_data.detach().cpu().numpy()

    del(
        norm_data,
        pt_data,
    )
    gc.collect()
    torch.cuda.empty_cache()

    gpu_mem_clear = (torch.cuda.memory_allocated() == torch.cuda.memory_reserved() == 0)
    print(f"GPU memory is clear: {gpu_mem_clear}\n")

    if not gpu_mem_clear:
        print(f"{torch.cuda.memory_summary()}\n")

    return out_data