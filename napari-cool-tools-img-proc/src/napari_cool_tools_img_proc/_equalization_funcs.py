""" """

import gc

import numpy as np
from napari.types import ImageData
from napari_cool_tools_io import device, torch, memory_stats
from tqdm import tqdm

from napari_cool_tools_img_proc import DType
from napari_cool_tools_img_proc._normalization_funcs import (
    normalize_data_in_range_func,
    normalize_data_in_range_pt_func,
)


# def init_bscan_preproc(img:ImageData,num_std:int=4,min_intensity:float=0.0,max_intensity:float=1.0,dtype:DTYPE=DTYPE.NP_FLOAT64):
# TODO Move this to denoise funcs 
def init_bscan_preproc(
    img: ImageData,
    num_std: int = 16,
    min_intensity: float = 0.0,
    max_intensity: float = 1.0,
    dtype: DType = DType.NP_FLOAT32,
):
    """
    Args:
    Returns:
    Raises:
    """
    out_img = background_removal_func(img)
    out_img = auto_brightness_adjust(
        out_img,
        num_std=num_std,
        min_intensity=min_intensity,
        max_intensity=max_intensity,
        dtype=dtype,
    )

    return out_img

def init_bscan_preproc_pt(
    img: ImageData,
    num_std: int = 16,
    min_intensity: float = 0.0,
    max_intensity: float = 1.0,
    dtype: DType = DType.NP_FLOAT16,
    use_accelerator: bool = True,
    numpy_out: bool = True,
    verbose: bool = False
):
    """
    Args:
    Returns:
    Raises:
    """
    # Background removal
    out_img_t = background_removal_pt(img,use_accelerator=use_accelerator,numpy_out=True,verbose=verbose)
    out_img_t = auto_brightness_adjust_pt(
        out_img_t,
        use_accelerator=use_accelerator,
        numpy_out=False,
        verbose=verbose,
    )

    if not numpy_out:
        # Clear cache to free up memory
        if use_accelerator:
            torch.cuda.empty_cache()
            if verbose:
                memory_stats()

        return out_img_t
    else:
        out_img_t = out_img_t.cpu().numpy().astype(dtype.value)
        if use_accelerator:
            torch.cuda.empty_cache()
            if verbose:
                memory_stats()

        return out_img_t


def background_removal_func(img: ImageData):
    """
    Args:
    Returns:
    Raises:
    """
    img_norm = normalize_data_in_range_func(
        img, min_val=0, max_val=1
    )  # (img-img.min())/(img.max()-img.min())
    img_adjust = np.clip((img_norm - img_norm.mean()), 0, 1)
    output_norm = normalize_data_in_range_func(
        img_adjust, min_val=0, max_val=1
    )  # (img_adjust-img_adjust.min())/(img_adjust.max()-img.min())
    return output_norm

def background_removal_pt(img: ImageData, numpy_out: bool = True, use_accelerator: bool = True, verbose: bool = False):
    """
    Args:
    Returns:
    Raises:
    """
    # set device
    if use_accelerator:
        current_device = device
    else:
        current_device = "cpu"

    # convert to tensor if necessary
    img = torch.as_tensor(img,device=current_device,dtype=torch.float16)

    img = normalize_data_in_range_pt_func(img,numpy_out=False,use_accelerator=use_accelerator)
    img = torch.clamp((img-img.mean()),0,1)
    img = normalize_data_in_range_pt_func(img,numpy_out=False,use_accelerator=use_accelerator)

    if numpy_out:
        img = img.detach().cpu().numpy()
        #del max_val, min_val
        gc.collect()

        # Clear cache to free up memory
        if use_accelerator:
            torch.cuda.empty_cache()
            if verbose:
                memory_stats()

        return img
    else:
        # Clear cache to free up memory
        if use_accelerator:
            torch.cuda.empty_cache()
            if verbose:
                memory_stats()
        return  img

def auto_brightness_adjust(
    img: ImageData,
    num_std: int = 16,
    min_intensity: float = 0.0,
    max_intensity: float = 1.0,
    dtype: DType = DType.NP_FLOAT32,
    in_place: bool = True,
):
    """
    Args:
    Returns:
    Raises:
    """
    # this should typically be run following background removal
    # calc non_zero mean and std
    non_zero_mask = img > 0
    max_val = img.max()
    non_zero_mean, non_zero_std = img[non_zero_mask].mean(), img[non_zero_mask].std()
    non_zero_total = len(img[non_zero_mask].flatten())

    # calc samples within num_std stds
    desired_stds = non_zero_std * num_std
    new_max = non_zero_mean + desired_stds
    desired_std_mask = img > new_max

    out_img = img

    out_img[desired_std_mask] = new_max
    out_img = normalize_data_in_range_func(
        out_img, min_val=min_intensity, max_val=max_intensity
    ).astype(dtype.value)

    # non zero percentage
    non_zero_desired_std_mask = (img > 0) & (img < new_max)
    desired_std_nonzero = len(img[non_zero_desired_std_mask].flatten())
    non_zero_percentage = desired_std_nonzero / non_zero_total
    print(
        f"Nonzero mean,std: ({non_zero_mean},{non_zero_std}), {num_std} stds above the mean includes {non_zero_percentage} of all nonzero values.\n"
    )
    print(f"New max intensity: {new_max} vs old max intensity: {max_val}\n")

    return out_img

def auto_brightness_adjust_pt(
    img: ImageData,
    num_std: int = 16,
    min_intensity: float = 0.0,
    max_intensity: float = 1.0,
    dtype: DType = DType.NP_FLOAT16,
    in_place: bool = True,
    use_accelerator: bool = True,
    numpy_out: bool = True,
    verbose: bool = True,
):
    """
    Args:
    Returns:
    Raises:
    """

    # set device
    if use_accelerator:
        current_device = device
    else:
        current_device = "cpu"
    
    # convert to tensor if necessary
    img = torch.as_tensor(img,device=current_device,dtype=torch.float16)

    # this should typically be run following background removal
    # calc non_zero mean and std
    non_zero_mask = img > 0
    max_val = img.max()
    non_zero_mean, non_zero_std = img[non_zero_mask].mean(), img[non_zero_mask].std()
    non_zero_total = len(img[non_zero_mask].flatten())

    del non_zero_mask
    gc.collect()
    torch.cuda.empty_cache()

    # calc samples within num_std stds
    desired_stds = non_zero_std * num_std
    new_max = non_zero_mean + desired_stds
    desired_std_mask = img > new_max

    img[desired_std_mask] = new_max

    del desired_std_mask
    gc.collect()
    torch.cuda.empty_cache()

    img = normalize_data_in_range_pt_func(
        img, min_val=min_intensity,max_val=max_intensity,numpy_out=False,use_accelerator=use_accelerator
    )

    if verbose:
        # non zero percentage
        non_zero_desired_std_mask = (img > 0) & (img < new_max)
        desired_std_nonzero = len(img[non_zero_desired_std_mask].flatten())
        non_zero_percentage = desired_std_nonzero / non_zero_total
        print(
            f"Nonzero mean,std: ({non_zero_mean},{non_zero_std}), {num_std} stds above the mean includes {non_zero_percentage} of all nonzero values.\n"
        )
        print(f"New max intensity: {new_max} vs old max intensity: {max_val}\n")
        del non_zero_desired_std_mask, desired_std_nonzero, non_zero_percentage

    del max_val, non_zero_mean, non_zero_total, desired_stds, new_max
    gc.collect()

    if numpy_out:
        img = img.detach().cpu().numpy().astype(dtype.value)
        #del max_val, min_val
        gc.collect()

        # Clear cache to free up memory
        if use_accelerator:
            torch.cuda.empty_cache()
            if verbose:
                memory_stats()

        return img
    else:
        # Clear cache to free up memory
        if use_accelerator:
            torch.cuda.empty_cache()
            if verbose:
                memory_stats()
        return  img


def clahe_func(
    data: ImageData,
    kernel_size=None,
    clip_limit: float = 0.01,
    nbins=256,
    norm_min=0,
    norm_max=1,
) -> ImageData:
    """"""
    from skimage.exposure import equalize_adapthist

    if data.ndim != 2 and data.ndim != 3:
        raise RuntimeError("CLAHE only works for data of 2 or 3 dimensions")

    dtype_in = data.dtype
    norm_data = normalize_data_in_range_func(data, min_val=norm_min, max_val=norm_max)

    if data.ndim == 2:
        init_out = equalize_adapthist(
            norm_data, kernel_size=kernel_size, clip_limit=clip_limit, nbins=nbins
        )
        img_out = init_out.astype(dtype_in)

    elif data.ndim == 3:
        for i in tqdm(range(len(data)), desc="CLAHE"):
            norm_data[i] = equalize_adapthist(
                norm_data[i],
                kernel_size=kernel_size,
                clip_limit=clip_limit,
                nbins=nbins,
            )

        img_out = norm_data.astype(dtype_in)

    return img_out


def clahe_pt_func(
    data: ImageData,
    kernel_size=None,
    clip_limit: float = 40.0,
    nbins=256,
    norm_min=0,
    norm_max=1,
) -> ImageData:
    """"""

    from kornia.enhance import equalize_clahe

    if data.ndim != 2 and data.ndim != 3:
        raise RuntimeError("CLAHE only works for data of 2 or 3 dimensions")

    norm_data = normalize_data_in_range_pt_func(
        data, min_val=norm_min, max_val=norm_max, numpy_out=False
    )
    # pt_data = torch.tensor(norm_data,device=device)
    pt_data = norm_data.to(device)

    if data.ndim == 2:
        equalized = equalize_clahe(pt_data, clip_limit)
        out_data = equalized.detach().cpu().numpy()
        del equalized
    elif data.ndim == 3:
        for i in tqdm(range(len(pt_data)), desc="CLAHE(PT)"):
            pt_data[i] = equalize_clahe(pt_data[i], clip_limit)

        out_data = pt_data.detach().cpu().numpy()

    del (
        norm_data,
        pt_data,
    )
    gc.collect()
    torch.cuda.empty_cache()

    gpu_mem_clear = torch.cuda.memory_allocated() == torch.cuda.memory_reserved() == 0
    print(f"GPU memory is clear: {gpu_mem_clear}\n")

    if not gpu_mem_clear:
        print(f"{torch.cuda.memory_summary()}\n")

    return out_data
