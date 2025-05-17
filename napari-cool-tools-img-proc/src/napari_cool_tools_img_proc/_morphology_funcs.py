"""
This module contains function code for morphological manipulation of images
"""
import gc
from tqdm import tqdm
import torch
from napari.types import ImageData
import kornia.morphology as morph
from napari_cool_tools_io import device

def morphological_dilation(data:ImageData,kernel_size:int=3,custom_kenel=False,volumetric_calc:bool=False,use_gpu:bool=False)->ImageData:
    '''
    Args:
    Returns:
    Raises:
    '''
    if not use_gpu:
        proc = "cpu"
    else:
        proc = "cuda"

    prev_type = data.dtype
    data_t = torch.tensor(data.copy()).to(torch.float32).to(proc)

    dims_needed = 4 - data_t.ndim
    if dims_needed < 4:
        for _ in  range(dims_needed):
            data_t = data_t.unsqueeze(0)
    elif dims_needed == 0:
        pass
    else:
        raise ValueError(f"morphological dilation function accepts inputs of 4 dimensions or less, {data_t.ndim} is too many")

    if not custom_kenel:
        kernel = torch.ones((kernel_size,kernel_size)).to(proc)
    else:
        kernel = torch.tensor([
            [0,0,1,0,0],
            [0,1,1,1,0],
            [1,1,1,1,1],
            [0,1,1,1,0],
            [0,0,1,0,0]
        ]).to(proc)

    print(f"data_t shape: {data_t.shape}\n")

    if (data_t.ndim == 1 or volumetric_calc) and not use_gpu:
        out_data_t = morph.dilation(data_t,kernel).squeeze()
    else:
        out_data_t = torch.zeros_like(data_t).to(proc)
        for i in tqdm(range(data_t.shape[-3]),desc="Dilating"):
            out_data_t[:,i,:,:] = morph.dilation(data_t[:,i,:,:].unsqueeze(0),kernel).squeeze()
        
    out_data = out_data_t.cpu().squeeze().numpy().astype(prev_type)

    del kernel
    del data_t
    del out_data_t

    gc.collect()
    torch.cuda.empty_cache()

    return out_data


def morphological_erosion(data:ImageData,kernel_size:int=3,custom_kenel=False,volumetric_calc:bool=False,use_gpu:bool=False)->ImageData:
    '''
    Args:
    Returns:
    Raises:
    '''
    prev_type = data.dtype
    data_t = torch.tensor(data.copy()).to(torch.float32)

    dims_needed = 4 - data_t.ndim
    if dims_needed < 4:
        for _ in  range(dims_needed):
            data_t = data_t.unsqueeze(0)
    elif dims_needed == 0:
        pass
    else:
        raise ValueError(f"morphological erosion function accepts inputs of 4 dimensions or less, {data_t.ndim} is too many")

    if not custom_kenel:
        kernel = torch.ones((kernel_size,kernel_size))
    else:
        kernel = torch.tensor([
            [0,0,1,0,0],
            [0,1,1,1,0],
            [1,1,1,1,1],
            [0,1,1,1,0],
            [0,0,1,0,0]
        ])

    print(f"data_t shape: {data_t.shape}\n")

    if data_t.ndim == 1 or volumetric_calc:
        out_data_t = morph.erosion(data_t,kernel).squeeze()
    else:
        out_data_t = torch.zeros_like(data_t)
        for i in tqdm(range(data_t.shape[-3]),desc="Eroding"):
            out_data_t[:,i,:,:] = morph.erosion(data_t[:,i,:,:].unsqueeze(0),kernel).squeeze()
        
    out_data = out_data_t.cpu().squeeze().numpy().astype(prev_type)


    return out_data
    