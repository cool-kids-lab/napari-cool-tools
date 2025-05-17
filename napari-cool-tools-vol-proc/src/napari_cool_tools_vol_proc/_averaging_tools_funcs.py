"""
This module contains code for averaging 2D slices
"""
import numpy as np
from enum import Enum
from tqdm import tqdm
from scipy.signal.windows import gaussian as sp_gauss
from skimage.measure import block_reduce
from napari.utils.notifications import show_info
from napari.types import ImageData
from napari.layers import Image, Layer
from napari_cool_tools_io import torch

class Implementation(Enum):
    NUMPY = "Numpy"
    TORCH = "Pytorch"
    KORNIA = "Kornia"
    CUPY = "Cupy"

def average_bscans(vol:ImageData, scans_per_avg:int=3) -> ImageData:
    """Function averaging every scans_per_avg images/B-scans togehter.
    Args:
        vol (ImageData): vol representing volumetric or image stack data
        scans_per_avg (int): number of consecutive images/B-scans to average together

    Returns:
        ImageData volume where values have been averaged every scans_per_avg images/B-scans along the depth dimension
    """

    averaged_array = block_reduce(vol, block_size=(scans_per_avg,1,1), func= np.mean)

    return averaged_array

def average_per_bscan(vol: ImageData, scans_per_avg: int = 3, axis = 0, trim: bool = False) -> ImageData:
    """Function averaging every scans_per_avg images/B-scans centered around each image/b-scan.
    Args:
        vol (ImageData): vol representing volumetric or image stack data
        scans_per_avg (int): number of consecutive images/B-scans to average together
        trim: (bool): Flag indicating that ends should be trimmed if image/B-scan index is less than (scans_per_avg - 1 / 2)

    Returns:
        ImageData volume where values at each index each slice is an average of the surrounding bscans from vol
    """
    
    if scans_per_avg % 2 == 1:
        offset = int((scans_per_avg - 1) / 2)

        #print(f"shape: {data.shape}, axis: {axis}, length of axis {data.shape[axis]}")

        length = vol.shape[axis]

        averaged_slices = []

        for i in tqdm(range(length),desc="Avg per B-scan"):
            if i >= offset and i < length - offset:
                #print(f"Averaging slices...\nGenerating new slice by averaging slices {i-offset} through {i+offset} of {length-1}")
                
                if axis == 0:
                    start0 = i-offset
                    end0 = i+offset+1
                    start1 = 0
                    end1 = vol.shape[1]
                    start2 = 0
                    end2 = vol.shape[2]
                elif axis == 1:
                    start0 = 0
                    end0 = vol.shape[0]
                    start1 = i-offset
                    end1 = i+offset+1
                    start2 = 0
                    end2 = vol.shape[2]
                elif axis == 2:
                    start0 = 0
                    end0 = vol.shape[0]
                    start1 = 0
                    end1 = vol.shape[1]
                    start2 = i-offset
                    end2 = i+offset+1
                else:
                    print(f"You done effed up!!")

                averaged_slice = vol[start0:end0,start1:end1,start2:end2].mean(axis)                
                
                #averaged_slice = data[i-offset:i+offset+1,:,:].mean(axis)
                averaged_slices.append(averaged_slice)
            else:
                if trim == False:
                    if i < offset:

                        if axis == 0:
                            start0 = i
                            end0 = i+1
                            start1 = 0
                            end1 = vol.shape[1]
                            start2 = 0
                            end2 = vol.shape[2]
                        elif axis == 1:
                            start0 = 0
                            end0 = vol.shape[0]
                            start1 = i
                            end1 = i+1
                            start2 = 0
                            end2 = vol.shape[2]
                        elif axis == 2:
                            start0 = 0
                            end0 = vol.shape[0]
                            start1 = 0
                            end1 = vol.shape[1]
                            start2 = i
                            end2 = i+1
                        else:
                            print(f"You done effed up!!")

                        averaged_slice = vol[start0:end0,start1:end1,start2:end2].squeeze(axis)

                        averaged_slices.append(averaged_slice)    

                        pass

                    elif i >= length - offset:

                        if axis == 0:
                            start0 = i
                            end0 = i+1
                            start1 = 0
                            end1 = vol.shape[1]
                            start2 = 0
                            end2 = vol.shape[2]
                        elif axis == 1:
                            start0 = 0
                            end0 = vol.shape[0]
                            start1 = i
                            end1 = i+1
                            start2 = 0
                            end2 = vol.shape[2]
                        elif axis == 2:
                            start0 = 0
                            end0 = vol.shape[0]
                            start1 = 0
                            end1 = vol.shape[1]
                            start2 = i
                            end2 = i+1
                        else:
                            print(f"You done effed up!!")

                        averaged_slice = vol[start0:end0,start1:end1,start2:end2].squeeze(axis)

                        averaged_slices.append(averaged_slice)  

                        pass
                else:
                    #print(f"You shouldn't be here {average_per_bscan}!!")
                    pass

        averaged_array = np.stack(averaged_slices, axis=axis)

        return averaged_array
    else:
        print(f"scans_per_avg should be an odd number please use an odd number for this value")
        show_info(f"scans_per_avg should be an odd number please use an odd number for this value")
        return vol

def average_per_bscan_pt(vol: ImageData, scans_per_avg: int = 3, axis = 0, trim: bool = False, ensemble:bool=True, gauss:bool=False) -> ImageData:
    """Function averaging every scans_per_avg images/B-scans centered around each image/b-scan.
    Args:
        vol (ImageData): vol representing volumetric or image stack data
        scans_per_avg (int): number of consecutive images/B-scans to average together
        trim (bool): Flag indicating that ends should be trimmed if image/B-scan index is less than (scans_per_avg - 1 / 2)
        ensemble (bool): Flag indicating that ensemble average should be genearated average is calculated for all 3 major axes
                         and the results are then averaged generating a more accurate result at the cost of speed.

    Returns:
        ImageData volume where values at each index each slice is an average of the surrounding bscans from vol
    """
    
    vol_t = torch.as_tensor(vol.copy())
    #vol_t = torch.as_tensor(np.ascontiguousarray(vol)) # compare with copy() # as contiguous is tied to the numpy array in memory and would be good for inplace changes
    buffer = int(scans_per_avg/2)
    # select axis
    #if axis !=0:
    #    vol_t = vol_t.swapaxes(0,axis).contiguous()

    ''' Stubb for implementation
    if gauss:
        g_dist = sp_gauss(scans_per_avg,1)
        w = g_dist[:buffer] # distribution will be symmetrical about 1 at current index take initial buffer sized array
    else:
        w = np.ones((buffer,),dtype=np.uint8) # make the weight 1 in case gaussian is not used
    '''

    if not ensemble:

        if axis != 0:
            vol_t = vol_t.swapaxes(0,axis)

        avg_t = torch.zeros_like(vol_t)
        prev_vol = torch.zeros_like(vol_t)
        next_vol = torch.zeros_like(vol_t)

        # calc indicies
        axis_len = vol_t.shape[0]
        idxs = torch.arange(buffer,axis_len-buffer)
        prev_i = idxs - buffer
        next_i = idxs + buffer

        #print(f"prev_i: {prev_i.shape}\n\nindxs: {idxs.shape}\n\nnext_i: {next_i.shape}\n\n")

        prev_vol[idxs] = vol_t[prev_i]
        next_vol[idxs] = vol_t[next_i]

        # generate data between prev or next and buffer and sum
        for i in range(buffer-1):
            prev_vol[idxs] = prev_vol[idxs] + vol_t[prev_i+i]
            next_vol[idxs] = next_vol[idxs] + vol_t[next_i-i]

        #print(f"prev_vol shape: {prev_vol.shape}, vol_i shape: {vol_t[idxs].shape}, next_vol shape: {next_vol.shape}\n")

        # calculate avg
        vol_t[idxs] = ((prev_vol[idxs] + vol_t[idxs] + next_vol[idxs])/scans_per_avg)

        if axis!=0:
            vol_t = vol_t.swapaxes(0,axis)

        avg_out = vol_t.cpu().numpy()

    else:
        vol_t1 = vol_t.swapaxes(0,1)
        vol_t2 = vol_t.swapaxes(0,2)
        
        #vol_t1 = torch.empty_like(vol_t)
        #vol_t2 = torch.empty_like(vol_t)

        axis_0_len = len(vol_t)
        axis_1_len = vol_t.shape[1]
        axis_2_len = vol_t.shape[2]

        idxs_0 = torch.arange(buffer,axis_0_len-buffer)
        prev_i_0 = idxs_0 - 1
        next_i_0 = idxs_0 + 1

        idxs_1 = torch.arange(buffer,axis_1_len-buffer)
        prev_i_1 = idxs_1 - 1
        next_i_1 = idxs_1 + 1

        idxs_2 = torch.arange(buffer,axis_2_len-buffer)
        prev_i_2 = idxs_2 - 1
        next_i_2 = idxs_2 + 1

        prev_vol = torch.zeros_like(vol_t)
        next_vol = torch.zeros_like(vol_t)
        prev_vol_1 = torch.zeros_like(vol_t1)
        next_vol_1 = torch.zeros_like(vol_t1)
        prev_vol_2 = torch.zeros_like(vol_t2)
        next_vol_2 = torch.zeros_like(vol_t2)

        prev_vol[idxs_0] = vol_t[prev_i_0]
        next_vol[idxs_0] = vol_t[next_i_0]
        prev_vol_1[idxs_1] = vol_t1[prev_i_1]
        next_vol_1[idxs_1] = vol_t1[next_i_1]
        prev_vol_2[idxs_2] = vol_t2[prev_i_2]
        next_vol_2[idxs_2] = vol_t2[next_i_2]

        # generate data between prev or next and buffer and sum
        for i in range(buffer-1):
            prev_vol[idxs_0] = prev_vol[idxs_0] + vol_t[prev_i_0+i]
            next_vol[idxs_0] = next_vol[idxs_0] + vol_t[next_i_0-i]
            prev_vol_1[idxs_1] = prev_vol_1[idxs_1] + vol_t1[prev_i_1+i]
            next_vol_1[idxs_1] = next_vol_1[idxs_1] + vol_t1[next_i_1-i]
            prev_vol_2[idxs_2] = prev_vol_2[idxs_2] + vol_t2[prev_i_2+i]
            next_vol_2[idxs_2] = next_vol_2[idxs_2] + vol_t2[next_i_2-i]

        # calculate avg
        vol_t[idxs_0] = ((prev_vol[idxs_0] + vol_t[idxs_0] + next_vol[idxs_0])/scans_per_avg)
        vol_t1[idxs_1] = ((prev_vol_1[idxs_1] + vol_t1[idxs_1] + next_vol_1[idxs_1])/scans_per_avg)
        vol_t2[idxs_2] = ((prev_vol_2[idxs_2] + vol_t2[idxs_2] + next_vol_2[idxs_2])/scans_per_avg)

        vol_t1 = vol_t1.swapaxes(0,1)
        vol_t2 = vol_t2.swapaxes(0,2)

        avg_out = ((vol_t+vol_t1+vol_t2)/3).cpu().numpy()

    
    # trim result if necessary (perhaps adjust for 3D trim of ensemble and alternate axes)
    if trim:
        trim_offset = int((scans_per_avg-1)/2)
        avg_out = avg_out[trim_offset:len(vol)-trim_offset]


    return avg_out