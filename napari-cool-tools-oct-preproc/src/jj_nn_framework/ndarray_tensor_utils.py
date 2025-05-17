"""
Utilities for working with ndarrays and tensors
"""
# bulitin libraries

# external libraries
import numpy as np
import torch
from numpy.typing import NDArray


def after_first_occurrence_1d(arr:NDArray,inclusive:bool=True):
    """
    """
    idx = arr.argmax()
    out = np.zeros_like(arr)
    limit = len(arr)
    if idx > 0:
        if inclusive:
            start = idx + 1
        else:
            start = idx
        stop = limit
        out[start:stop] = 1
  
    return out.astype(bool)

def after_last_occurrence_1d(arr:NDArray,inclusive:bool=True):
    """
    """
    limit = len(arr)
    out = np.zeros_like(arr)
    idx = arr[::-1].argmax()
    if idx > 0:
        if inclusive:
            start = limit-idx
        else:
            start = limit-(idx-1)
        stop = limit
        out[start:stop] = 1
    
    return out.astype(bool)

def find_first_occurrence_1d(arr:NDArray):
    """
    """
    idx = arr.argmax()
    out = np.zeros_like(arr)
    limit = len(arr)
    if idx > 0:
        start = idx
        stop = limit
        out[start:start+1] = 1
  
    return out.astype(bool)

def find_last_occurrence_1d(arr:NDArray):
    """
    """
    limit = len(arr)
    out = np.zeros_like(arr)
    idx = arr[::-1].argmax()
    if idx > 0:
        start = limit-idx
        stop = limit - (idx+1)
        out[start:start+1] = 1
    
    return out.astype(bool)

def replace_value_in_mask(arr:NDArray,mask:NDArray,val:float,new_val:float):
    """
    """
    masked_arr = arr * mask
    mask2 = masked_arr == val
    out_arr = arr.copy()
    out_arr[mask2] = new_val
    return out_arr

def reverse_slice(arr:NDArray,step=1)-> NDArray:
    """
    Generate a reverse slice for 1D NDArray
    """
    return slice(len(arr)-1,0,-step)

def select_indicies_along_axis(arr:NDArray,val:float,axis:int=0,occurrence="first",inverse:bool=False,inclusive:bool=True):
    """
    Given a value v select all indicies that appear after the first or last occurence of that value in the ndarray
    
    params
    returns
    errors
    
    """
    # get mask of desired value
    val_mask = arr == val
    if occurrence == "first":
        out_mask = np.apply_along_axis(find_first_occurrence_1d,axis,val_mask).astype(bool)
        if not inverse:
            return out_mask
        else:
            return ~out_mask
    elif occurrence == 'last':
        out_mask = np.apply_along_axis(find_last_occurrence_1d,axis,val_mask).astype(bool)
        if not inverse:
            return out_mask
        else:
            return ~out_mask
    else:
        raise ValueError(f"{occurrence} is not implemented for the relative parameter of select_incicies_along_axis\n Valid options are 'first' or 'last'\n")

def mask_indicies_along_axis(arr:NDArray,val:float,axis:int=0,occurrence="first",relative="after",inclusive:bool=True):
    """
    Given a value v select all indicies that appear after the first or last occurence of that value in the ndarray
    
    params
    returns
    errors
    
    """
    # get mask of desired value
    val_mask = val == arr
    if occurrence == "first":
        out_mask = np.apply_along_axis(after_first_occurrence_1d,axis,val_mask,{"inclusive":inclusive}).astype(bool)
        if relative == "after":
            return out_mask
        elif relative == "before":
            return ~out_mask
        else:
            raise ValueError(f"{relative} is not implemented for the relative parameter of select_incicies_along_axis\n Valid options are 'after' or 'before'\n")
    elif occurrence == 'last':
        out_mask = np.apply_along_axis(after_last_occurrence_1d,axis,val_mask,{"inclusive":inclusive}).astype(bool)
        if relative == "after":
            return out_mask
        elif relative == "before":
            return ~out_mask
        else:
            raise ValueError(f"{relative} is not implemented for the relative parameter of select_incicies_along_axis\n Valid options are 'after' or 'before'\n")
    else:
        raise ValueError(f"{occurrence} is not implemented for the relative parameter of select_incicies_along_axis\n Valid options are 'first' or 'last'\n")

def get_int_segs(arr:NDArray):
    """
    """
    


def gen_comp_arr_along_axis(arr:NDArray,axis:int=1):
    """
    generate comparison array by appending sentinel values to input array

    params
    returns
    errors

    """
    #axes = np.arange(arr.ndim)
    #other_axes = set(axes) - {axis} # as axis is an integer this is necessary to pass it to set() it must be iterable so set((axis,)) would work the {} to define sets is not setup for the NDarray type but set() works
    list_arr_shape = list(arr.shape)
    list_arr_shape[axis] = 1    # flatten along axis
    sentinel_shape = tuple(list_arr_shape)
    senteinel = np.full(sentinel_shape,-1)
    comp_arr = np.append(senteinel,arr,axis=axis)
    return comp_arr

def select_indicies_along_axis_vector_try(arr:NDArray,val:float,axis:int=0):
    """
    Given a value v select all indicies that appear after the last occurence of that value in the ndarray
    
    params
    returns
    errors
    
    """
    # get number of axes
    indxs = []
    num_axes = arr.ndim
    axis_len = arr.shape[axis]
    limit = axis_len - 1
    for ax in np.arange(num_axes):
        if ax == axis:
            indxs.append(reverse_slice(arr))
        else:
            indxs.append(slice(0,axis_len))

    mask = val == arr # new
    rev_axis_arr = arr[tuple(indxs)]
    #last_occur = rev_axis_arr.argmax(val,axis=axis)
    #rev_axis_arr = arr[tuple(indxs)] # new unfinished
    #last_occur = (limit) - last_occur

    return rev_axis_arr #last_occur