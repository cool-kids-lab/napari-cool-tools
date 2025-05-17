"""
This module contains code for averaging 2D slices
"""
import numpy as np
from tqdm import tqdm
from skimage.measure import block_reduce
from napari.utils.notifications import show_info
from napari.qt.threading import thread_worker
from napari.layers import Image, Layer
from napari_cool_tools_io import viewer
from napari_cool_tools_vol_proc._averaging_tools_funcs import average_bscans, average_per_bscan, average_per_bscan_pt, Implementation

def average_bscans_plugin(vol:Image, scans_per_avg:int=3):
    """Function averaging every scans_per_avg images/B-scans togehter.
    Args:
        vol (Image): vol representing volumetric or image stack data
        scans_per_avg (int): number of consecutive images/B-scans to average together

    Returns:
        Layer volume where values have been averaged every scans_per_avg images/B-scans along the depth dimension
    """
    
    average_bscans_thread(vol=vol,scans_per_avg=scans_per_avg)

    return

@thread_worker(connect={"returned": viewer.add_layer})
def average_bscans_thread(vol:Image, scans_per_avg:int=3) -> Layer:
    """Function averaging every scans_per_avg images/B-scans togehter.
    Args:
        vol (Image): vol representing volumetric or image stack data
        scans_per_avg (int): number of consecutive images/B-scans to average together

    Returns:
        Layer volume where values have been averaged every scans_per_avg images/B-scans along the depth dimension
    """

    name = f"{vol.name}_avg_{scans_per_avg}"
    add_kwargs = {"name":name}
    layer_type = "image"
    show_info(f'Average B-scans thread has started')
    avg_data = average_bscans(vol.data,scans_per_avg=scans_per_avg)
    show_info(f'Average B-scans thread has completed')
    layer = Layer.create(avg_data,add_kwargs,layer_type)

    return layer

def average_per_bscan_plugin(vol: Image, scans_per_avg: int = 3, axis = 0, ensemble:bool=True, gauss:bool=True, trim: bool = False, implementation:Implementation=Implementation.TORCH):
    """Function averaging every scans_per_avg images/B-scans centered around each image/b-scan.
    Args:
        vol (Image): vol representing volumetric or image stack data
        scans_per_avg (int): number of consecutive images/B-scans to average together
        trim: (bool): Flag indicating that ends should be trimmed if image/B-scan index is less than (scans_per_avg - 1 / 2)

    Returns:
        Layer volume where values at each index each slice is an average of the surrounding bscans from vol
    """

    average_per_bscan_thread(vol=vol,scans_per_avg=scans_per_avg,axis=axis,trim=trim,implementation=implementation,ensemble=ensemble,gauss=gauss)

    return

@thread_worker(connect={"returned": viewer.add_layer})    
def average_per_bscan_thread(vol: Image, scans_per_avg: int = 3, axis = 0, ensemble:bool=True, gauss:bool=True, trim: bool = False, implementation:Implementation=Implementation.TORCH) -> Layer:
    """Function averaging every scans_per_avg images/B-scans centered around each image/b-scan.
    Args:
        vol (Image): vol representing volumetric or image stack data
        scans_per_avg (int): number of consecutive images/B-scans to average together
        trim: (bool): Flag indicating that ends should be trimmed if image/B-scan index is less than (scans_per_avg - 1 / 2)

    Returns:
        Layer volume where values at each index each slice is an average of the surrounding bscans from vol
    """

    name = f"{vol.name}_{scans_per_avg}_per"
    add_kwargs = {"name":name}
    layer_type = "image"
    show_info(f'Sliding Window B-scan average thread has started')
    if implementation == Implementation.TORCH:
        avg_data = average_per_bscan_pt(vol.data,scans_per_avg=scans_per_avg,axis=axis,trim=trim,ensemble=ensemble,gauss=gauss)
    elif implementation == Implementation.NUMPY:
        avg_data = average_per_bscan(vol.data,scans_per_avg=scans_per_avg,axis=axis,trim=trim)
    else:
        raise RuntimeError(f"A {implementation.value} version of this function has not been implemented.")
    
    show_info(f'Sliding Window B-scan average thread has completed')
    layer = Layer.create(avg_data,add_kwargs,layer_type)

    return layer