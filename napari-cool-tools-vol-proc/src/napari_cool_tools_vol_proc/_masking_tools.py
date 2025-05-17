"""
This module contains code for mainpulating volumetric data with label masks.
"""

import numpy as np
from math import sqrt
from typing import Tuple,List
from napari.utils.notifications import show_info
from napari.qt.threading import thread_worker
from napari.layers import Image, Labels, Layer
from napari_cool_tools_io import viewer
from napari_cool_tools_vol_proc._masking_tools_funcs import (
    isolate_labeled_volume, 
    project_2d_mask,create_blank_lable_from_layer,
    mask_relative_to_existing_label,
    mask_interface_of_existing_label,
    bscan_label_cleanup,
    group_labels
) 

def bscan_label_cleanup_plugin(lbl:Labels,input_label_vals=[0,1,2],lower_threshold:int=2,upper_threshold:int=400,output_label_val:int=10,hightlight_features:bool=True,debug:bool=False):
    '''
    Args:
    Returns:
    Raises:
    '''
    bscan_label_cleanup_thread(lbl=lbl,input_label_vals=input_label_vals,lower_threshold=lower_threshold,upper_threshold=upper_threshold,output_label_val=output_label_val,hightlight_features=hightlight_features,debug=debug)
    return



@thread_worker(connect={"returned": viewer.add_layer})
def bscan_label_cleanup_thread(lbl:Labels,input_label_vals:list=[0,1,2],lower_threshold:int=2,upper_threshold:int=400,output_label_val:int=10,hightlight_features:bool=True,debug:bool=False)->Layer:
    '''
    Args:
    Returns:
    Raises:
    '''
    show_info(f'Bscan Label Cleanup thread started')


    name = f'{lbl.name}_CleanLbls'
    layer_type = 'labels'
    add_kwargs = {'name':f'{name}'}


    proc_data = bscan_label_cleanup(data=lbl.data,input_label_vals=input_label_vals,lower_threshold=lower_threshold,upper_threshold=upper_threshold,output_label_val=output_label_val,hightlight_features=hightlight_features,debug=debug)


    layer = Layer.create(proc_data,add_kwargs,layer_type)


    show_info(f'Bscan Label Cleanup thread completed')
    return layer

def group_labels_plugin(lbl:Labels,input_label_vals:List=[0,]):
    '''
    Args:
    Returns:
    Raises:
    '''
    group_labels_thread(lbl=lbl,input_label_vals=input_label_vals)
    return



@thread_worker(connect={"returned": viewer.add_layer})
def group_labels_thread(lbl:Labels,input_label_vals=[1,2])->Layer:
    '''
    Args:
    Returns:
    Raises:
    '''
    show_info(f'Group Labels thread started')


    name = f'{lbl.name}_CleanLbls'
    layer_type = 'labels'
    add_kwargs = {'name':f'{name}'}


    proc_data = group_labels(data=lbl.data,input_label_vals=input_label_vals)


    layer = Layer.create(proc_data,add_kwargs,layer_type)


    show_info(f'Bscan Label Cleanup thread completed')
    return layer

def mask_relative_to_existing_label_plugin(lbl:Labels,occurence="first",relative="before",axis:int=0,input_label_val=1,output_label_val=10,volumetric_calc:bool=False):
    '''
    Args:
    Returns:
    Raises:
    '''
    mask_relative_to_existing_label_thread(lbl=lbl,occurence=occurence,relative=relative,axis=axis,input_label_val=input_label_val,output_label_val=output_label_val,volumetric_calc=volumetric_calc)
    return



@thread_worker(connect={"returned": viewer.add_layer})
def mask_relative_to_existing_label_thread(lbl:Labels,occurence="first",relative="before",axis:int=0,input_label_val=1,output_label_val=10,volumetric_calc:bool=False)->Layer:
    '''
    Args:
    Returns:
    Raises:
    '''
    show_info(f'Mask Relative thread started')


    name = f'{lbl.name}_RelMask'
    layer_type = 'labels'
    add_kwargs = {'name':f'{name}'}


    proc_data = mask_relative_to_existing_label(data=lbl.data,occurence=occurence,relative=relative,axis=axis,input_label_val=input_label_val,output_label_val=output_label_val,volumetric_calc=volumetric_calc)

    print(f"proc_data shape: {proc_data.shape},proc_data values: {np.unique(proc_data)}\n")

    layer = Layer.create(proc_data,add_kwargs,layer_type)


    show_info(f'Mask Relative thread completed')
    return layer


def mask_interface_of_existing_label_plugin(lbl:Labels,occurence="first",inverse:bool=False,axis:int=0,input_label_val=1,output_label_val=10,volumetric_calc:bool=False):
    '''
    Args:
    Returns:
    Raises:
    '''
    mask_interface_of_existing_label_thread(lbl=lbl,occurence=occurence,inverse=inverse,axis=axis,input_label_val=input_label_val,output_label_val=output_label_val,volumetric_calc=volumetric_calc)
    return



@thread_worker(connect={"returned": viewer.add_layer})
def mask_interface_of_existing_label_thread(lbl:Labels,occurence="first",inverse:bool=False,axis:int=0,input_label_val=1,output_label_val=10,volumetric_calc:bool=False)->Layer:
    '''
    Args:
    Returns:
    Raises:
    '''
    show_info(f'Mask Relative thread started')


    name = f'{lbl.name}_RelMask'
    layer_type = 'labels'
    add_kwargs = {'name':f'{name}'}


    proc_data = mask_interface_of_existing_label(data=lbl.data,occurence=occurence,inverse=inverse,axis=axis,input_label_val=input_label_val,output_label_val=output_label_val,volumetric_calc=volumetric_calc)

    print(f"proc_data shape: {proc_data.shape},proc_data values: {np.unique(proc_data)}\n")

    layer = Layer.create(proc_data,add_kwargs,layer_type)


    show_info(f'Mask Relative thread completed')
    return layer


def create_blank_lable_from_layer_plugin(img:Image)->Labels:
    """"""
    name = f"{img.name}_labels"
    labels = create_blank_lable_from_layer(img.data)
    layer_type ="labels"
    add_kwargs = {"name":f"{name}"}
    layer = Layer.create(labels,add_kwargs,layer_type)
    return layer

def isolate_labeled_volume_plugin(img:Image,label_vol:Labels,label:int)->Image:
    """"""
    isolate_labeled_volume_thread(img=img,label_vol=label_vol,label=label)
    return
    
@thread_worker(connect={"returned": viewer.add_layer})
def isolate_labeled_volume_thread(img:Image,label_vol:Labels,label:int)->Image:
    """"""
    show_info(f"Isolate labeled volume thread started")
    name = f"{img.name}_{label}_mask"
    layer_type = 'image'
    add_kwargs = {"name":f"{name}"}

    out_data = isolate_labeled_volume(img_data=img.data,lbl_data=label_vol.data,label=label)

    layer = Layer.create(out_data,add_kwargs,layer_type)
    show_info(f"Isolate labeled volume thread completed")

    return layer

def project_2d_mask_plugin(img:Image,label_vol:Labels,axis:int=1,swap_axes:bool=True)->Labels:
    """"""
    project_2d_mask_thread(img=img,label_vol=label_vol,axis=axis,swap_axes=swap_axes)
    return

@thread_worker(connect={"returned": viewer.add_layer})
def project_2d_mask_thread(img:Image,label_vol:Labels,axis:int=1,swap_axes:bool=True)->Labels:
    """"""
    show_info(f"Project Label along asis thread started")
    name = f"{label_vol.name}_prjct({axis})"
    layer_type = 'labels'
    add_kwargs = {"name":f"{name}"}

    out_data = project_2d_mask(img_data=img.data,lbl_data=label_vol.data,axis=axis,swap_axes=swap_axes)

    layer = Layer.create(out_data,add_kwargs,layer_type)
    show_info(f"Project Label along asis thread completed")

    return layer


def find_brightest_avg_pixels(vol:Image,pixels_to_avg:int=8,axis:int=1) -> Labels:
    ''''''
    data = vol.data

    reshape = data.reshape(-1,pixels_to_avg,data.shape[-1])
    reshape_avg = reshape.mean(1)

    viewer.add_image(reshape_avg)

    col_idx = reshape_avg.max(0).astype(np.uint8)
    row_idx = np.arange(reshape_avg.shape[1])

    print(col_idx,row_idx)

    bright_mask = np.zeros_like(reshape_avg,dtype=np.uint8)
    bright_mask[(col_idx,row_idx)] = 2

    print(col_idx,row_idx)
    viewer.add_labels(bright_mask)


    
