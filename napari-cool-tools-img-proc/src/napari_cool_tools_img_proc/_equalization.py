"""
This module contains code for equalizing image values
"""
from napari.utils.notifications import show_info
from napari.layers import Image, Layer
from napari.qt.threading import thread_worker
from napari_cool_tools_io import torch,viewer,memory_stats
from napari_cool_tools_img_proc._equalization_funcs import (
    DTYPE,
    clahe_func, clahe_pt_func,
    background_removal_func, init_bscan_preproc
)

def init_bscan_preproc_plugin(img:Image,num_std:int=16,min_intensity:float=0.0,max_intensity:float=255,dtype:DTYPE=DTYPE.NP_UINT8):
    '''
    Args:
    Returns:
    Raises:
    '''
    init_bscan_preproc_thread(img=img,num_std=num_std,min_intensity=min_intensity,max_intensity=max_intensity,dtype=dtype)
    return



@thread_worker(connect={"returned": viewer.add_layer})
def init_bscan_preproc_thread(img:Image,num_std:int=16,min_intensity:float=0.0,max_intensity:float=255,dtype:DTYPE=DTYPE.NP_UINT8)->Layer:
    '''
    Args:
    Returns:
    Raises:
    '''
    show_info(f'Init Bscan Preproc thread started')


    name = f'{img.name}_InitPreproc'
    layer_type = 'image'
    add_kwargs = {'name':f'{name}'}


    proc_data = init_bscan_preproc(img=img.data,num_std=num_std,min_intensity=min_intensity,max_intensity=max_intensity,dtype=dtype)


    layer = Layer.create(proc_data,add_kwargs,layer_type)


    show_info(f'Thread Name thread completed')
    return layer

def background_removal_plugin(img:Image):
    '''
    Args:
    Returns:
    Raises:
    '''
    background_removal_thread(img=img)
    return


@thread_worker(connect={"returned": viewer.add_layer})
def background_removal_thread(img:Image)->Image:
    '''
    Args:
    Returns:
    Raises:
    '''
    show_info(f'Background removal thread started')


    name = f'{img.name}_bg_corrected'
    layer_type = 'image'
    add_kwargs = {'name':f'{name}'}


    proc_data = background_removal_func(img=img.data)


    layer = Layer.create(proc_data,add_kwargs,layer_type)


    show_info(f'Background removal thread completed')
    return layer


def clahe(img:Image, kernel_size=None,clip_limit:float=0.01,nbins=256,norm_min=0,norm_max=1,pt_K:bool=True) -> Layer:
    ''''''
    clahe_thread(img=img,kernel_size=kernel_size,clip_limit=clip_limit,nbins=nbins,norm_min=norm_min,norm_max=norm_max,pt_K=pt_K)

    return

@thread_worker(connect={"returned": viewer.add_layer},progress=True)
def clahe_thread(img:Image, kernel_size=None,clip_limit:float=0.01,nbins=256,norm_min=0,norm_max=1,pt_K:bool=True) -> Layer:
    ''''''
    show_info(f'Autocontrast (CLAHE) thread has started')

    name = img.name

    # optional kwargs for viewer.add_* method
    add_kwargs = {"name": f"{name}_CLAHE"}

    # optional layer type argument
    layer_type = "image"

    if pt_K:
        output = clahe_pt_func(data=img.data,kernel_size=kernel_size,clip_limit=clip_limit,nbins=nbins,norm_min=norm_min,norm_max=norm_max)
        torch.cuda.empty_cache()
        memory_stats()
    else:
        output = clahe_func(data=img.data,kernel_size=kernel_size,clip_limit=clip_limit,nbins=nbins,norm_min=norm_min,norm_max=norm_max)

    layer = Layer.create(output,add_kwargs,layer_type)

    show_info(f'Autocontrast (CLAHE) thread has completed')
    return layer
    
def match_histogram(target_histogram:Image,debug:bool=False):
    """"""
    from skimage.exposure import match_histograms

    target_data = target_histogram.data
    current_selection = list(viewer.layers.selection)
    
    for layer in current_selection:
        matched = match_histograms(layer.data,target_data,channel_axis=-1)
        layer.data[:] = matched[:]
    return layer