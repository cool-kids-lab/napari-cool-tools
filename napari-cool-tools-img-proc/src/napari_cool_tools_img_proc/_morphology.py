"""
COOL Tool Image Morphology plugins
"""
from typing import List, Tuple, Generator
from napari.utils.notifications import show_info
from napari.qt.threading import thread_worker
from napari.layers import Image, Layer
from napari_cool_tools_io import viewer
from napari_cool_tools_img_proc._morphology_funcs import morphological_dilation, morphological_erosion
from napari_cool_tools_io import memory_stats


def morphological_dilation_plugin(lyr:Layer,kernel_size:int=3,custom_kenel:bool=False,volumetric_calc:bool=False,use_gpu:bool=False):
    '''
    Args:
    Returns:
    Raises:
    '''
    morphological_dilation_thread(lyr=lyr,kernel_size=kernel_size,custom_kenel=custom_kenel,volumetric_calc=volumetric_calc,use_gpu=use_gpu)
    return


@thread_worker(connect={"returned": viewer.add_layer})
def morphological_dilation_thread(lyr:Layer,kernel_size:int=3,custom_kenel:bool=False,volumetric_calc:bool=False,use_gpu:bool=False)->Layer:
    '''
    Args:
    Returns:
    Raises:
    '''
    show_info(f'Dilation thread started')


    name = f'{lyr.name}_Dilated'
    layer_type = lyr.as_layer_data_tuple()[2]
    add_kwargs = {'name':f'{name}'}


    proc_data = morphological_dilation(data=lyr.data,kernel_size=kernel_size,custom_kenel=custom_kenel,volumetric_calc=volumetric_calc,use_gpu=use_gpu)


    layer = Layer.create(proc_data,add_kwargs,layer_type)
    memory_stats()

    show_info(f'Dilation thread completed')
    return layer


def morphological_erosion_plugin(lyr:Layer,kernel_size:int=3,custom_kenel:bool=False,volumetric_calc:bool=False,use_gpu:bool=False):
    '''
    Args:
    Returns:
    Raises:
    '''
    morphological_erosion_thread(lyr=lyr,kernel_size=kernel_size,custom_kenel=custom_kenel,volumetric_calc=volumetric_calc,use_gpu=use_gpu)
    return


@thread_worker(connect={"returned": viewer.add_layer})
def morphological_erosion_thread(lyr:Layer,kernel_size:int=3,custom_kenel:bool=False,volumetric_calc:bool=False,use_gpu:bool=False)->Layer:
    '''
    Args:
    Returns:
    Raises:
    '''
    show_info(f'Erosion thread started')


    name = f'{lyr.name}_Dilated'
    layer_type = lyr.as_layer_data_tuple()[2]
    add_kwargs = {'name':f'{name}'}


    proc_data = morphological_erosion(data=lyr.data,kernel_size=kernel_size,custom_kenel=custom_kenel,volumetric_calc=volumetric_calc,use_gpu=use_gpu)


    layer = Layer.create(proc_data,add_kwargs,layer_type)
    memory_stats()

    show_info(f'Erosion thread completed')
    return layer