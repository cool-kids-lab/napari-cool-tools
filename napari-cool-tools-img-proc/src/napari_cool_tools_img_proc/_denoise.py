"""
This module contains code for denoising images
"""
from napari.utils.notifications import show_info
from napari.layers import Image, Layer
from napari.types import ImageData
from napari.qt.threading import thread_worker
from napari_cool_tools_io import torch,viewer,device,memory_stats
from napari_cool_tools_img_proc._denoise_funcs import diff_of_gaus, denoise_tv


def diff_of_gaus_plugin(img:Image, low_sigma:float=1.0, high_sigma:float=20.0, mode='nearest',cval=0, channel_axis=None, truncate=4.0, pt=False) -> Layer:
    """Implementation of median filter function
    Args:
        img (Image): Image/Volume to be segmented.
        low_sigma (float): standard deviation for lower intensity gaussian filter
        high_sigma (float): standard deviation for higher intensity gaussian filter
        mode (str): how input array is extended when filter overlaps border 
                    reflect, constant, nearest, mirror, wrap, grid-constant, grid-mirror, grid-wrap
                    for option descriptions refer to https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.gaussian_filter.html
        cval (int): value to fill past edges in "constant" mode
        channel_axis (int or none): optional if None image assumed to be grayscale otherwise indicates axis that denotes color channels
        truncate (float): number of standard deviations to filter 
        pt (bool): flag indicatiing whether to use pytorch implementation
        
    Returns:
        Image Layer that has had difference of gaussians applied to it  with '_Band-pass' suffix added to name.
    """

    diff_of_gaus_thread(img=img,low_sigma=low_sigma,high_sigma=high_sigma,mode=mode,cval=cval,channel_axis=channel_axis,truncate=truncate,pt=pt)

@thread_worker(connect={"returned": viewer.add_layer},progress=True)
def diff_of_gaus_thread(img:Image, low_sigma, high_sigma=None, mode='nearest',cval=0, channel_axis=None, truncate=4.0, pt=False) -> Layer:
    """Implementation of median filter function
    Args:
        img (Image): Image/Volume to be segmented.
        low_sigma (float): standard deviation for lower intensity gaussian filter
        high_sigma (float): standard deviation for higher intensity gaussian filter
        mode (str): how input array is extended when filter overlaps border 
                    reflect, constant, nearest, mirror, wrap, grid-constant, grid-mirror, grid-wrap
                    for option descriptions refer to https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.gaussian_filter.html
        cval (int): value to fill past edges in "constant" mode
        channel_axis (int or none): optional if None image assumed to be grayscale otherwise indicates axis that denotes color channels
        truncate (float): number of standard deviations to filter
        pt (bool): flag indicatiing whether to use pytorch implementation
        
    Returns:
        Image Layer that has had difference of gaussians applied to it  with '_Band-pass' suffix added to name.
    """
    show_info("Difference of Gaussian thread has started")
    name = f"{img.name}_DoG"
    add_kwargs = {"name":name}
    layer_type = 'image'
    out_data = diff_of_gaus(data=img.data,low_sigma=low_sigma,high_sigma=high_sigma,mode=mode,cval=cval,channel_axis=channel_axis,truncate=truncate,pt=pt)
    layer = Layer.create(out_data,add_kwargs,layer_type)
    torch.cuda.empty_cache()
    memory_stats()
    show_info("Difference of Gaussian thread has completed")
    return layer
    
def denoise_tv_plugin(img:Image, weight:float=0.1) -> Layer:
    ''''''
    denoise_tv_thread(img=img,weight=weight)
    return

@thread_worker(connect={"returned": viewer.add_layer},progress=True)
def denoise_tv_thread(img:Image, weight:float=0.1) -> Layer:
    ''''''
    show_info(f'Denoise Total Variation thread has started')
    denoise_data = denoise_tv(data=img.data,weight=weight)
    print("\n\nWe MADE IT HERE!!\n\n")
    name = f"{img.name}_TV"
    add_kwargs = {"name":f"{name}"}
    layer_type = 'image'
    layer = Layer.create(denoise_data,add_kwargs,layer_type)
    show_info(f'Denoise Total Variation thread has completed')
    return layer