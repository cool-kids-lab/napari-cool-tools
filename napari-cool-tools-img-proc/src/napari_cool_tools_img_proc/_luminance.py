"""
This module contains code for adjusting image luminance
"""

from tqdm import tqdm
from napari.utils.notifications import show_info
from napari.layers import Image, Layer
from napari.qt.threading import thread_worker
from napari_cool_tools_io import torch,viewer,memory_stats
from napari_cool_tools_img_proc._luminance_funcs import adjust_log_func, adjust_log_pt_func

def adjust_gamma(img:Image, gamma:float=1, gain:float=1) -> Layer:
    """Pass through function of skimage.exposure adjust_log function.
    
    Args:
        img (Image): Image to be adjusted.
        gamma(float): Non negative real number.
        gain (float): Constant multiplier.
        
    Returns:
        Gamma corrected output image with '_LC' suffix added to name."""
    
    adjust_gamma_thread(img=img,gamma=gamma,gain=gain)
    return

@thread_worker(connect={"returned": viewer.add_layer},progress=True)
def adjust_gamma_thread(img:Image, gamma:float=1, gain:float=1) -> Layer:
    """Pass through function of skimage.exposure adjust_log function.
    
    Args:
        img (Image): Image to be adjusted.
        gamma(float): Non negative real number.
        gain (float): Constant multiplier.
        
    Returns:
        Gamma corrected output image with '_LC' suffix added to name."""
    
    show_info(f"Adjust gamma thread started")
    output = adjust_gamma_func(img=img,gamma=gamma,gain=gain)
    show_info(f"Adjust gamma thread completed")
    return output

def adjust_gamma_func(img:Image, gamma:float=1, gain:float=1) -> Layer:
    """Pass through function of skimage.exposure adjust_log function.
    
    Args:
        img (Image): Image to be adjusted.
        gamma(float): Non negative real number.
        gain (float): Constant multiplier.
        
    Returns:
        Gamma corrected output image with '_LC' suffix added to name."""
    
    from skimage.exposure import adjust_gamma

    data = img.data.copy()

    try:
        assert data.ndim == 2 or data.ndim == 3, "Only works for data of 2 or 3 diminsions"
    except AssertionError as e:
        print("An error Occured:", str(e))
    else:
        
        name = f"{img.name}_GC"
        layer_type = "image"
        add_kwargs = {"name": f"{name}"}

        if data.ndim == 2:
            log_corrected = adjust_gamma(data,gamma=gamma,gain=gain)
            layer = Layer.create(log_corrected,add_kwargs,layer_type)
        elif data.ndim == 3:
            for i in tqdm(range(len(data)),desc="Gamma Correction"):
                data[i] = adjust_gamma(data[i],gamma=gamma,gain=gain)

            layer = Layer.create(data,add_kwargs,layer_type)

    return layer    

def adjust_log(img:Image, gain:float=1, inv:bool=False, pt_K:bool=True) -> Layer:
    """Pass through function of skimage.exposure adjust_log function.
    
    Args:
        img (Image): Image to be adjusted.
        gain (float): Constant multiplier.
        inv (bool): If True performs inverse log correction instead of log correction.
        gpu (bool): If True attempts to use pytorch gpu version of function
        
    Returns:
        Logarithm corrected output image with '_LC' suffix added to name."""
    
    adjust_log_thread(img=img,gain=gain,inv=inv,pt_K=pt_K)
    #return

@thread_worker(connect={"returned": viewer.add_layer},progress=True)
def adjust_log_thread(img:Image, gain:float=1, inv:bool=False, pt_K:bool=True) -> Layer:
    """Pass through function of skimage.exposure adjust_log function.
    
    Args:
        img (Image): Image to be adjusted.
        gain (float): Constant multiplier.
        inv (bool): If True performs inverse log correction instead of log correction.
        gpu (bool): If True attempts to use pytorch gpu version of function
        
    Returns:
        Logarithm corrected output image with '_LC' suffix added to name."""
    
    show_info(f"Adjust log thread started")

    name = f"{img.name}_LC"
    layer_type = "image"
    add_kwargs = {"name": f"{name}"}

    if pt_K:
        output = adjust_log_pt_func(data=img.data,gain=gain,inv=inv)
        torch.cuda.empty_cache()
        memory_stats()
    else:
        output = adjust_log_func(data=img.data,gain=gain,inv=inv)

    layer = Layer.create(output,add_kwargs,layer_type)

    show_info(f"Adjust log thread completed")
    return layer