"""
This module contains code for filtering images
"""
from tqdm import tqdm
from enum import Enum
from numpy import ndarray
from napari.utils.notifications import show_info
from napari.layers import Image, Layer
from napari.qt.threading import thread_worker
from napari_cool_tools_io import torch,viewer,device,memory_stats

def filter_bilateral(img:Image,kernel_size:int=1,s0:int=10,s1:int=10) -> Image:
    ''''''
    u_byte = img_as_ubyte(img)
    bilat = mean_bilateral(u_byte,disk(kernel_size),s0=s0,s1=s1)
    return bilat

def filter_median(img:Image,kernel_size:int=1,applications:int=1) -> Image:
    ''''''
    target = img
    for i in range(applications):
        img_med = median_filter(target,disk(kernel_size))
        target = img_med
    return img_med

def sharpen_um(img:Image, radius:float=1.0, amount:float=1.0, preserve_range=False, channel_axis=None) -> Image:
    ''''''
    sharp_img = unsharp_mask(img, radius=radius,amount=amount, preserve_range=preserve_range, channel_axis=channel_axis)
    return sharp_img

def filter_bilateral(img:Image,kernel_size:int=5,sc:float=0.1,s0:int=10,s1:int=10):
    """Implementation of bilateral filter function
    Args:
        img (Image): Image/Volume to be segmented.
        kernel_size (int): Dimension of symmetrical kernel for Kornia implementation kernel should be odd number
        sc (float): sigma_color Standard deviation for grayvalue/color distance (radiometric similarity). A larger value results in averaging of pixels with larger radiometric differences
        s0 (int): standard deviation of fist dimension of the kernel for range distance. A larger value results in averaging of pixels with larger spatial differences.
        s1 (int): standard deviation of the 2nd dimension of the kernel for range distance. A larger value results in averaging of pixels with larger spatial differences.
        
    Returns:
        Image Layer that has been bilaterally filtered  with '_Bilat_(kernel_size)' suffix added to name.
    """
    filter_bilateral_thread(img=img,kernel_size=kernel_size,sc=sc,s0=s0,s1=s1)
    return

@thread_worker(connect={"returned": viewer.add_layer},progress=True)
def filter_bilateral_thread(img:Image,kernel_size:int=5,sc:float=0.1,s0:int=10,s1:int=10) -> Image:
    """Implementation of bilateral filter function
    Args:
        img (Image): Image/Volume to be segmented.
        kernel_size (int): Dimension of symmetrical kernel for Kornia implementation kernel should be odd number
        sc (float): sigma_color Standard deviation for grayvalue/color distance (radiometric similarity). A larger value results in averaging of pixels with larger radiometric differences
        s0 (int): standard deviation of fist dimension of the kernel for range distance. A larger value results in averaging of pixels with larger spatial differences.
        s1 (int): standard deviation of the 2nd dimension of the kernel for range distance. A larger value results in averaging of pixels with larger spatial differences.
        
    Returns:
        Image Layer that has been bilaterally filtered  with '_Bilat_(kernel_size)' suffix added to name.
    """
    show_info(f'Bilateral Filter thread has started')
    output = filter_bilateral_pt_func(img=img,kernel_size=kernel_size,sc=sc,s0=s0,s1=s1)
    torch.cuda.empty_cache()
    memory_stats()
    show_info(f'Bilateral Filter thread has completed')
    
    return output


def filter_bilateral_pt_func(img:Image,kernel_size:int=5,sc:float=0.1,s0:int=10,s1:int=10,border_type:str='reflect',color_distance_type:str='l1') -> Image:
    """Implementation of bilateral filter function
    Args:
        img (Image): Image/Volume to be segmented.
        kernel_size (int): Dimension of symmetrical kernel for Kornia implementation kernel should be odd number
        sc (float): sigma_color Standard deviation for grayvalue/color distance (radiometric similarity). A larger value results in averaging of pixels with larger radiometric differences
        s0 (int): standard deviation of fist dimension of the kernel for range distance. A larger value results in averaging of pixels with larger spatial differences.
        s1 (int): standard deviation of the 2nd dimension of the kernel for range distance. A larger value results in averaging of pixels with larger spatial differences.
        
    Returns:
        Image Layer that has been bilaterally filtered  with '_Bilat_(kernel_size)' suffix added to name.
    """
    
    from kornia.filters import bilateral_blur
    
    name = img.name

    # optional kwargs for viewer.add_* method
    add_kwargs = {"name": f"{name}_Bilat_{kernel_size}"}

    # optional layer type argument
    layer_type = "image"

    data = img.data.copy()

    try:
        assert data.ndim == 2 or data.ndim == 3, "Only works for data of 2 or 3 dimensions"
    except AssertionError as e:
        print("An error Occured:", str(e))
    else:

        pt_data = torch.tensor(data,device=device)

        if data.ndim == 2:
            in_data = pt_data.unsqueeze(0).unsqueeze(0)
            blur_data = bilateral_blur(in_data,(kernel_size,kernel_size),sc,(s0,s1)).squeeze()
            out_data = blur_data.detach().cpu().numpy()
            layer = Layer.create(out_data,add_kwargs,layer_type)
        elif data.ndim == 3:
            for i in tqdm(range(len(pt_data)),desc="Bilateral Blur"):
                in_data = pt_data[i].unsqueeze(0).unsqueeze(0)
                pt_data[i] = bilateral_blur(in_data,(kernel_size,kernel_size),sc,(s0,s1)).squeeze()

            out_data = pt_data.detach().cpu().numpy()
            layer = Layer.create(out_data,add_kwargs,layer_type)

        return layer
    
def sharpen_um(img:Image,kernel_size:int=3,s0:int=10,s1:int=10):
    """Implementation of Unsharm Mask function
    Args:
        img (Image): Image/Volume to be segmented.
        kernel_size (int): Dimension of symmetrical kernel for Kornia implementation kernel should be odd number
        s0 (int): standard deviation of fist dimension of the kernel for range distance. A larger value results in averaging of pixels with larger spatial differences.
        s1 (int): standard deviation of the 2nd dimension of the kernel for range distance. A larger value results in averaging of pixels with larger spatial differences.
        
    Returns:
        Image Layer that has been sharpened  with '_UM_(kernel_size)' suffix added to name.
    """
    sharpen_um_thread(img=img,kernel_size=kernel_size,s0=s0,s1=s1)
    return

@thread_worker(connect={"returned": viewer.add_layer},progress=True)
def sharpen_um_thread(img:Image,kernel_size:int=3,s0:int=10,s1:int=10)-> Image:
    """Implementation of Unsharm Mask function
    Args:
        img (Image): Image/Volume to be segmented.
        kernel_size (int): Dimension of symmetrical kernel for Kornia implementation kernel should be odd number
        s0 (int): standard deviation of fist dimension of the kernel for range distance. A larger value results in averaging of pixels with larger spatial differences.
        s1 (int): standard deviation of the 2nd dimension of the kernel for range distance. A larger value results in averaging of pixels with larger spatial differences.
        
    Returns:
        Image Layer that has been sharpened  with '_UM_(kernel_size)' suffix added to name.
    """
    show_info(f'Unsharp Mask Filter thread has started')
    output = sharpen_um_pt_func(img=img,kernel_size=kernel_size,s0=s0,s1=s1)
    torch.cuda.empty_cache()
    memory_stats()
    show_info(f'Unsharp Mask Filter thread has completed')
    return output

def sharpen_um_pt_func(img:Image,kernel_size:int=3,s0:int=10,s1:int=10)-> Image:
    """Implementation of Unsharm Mask function
    Args:
        img (Image): Image/Volume to be segmented.
        kernel_size (int): Dimension of symmetrical kernel for Kornia implementation kernel should be odd number
        s0 (int): standard deviation of fist dimension of the kernel for range distance. A larger value results in averaging of pixels with larger spatial differences.
        s1 (int): standard deviation of the 2nd dimension of the kernel for range distance. A larger value results in averaging of pixels with larger spatial differences.
        
    Returns:
        Image Layer that has been sharpened  with '_UM_(kernel_size)' suffix added to name.
    """
    from kornia.filters import unsharp_mask
    
    name = img.name

    # optional kwargs for viewer.add_* method
    add_kwargs = {"name": f"{name}_UM_{kernel_size}"}

    # optional layer type argument
    layer_type = "image"

    data = img.data.copy()

    try:
        assert data.ndim == 2 or data.ndim == 3, "Only works for data of 2 or 3 dimensions"
    except AssertionError as e:
        print("An error Occured:", str(e))
    else:

        pt_data = torch.tensor(data,device=device)

        if data.ndim == 2:
            in_data = pt_data.unsqueeze(0).unsqueeze(0)
            um_data = unsharp_mask(in_data,(kernel_size,kernel_size),(s0,s1)).squeeze()
            out_data = um_data.detach().cpu().numpy()
            layer = Layer.create(out_data,add_kwargs,layer_type)
        elif data.ndim == 3:
            for i in tqdm(range(len(pt_data)),desc="Unsharp Mask"):
                in_data = pt_data[i].unsqueeze(0).unsqueeze(0)
                pt_data[i] = unsharp_mask(in_data,(kernel_size,kernel_size),(s0,s1)).squeeze()

            out_data = pt_data.detach().cpu().numpy()
            layer = Layer.create(out_data,add_kwargs,layer_type)

        return layer
    
def filter_median(img:Image,kernel_size:int=3):
    """Implementation of median filter function
    Args:
        img (Image): Image/Volume to be segmented.
        kernel_size (int): Dimension of symmetrical kernel for Kornia implementation kernel should be odd number
        
    Returns:
        Image Layer that has median blur  with '_Med_(kernel_size)' suffix added to name.
    """
    filter_median_thread(img=img,kernel_size=kernel_size)
    return

@thread_worker(connect={"returned": viewer.add_layer},progress=True)
def filter_median_thread(img:Image,kernel_size:int=3)-> Image:
    """Implementation of median filter function
    Args:
        img (Image): Image/Volume to be segmented.
        kernel_size (int): Dimension of symmetrical kernel for Kornia implementation kernel should be odd number
        
    Returns:
        Image Layer that has median blur  with '_Med_(kernel_size)' suffix added to name.
    """
    show_info(f'Median Filter thread has started')
    output = filter_median_pt_func(img=img,kernel_size=kernel_size)
    torch.cuda.empty_cache()
    memory_stats()
    show_info(f'Median Filter thread has completed')
    return output

def filter_median_pt_func(img:Image,kernel_size:int=3)-> Image:
    """Implementation of median filter function
    Args:
        img (Image): Image/Volume to be segmented.
        kernel_size (int): Dimension of symmetrical kernel for Kornia implementation kernel should be odd number
        
    Returns:
        Image Layer that has median blur  with '_Med_(kernel_size)' suffix added to name.
    """
    from kornia.filters import median_blur
    
    name = img.name

    # optional kwargs for viewer.add_* method
    add_kwargs = {"name": f"{name}_Med_{kernel_size}"}

    # optional layer type argument
    layer_type = "image"

    data = img.data.copy()

    try:
        assert data.ndim == 2 or data.ndim == 3, "Only works for data of 2 or 3 dimensions"
    except AssertionError as e:
        print("An error Occured:", str(e))
    else:

        pt_data = torch.tensor(data,device=device)

        if data.ndim == 2:
            in_data = pt_data.unsqueeze(0).unsqueeze(0)
            um_data = median_blur(in_data,(kernel_size,kernel_size)).squeeze()
            out_data = um_data.detach().cpu().numpy()
            layer = Layer.create(out_data,add_kwargs,layer_type)
        elif data.ndim == 3:
            for i in tqdm(range(len(pt_data)),desc="Median Filter"):
                in_data = pt_data[i].unsqueeze(0).unsqueeze(0)
                pt_data[i] = median_blur(in_data,(kernel_size,kernel_size)).squeeze()

            out_data = pt_data.detach().cpu().numpy()
            layer = Layer.create(out_data,add_kwargs,layer_type)

        return layer

class KnBorderType(Enum):
    """Enum for Kornia border_type parameter."""
    constant = 'constant'
    reflect = 'reflect'
    replicate = 'replicate'
    circular = 'circular'

def filter_gaussian_blur_plg(img:Image,kernel_size:int=3,sigma:float=1,border_type:KnBorderType=KnBorderType.reflect,separable:bool=True):
    """Implementation of Kornia's gausian blur filter function
    Args:
        img (Image): Image/Volume to be segmented.
        kernel_size (int): Dimension of symmetrical kernel for Kornia implementation kernel should be odd number
        sigma (int): standard deviation of the kernel
        border_type (KnBorderType(Enum)): padding mode applied prior to convolution options = 'constant', 'reflect', 'replicate' or 'circular'
        separable (bool): run as composition of 2 1D convolutions
        
    Returns:
        Image Layer that has gaussian blur  with '_GB_(kernel_size)' suffix added to name.
    """

    filter_gaussian_blur_thread(img=img,kernel_size=kernel_size,sigma=sigma,border_type=border_type.value,separable=separable)

    return

@thread_worker(connect={"returned": viewer.add_layer},progress=True)
def filter_gaussian_blur_thread(img:Image,kernel_size:int=3,sigma:float=1,border_type:str='reflect',separable:bool=True)->Image:
    """Implementation of Kornia's gausian blur filter function
    Args:
        img (Image): Image/Volume to be segmented.
        kernel_size (int): Dimension of symmetrical kernel for Kornia implementation kernel should be odd number
        sigma (int): standard deviation of the kernel
        border_type (KnBorderType(Enum)): padding mode applied prior to convolution options = 'constant', 'reflect', 'replicate' or 'circular'
        separable (bool): run as composition of 2 1D convolutions
        
    Returns:
        Image Layer that has gaussian blur  with '_GB_(kernel_size)' suffix added to name.
    """
    show_info(f'Gaussian Blur Filter thread has started')

    name = img.name

    # optional kwargs for viewer.add_* method
    add_kwargs = {"name": f"{name}_GBlur_{kernel_size}"}

    # optional layer type argument
    layer_type = "image"
    data = img.data.copy()
    out_data = filter_gaussian_blur_kn(data=data,kernel_size=kernel_size,sigma=sigma,border_type=border_type,separable=separable)
    output = Layer.create(out_data,add_kwargs,layer_type)

    torch.cuda.empty_cache()
    memory_stats()
    show_info(f'Gaussian Blur Filter thread has completed')

    return output


def filter_gaussian_blur_kn(data:ndarray,kernel_size:int=3,sigma:float=1.0,border_type:str='reflect',separable:bool=True)-> ndarray:
    """Implementation of Kornia's gausian blur filter function
    Args:
        img (Image): Image/Volume to be segmented.
        kernel_size (int): Dimension of symmetrical kernel for Kornia implementation kernel should be odd number
        sigma (float): standard deviation of the kernel
        border_type (KnBorderType(Enum)): padding mode applied prior to convolution options = 'constant', 'reflect', 'replicate' or 'circular'
        separable (bool): run as composition of 2 1D convolutions
        
    Returns:
        Image Layer that has gaussian blur  with '_GB_(kernel_size)' suffix added to name.
    """
    from kornia.filters import gaussian_blur2d
    
    try:
        assert data.ndim == 2 or data.ndim == 3, "Only works for data of 2 or 3 dimensions"
    except AssertionError as e:
        print("An error Occured:", str(e))
    else:

        pt_data = torch.tensor(data,device=device)

        if data.ndim == 2:
            in_data = pt_data.unsqueeze(0).unsqueeze(0)
            blur_data = gaussian_blur2d(in_data,(kernel_size,kernel_size),(sigma,sigma),border_type,separable).squeeze()
            out_data = blur_data.detach().cpu().numpy()
            
        elif data.ndim == 3:
            for i in tqdm(range(len(pt_data)),desc="Gaussian Blur Filter"):
                in_data = pt_data[i].unsqueeze(0).unsqueeze(0)
                pt_data[i] = gaussian_blur2d(in_data,(kernel_size,kernel_size),(sigma,sigma),border_type,separable).squeeze()

            out_data = pt_data.detach().cpu().numpy()

        return out_data
        

        '''''
        name = img.name

        # optional kwargs for viewer.add_* method
        add_kwargs = {"name": f"{name}_Med_{kernel_size}"}

        # optional layer type argument
        layer_type = "image"

        data = img.data.copy()

        layer = Layer.create(out_data,add_kwargs,layer_type)
        '''