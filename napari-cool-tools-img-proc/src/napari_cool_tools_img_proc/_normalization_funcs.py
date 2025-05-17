"""
This module contains function code for normalizing image values
"""
from napari.utils.notifications import show_info
from napari.layers import Image, Layer
from napari.types import ImageData
from napari_cool_tools_io import torch, device

def  standardize_data_func(img: ImageData) -> ImageData:
    """Function to standardize the image data with mean of 0 and achieve std of 1.0

        Args: img (ImageData): ndarray representing image data

        Returns: ImageData with standarized values with zero mean and standard deviation approx 1.0

    """

    standardized = (img - img.mean())/img.std()

    return standardized


def normalize_in_range_func(img: Image, min_val:float = 0.0, max_val:float = 1.0, in_place:bool = False) -> Layer:
    """Function to map image/B-scan values to a specific range between min_val and max_val.

    Args:
        img (Image): ndarray representing image data
        min_val (float): minimum value of range that image values are to be mapped to
        max_val (float): maximum value of range that image values are to be mapped to
        in_place (bool): flag indicating whether to modify the image in place or return new image

    Returns:
        Image with normalized values mapped between range of min_val and max_val is in_place
    """
    
    data = img.data
    norm_data = (max_val - min_val) * ((data-data.min())/ (data.max()-data.min())) + min_val

    if in_place:
        name = f"{img.name}_Norm_{min_val}-{max_val}"
        #new_name = f"pre_norm_{img.name}"
        #img.name = new_name
        add_kwargs = {"name":name}
        layer_type = 'image'
        layer = Layer.create(norm_data,add_kwargs,layer_type)
        return layer
    else:
        name = f"{img.name}_norm_{min_val}_{max_val}"
        add_kwargs = {"name":name}
        layer_type = "image"
        layer = Layer.create(norm_data,add_kwargs,layer_type)
        return layer
    
def normalize_in_range_pt_func(img: Image, min_val:float = 0.0, max_val:float = 1.0, in_place:bool = False) -> Layer:
    """Function to map image/B-scan values to a specific range between min_val and max_val.

    Args:
        img (Image): ndarray representing image data
        min_val (float): minimum value of range that image values are to be mapped to
        max_val (float): maximum value of range that image values are to be mapped to
        in_place (bool): flag indicating whether to modify the image in place or return new image

    Returns:
        Image with normalized values mapped between range of min_val and max_val is in_place
    """
    
    data = img.data.copy()
    pt_data = torch.tensor(data,device=device)
    norm_data = (max_val - min_val) * ((pt_data-pt_data.min())/ (pt_data.max()-pt_data.min())) + min_val

    if in_place:
        name = f"{img.name}_Norm_{min_val}-{max_val}"
        #new_name = f"pre_norm_{img.name}"
        #img.name = new_name
        add_kwargs = {"name":name}
        layer_type = 'image'
        layer = Layer.create(norm_data.detach().cpu().numpy(),add_kwargs,layer_type)
        return layer
    else:
        name = f"{img.name}_norm_{min_val}_{max_val}"
        add_kwargs = {"name":name}
        layer_type = "image"
        layer = Layer.create(norm_data.detach().cpu().numpy(),add_kwargs,layer_type)
        return layer

def normalize_data_in_range_func(img: ImageData, min_val:float = 0.0, max_val:float = 1.0) -> ImageData:
    """Function to map image/B-scan values to a specific range between min_val and max_val.

    Args:
        img (Image): ndarray representing image data
        min_val (float): minimum value of range that image values are to be mapped to
        max_val (float): maximum value of range that image values are to be mapped to
        numpy_out (bool): flag indicating whether to return torch tensor or numpy ndarray

    Returns:
        Image with normalized values mapped between range of min_val and max_val is in_place
    """
    
    data = img
    norm_data = (max_val - min_val) * ((data-data.min())/ (data.max()-data.min())) + min_val

    out = norm_data

    return out    
    
def normalize_data_in_range_pt_func(img: ImageData, min_val:float = 0.0, max_val:float = 1.0, numpy_out:bool = True, device='cpu') -> ImageData:
    """Function to map image/B-scan values to a specific range between min_val and max_val.

    Args:
        img (Image): ndarray representing image data
        min_val (float): minimum value of range that image values are to be mapped to
        max_val (float): maximum value of range that image values are to be mapped to
        numpy_out (bool): flag indicating whether to return torch tensor or numpy ndarray

    Returns:
        Image with normalized values mapped between range of min_val and max_val is in_place
    """

    pt_data = torch.tensor(img,device=device)
    norm_data = (max_val - min_val) * ((pt_data-pt_data.min())/ (pt_data.max()-pt_data.min())) + min_val

    if numpy_out:
        out = norm_data.detach().cpu().numpy()
    else:
        out = norm_data

    return out