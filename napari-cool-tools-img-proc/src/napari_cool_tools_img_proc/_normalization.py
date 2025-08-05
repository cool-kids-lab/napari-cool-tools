"""
This module contains code for normalizing image values
"""

# import torch
import numpy as np
from napari.layers import Image, Layer
from napari.qt.threading import thread_worker
from napari.types import ImageData
from napari.utils.notifications import show_info
from napari_cool_tools_io import device, torch, viewer

from napari_cool_tools_img_proc import DType
from napari_cool_tools_img_proc._normalization_funcs import (
    standardize_data_func,
    normalize_data_in_range_func,
)


def convert_dtype_plugin(img: Image,datatype:DType=DType.NP_FLOAT32,debug=False):
    """"""
    convert_dtype_thread(img=img,datatype=datatype,debug=debug)

    return


@thread_worker(connect={"returned": viewer.add_layer})
def convert_dtype_thread(img: Image,datatype:DType=DType.NP_FLOAT32,debug=False):
    """"""

    show_info("Type conversion thread started")
    name = img.name
    add_kwargs = {"name": f"{name}_std"}
    layer_type = "image"

    min_val = 0.0
    max_val = 1.0

    if debug:
        print(f"datatype/value: {datatype,datatype.value}")

    match datatype:
        case DType.NP_FLOAT64 | DType.NP_FLOAT32 | DType.NP_FLOAT16:
            pass #max_val = 1.0
        case DType.NP_UINT8:
            max_val = 255.0

    if debug:
        print(f"max val: {max_val}")
    
    new_type_data = normalize_data_in_range_func(img.data.astype(DType.NP_FLOAT64.value),min_val=min_val,max_val=max_val).astype(datatype.value)

    layer = Layer.create(new_type_data, add_kwargs, layer_type)

    show_info("Type conversion thread ended")

    return layer

def standardize_image_plugin(img: Image):
    """Function to standardize the image data with mean of 0 and achieve std of 1.0

    Args: img (Image): ndarray representing image data

    Returns: Image with standarized values with zero mean and standard deviation approx 1.0

    Raises: NotImplementedError for any datatype that is not either a floating point type or an unsigned integer

    """

    if (
        #img.data.dtype != np.dtype(np.uint8)
        img.data.dtype != np.dtype(np.float16)
        and img.data.dtype != np.dtype(np.float32)
        and img.data.dtype != img.data(np.float64)
    ):
        raise NotImplementedError(
            f"This function does not support {img.data.dtype}, uint8, float16, float32, float64, and float128 are supported datatypes"
        )

    standardize_img_thread(img)

    return


@thread_worker(connect={"returned": viewer.add_layer})
def standardize_img_thread(img: Image) -> Layer:
    """Function to standardize the image data with mean of 0 and achieve std of 1.0

    Args: img (Image): ndarray representing image data

    Returns: Image with standarized values with zero mean and standard deviation approx 1.0

    """
    show_info("Standardization thread started")
    name = img.name
    add_kwargs = {"name": f"{name}_std"}
    layer_type = "image"

    #input_type = img.data.dtype

    #if input_type != np.dtype(np.uint8):
    data = img.data
    std_data = standardize_data_func(data)
    # else:
    #     data = normalize_data_in_range_func(
    #         img.data.astype(np.float64), min_val=0.0, max_val=1.0
    #     )
    #     std_data = standardize_data_func(data)
    #     std_data = normalize_data_in_range_func(
    #         std_data, min_val=0.0, max_val=255.0
    #     ).astype(input_type)

    layer = Layer.create(std_data, add_kwargs, layer_type)

    show_info("Standardization thread ended")

    return layer


def normalize_in_range(
    img: Image, min_val: float = 0.0, max_val: float = 1.0, in_place: bool = False
) -> Layer:
    """Function to map image/B-scan values to a specific range between min_val and max_val.

    Args:
        img (Image): ndarray representing image data
        min_val (float): minimum value of range that image values are to be mapped to
        max_val (float): maximum value of range that image values are to be mapped to
        in_place (bool): flag indicating whether to modify the image in place or return new image

    Returns:
        Image with normalized values mapped between range of min_val and max_val is in_place

    Raises: NotImplementedError for any datatype that is not either a floating point type or an unsigned integer

    """

    if (
        img.data.dtype != np.dtype(np.uint8)
        and img.data.dtype != np.dtype(np.float16)
        and img.data.dtype != np.dtype(np.float32)
        and img.data.dtype != img.data(np.float64)
    ):
        raise NotImplementedError(
            f"This function does not support {img.data.dtype}, uint8, float16, float32, float64, and float128 are supported datatypes"
        )

    normalize_in_range_thread(
        img=img, min_val=min_val, max_val=max_val, in_place=in_place
    )
    return


@thread_worker(connect={"returned": viewer.add_layer}, progress=True)
def normalize_in_range_thread(
    img: Image, min_val: float = 0.0, max_val: float = 1.0, in_place: bool = False
) -> Layer:
    """Function to map image/B-scan values to a specific range between min_val and max_val.

    Args:
        img (Image): ndarray representing image data
        min_val (float): minimum value of range that image values are to be mapped to
        max_val (float): maximum value of range that image values are to be mapped to
        in_place (bool): flag indicating whether to modify the image in place or return new image

    Returns:
        Image with normalized values mapped between range of min_val and max_val is in_place
    """
    show_info("Normalization thread started")

    name = img.name
    add_kwargs = {"name": f"{name}_std"}
    layer_type = "image"

    #input_type = img.data.dtype

    #convert_flag = False

    #if input_type != np.dtype(np.uint8):
    #data = img.data
    # else:
    #     data = img.data.astype(np.float64)
    #     convert_flag = True

    nrm_data = normalize_data_in_range_func(img.data,min_val=min_val,max_val=max_val)
    
    # if convert_flag:
    #     nrm_data = nrm_data.astype(input_type)

    layer = Layer.create(nrm_data, add_kwargs, layer_type)

    return layer


def normalize_in_range_func_old(
    img: Image, min_val: float = 0.0, max_val: float = 1.0, in_place: bool = True
) -> Layer:
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
    norm_data = (max_val - min_val) * (
        (data - data.min()) / (data.max() - data.min())
    ) + min_val

    if in_place:
        name = f"{img.name}_Norm_{min_val}-{max_val}"
        # new_name = f"pre_norm_{img.name}"
        # img.name = new_name
        add_kwargs = {"name": name}
        layer_type = "image"
        layer = Layer.create(norm_data, add_kwargs, layer_type)
        return layer
    else:
        name = f"{img.name}_norm_{min_val}_{max_val}"
        add_kwargs = {"name": name}
        layer_type = "image"
        layer = Layer.create(norm_data, add_kwargs, layer_type)
        return layer


def normalize_in_range_pt_func_old(
    img: Image, min_val: float = 0.0, max_val: float = 1.0, in_place: bool = True
) -> Layer:
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
    pt_data = torch.tensor(data, device=device)
    norm_data = (max_val - min_val) * (
        (pt_data - pt_data.min()) / (pt_data.max() - pt_data.min())
    ) + min_val

    if in_place:
        name = f"{img.name}_Norm_{min_val}-{max_val}"
        # new_name = f"pre_norm_{img.name}"
        # img.name = new_name
        add_kwargs = {"name": name}
        layer_type = "image"
        layer = Layer.create(norm_data.detach().cpu().numpy(), add_kwargs, layer_type)
        return layer
    else:
        name = f"{img.name}_norm_{min_val}_{max_val}"
        add_kwargs = {"name": name}
        layer_type = "image"
        layer = Layer.create(norm_data.detach().cpu().numpy(), add_kwargs, layer_type)
        return layer


def normalize_data_in_range_func_old(
    img: ImageData, min_val: float = 0.0, max_val: float = 1.0
) -> ImageData:
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
    norm_data = (max_val - min_val) * (
        (data - data.min()) / (data.max() - data.min())
    ) + min_val

    out = norm_data

    return out


def normalize_data_in_range_pt_func_old(
    img: ImageData, min_val: float = 0.0, max_val: float = 1.0, numpy_out: bool = True
) -> ImageData:
    """Function to map image/B-scan values to a specific range between min_val and max_val.

    Args:
        img (Image): ndarray representing image data
        min_val (float): minimum value of range that image values are to be mapped to
        max_val (float): maximum value of range that image values are to be mapped to
        numpy_out (bool): flag indicating whether to return torch tensor or numpy ndarray

    Returns:
        Image with normalized values mapped between range of min_val and max_val is in_place
    """

    data = img.copy()
    pt_data = torch.tensor(data, device=device)
    norm_data = (max_val - min_val) * (
        (pt_data - pt_data.min()) / (pt_data.max() - pt_data.min())
    ) + min_val

    if numpy_out:
        out = norm_data.detach().cpu().numpy()
    else:
        out = norm_data

    return out
