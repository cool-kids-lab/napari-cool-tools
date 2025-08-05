"""
This module contains code for 2D neural network visualization tools
"""

from enum import Enum

import numpy as np
import torch
from napari.layers import Image, Layer
from napari.qt.threading import thread_worker
from napari.utils.notifications import show_info
from napari_cool_tools_io import viewer
from numpy import ndarray


class NpBorderType(Enum):
    """Enum for Numpy border_type parameter."""

    constant = "constant"
    edge = "edge"
    linear_ramp = "linear_ramp"
    maximum = "maximum"
    mean = "mean"
    median = "median"
    minimum = "minimum"
    reflect = "reflect"
    symmetric = "symmetric"
    wrap = "wrap"
    empty = "empty"


def pad_image2D_plg(
    img: Image,
    axis0_before: int = 12,
    axis0_after: int = 12,
    axis1_before: int = 0,
    axis1_after: int = 0,
    mode: NpBorderType = NpBorderType.constant,
):
    """"""
    pad_image2D_thread(
        img=img,
        axis0_before=axis0_before,
        axis0_after=axis0_after,
        axis1_before=axis1_before,
        axis1_after=axis1_after,
        mode=mode.value,
    )

    return


@thread_worker(connect={"returned": viewer.add_layer}, progress=True)
def pad_image2D_thread(
    img: Image,
    axis0_before: int = 12,
    axis0_after: int = 12,
    axis1_before: int = 0,
    axis1_after: int = 0,
    mode: str = "constant",
) -> Image:
    """"""
    show_info("Pad 2D thread has started")

    name = img.name

    # optional kwargs for viewer.add_* method
    add_kwargs = {
        "name": f"{name}_Pad_{axis0_before}-{axis0_after}_{axis1_before}-{axis1_after}"
    }

    # optional layer type argument
    layer_type = "image"
    data = img.data.copy()
    out_data = pad_image2D_np(
        data=data,
        axis0_before=axis0_before,
        axis0_after=axis0_after,
        axis1_before=axis1_before,
        axis1_after=axis1_after,
        mode=mode,
    )
    output = Layer.create(out_data, add_kwargs, layer_type)

    show_info("Pad 2D thread has completed")

    return output


def pad_image2D_np(
    data: ndarray,
    axis0_before: int = 12,
    axis0_after: int = 12,
    axis1_before: int = 0,
    axis1_after: int = 0,
    mode: str = "constant",
) -> ndarray:
    """"""
    pad_width = ((axis0_before, axis0_after), (axis1_before, axis1_after))
    out_data = np.pad(data, pad_width, mode)

    return out_data


class NpPoolType(Enum):
    """Enum for Numpy pool_type parameter."""

    max = "max"
    avg = "avg"


def pool_2D_plg(
    img: Image, block_size: int = 2, pooling: NpPoolType = NpPoolType.max
) -> Image:
    """"""
    pool_2D_thread(img=img, pooling=pooling)

    return


@thread_worker(connect={"returned": viewer.add_layer}, progress=True)
def pool_2D_thread(
    img: Image, block_size: int = 2, pooling: NpPoolType = NpPoolType.max
) -> Image:
    """"""

    show_info("Pooling 2D thread has started")
    name = img.name

    # optional kwargs for viewer.add_* method
    add_kwargs = {"name": f"{name}_{pooling.name}_Pool"}

    # optional layer type argument
    layer_type = "image"
    data = img.data.copy()
    out_data = pool_2D(data=data, block_size=block_size, pooling=pooling)
    output = Layer.create(out_data, add_kwargs, layer_type)
    show_info("Pooling 2D thread has completed")

    return output


def pool_2D(
    data: ndarray, block_size: int = 2, pooling: NpPoolType = NpPoolType.max
) -> ndarray:
    """"""
    from skimage.measure import block_reduce

    if pooling.value == "max":
        pool_func = np.max
    elif pooling.value == "avg":
        pool_func = np.mean
    out_data = block_reduce(data, block_size=block_size, func=pool_func)
    return out_data

def U_mask(data:ndarray)->ndarray:
    """"""
    from jj_nn_framework.yakub_complex_conjugate_unet import UMask
    from torchvision.transforms.v2 import Resize
    
    u_mask = UMask(in_channels=1,out_channels=1)
    resize_down = Resize((512,1024))
    
    t_data = torch.tensor(data.copy()).permute(-3,-1,-2)
    resize_up = Resize((t_data.shape[-2],t_data.shape[-1]))
    t_data =  resize_down(t_data)
    t_sample = t_data[len(t_data)//2]

    with torch.no_grad():
        t_out = u_mask(t_sample.unsqueeze(0).unsqueeze(0)).detach()
    
    t_out = resize_up(t_out)
    t_out = t_out.squeeze().permute(-1,-2).numpy()
    return t_out
