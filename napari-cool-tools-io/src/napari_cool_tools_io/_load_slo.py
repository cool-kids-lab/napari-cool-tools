"""
This module contains code for loading .prof files wihout .xml metadata
"""
from pathlib import Path
from typing import Literal,Tuple,List
from enum import Enum

import numpy as np
from magicgui import magic_factory
from napari.layers import Layer


@magic_factory()
def load_slo(
    path=Path("."),
    overide_HxW_dimensions:bool=False,
    h=1250,
    w=1250,
    align_ascans=True,
    #temp_fix=True,
    #improve_reg=True,
    subpixel_reg=True,
    stack:bool=True,
    shift_direction:Literal[-1,1] = -1,
    scale_factor:Tuple[float,float]=(1.0,0.25),
    order:Literal["SubpixReg-Scale","Scale-SubpixReg"] = "SubpixReg-Scale",
    verbose:bool=False,
) -> Layer:  # LayerDataTuple:
    """ """
    import os.path as ospath
    import xml.etree.ElementTree as ET
    import torch
    from torchvision.transforms import v2,InterpolationMode
    from scipy.ndimage import fourier_shift

    ## define chunks as little endian f32 4 byte floats with HEIGHT values
    ## per row and WIDTH values per column
    # dot_prof = np.dtype(('<f4', (HEIGHT, WIDTH)))

    ## generate numpy array by loading 400 * 496 * f32 sized data chunks
    ## and stacking them until end of file is reached
    # b_scan = np.fromfile(path, dtype=dot_prof, count=-1)
    #
    ## transpose array so that x and y are switched then flip array
    ## to better orient b-scans for manual segmentation
    # display = np.flip(b_scan.transpose(0,2,1), 1)

    # temp = get_reader(path)

    # display = temp(path)

    # if subpixel:
    #     improve_reg = True
    if verbose:
        print(f"path:{path}, width: {w}, height: {h}")

    # isolate file name from path and .prof extension
    # file_name = ospath.basename(path)
    head, tail = ospath.split(path)
    file_name = tail.replace(".", "_")

    if verbose:
        print(f"layer_name: {file_name}\n")

    # # define chunks as little endian f32 4 byte floats with HEIGHT values
    # # per row and WIDTH values per column
    # mip_z = np.dtype(("<f8", (h, w)))  # saved as double precision f64 8 byte
    # # mip_z = np.dtype(('<f4', (h, w)))

    # # generate numpy array by loading h * w * f32 sized data chunks
    # # and stacking them until end of file is reached
    # enface = np.fromfile(path, dtype=mip_z, count=-1)

    # get metadata
    path_parent = path.parent
    path_stem = path.stem
    xml_path = path_parent / f"{path_stem}.xml"

    # load and correct files
    if verbose:
        print(f"getting meta data from {xml_path}.")
    slo_tree = ET.parse(xml_path)
    root = slo_tree.getroot()
    xml_image_size = root[0].find('Image_Size')
    height = int(xml_image_size.get("Width"))
    width = int(xml_image_size.get("Height"))
    if verbose:
        print(f"WxH: {width}x{height}")

    if overide_HxW_dimensions:
        height = h
        width = w

    # define chunks as little endian f32 4 byte floats with HEIGHT values
    # per row and WIDTH values per column
    mip_z = np.dtype(("<f8", (height, width)))  # saved as double precision f64 8 byte
    enface = np.fromfile(path, dtype=mip_z, count=-1)

    # display = enface
    # display = enface.transpose(0,2,1)

    if order == "Scale-SubpixReg":
        if scale_factor[0] != 1.0 or scale_factor[1] != 1.0:
            scale_factor_t = torch.Tensor(scale_factor)
            display_shape_t = torch.Tensor(enface.shape[1:])
            new_shape_t = (scale_factor_t * display_shape_t).to(torch.uint32)
            #new_shape = new_shape_t.round().to(torch.uint32).numpy().astype(np.uint32)
            if verbose:
                print(f"{scale_factor_t} x {display_shape_t} = {new_shape_t}") #: {new_shape}")
            # new_size = torch.Tensor(scale_factor)*torch.Tensor(display.shape[1:]).round().numpy().astype(np.uint8)
            # print(f"new size: {new_size}")
            enface = v2.functional.resize(torch.from_numpy(enface),new_shape_t,InterpolationMode.BILINEAR).numpy()

            h,w = enface.shape[-2],enface.shape[-1]


    if align_ascans is True:
        display = np.empty_like(enface)
        display[:, ::2, :] = enface[:, ::2, :]
        display[:, 1::2, :] = np.flip(enface[:, 1::2, :], 2)

        # convert from sinusoidal space to linear space
        Xn = np.arange(width)
        x_org = (width / 2) * np.sin(2 * np.pi / (2 * width) * Xn - np.pi / 2) + (
            width / 2
        )

        interp_sin_lin = np.empty_like(enface)

        if verbose:
            print(f"Xn: {Xn}\nx_org:{x_org}\n")

        d = enface.shape[0]

        for i in range(d):
            for j in range(height):
                interp_sin_lin[i, j, :] = np.interp(
                    Xn, x_org, display[i, j, :]
                )

        display[:] = interp_sin_lin[:]

    else:
        display = enface

    if subpixel_reg is True:
        from skimage.registration import phase_cross_correlation

        even = display[:, ::2, :]
        odd = display[:, 1::2, :]

        shift, error, diffphase = phase_cross_correlation(
            even, odd, upsample_factor=100
        )

        input_ = np.fft.fft2(odd)
        result = fourier_shift(input_, (0.0, 0.0, shift_direction*shift[2]), axis=2)
        result = np.fft.ifft2(result)
        odd_shift = result.real

        if verbose:
            print(f"shift: {shift}\n")

        registered = np.empty_like(display)
        registered[:, ::2, :] = even
        registered[:, 1::2, :] = odd_shift

        display = registered
    if verbose:
        print(display.shape[1:])

    if order == "SubpixReg-Scale":
        if scale_factor[0] != 1.0 or scale_factor[1] != 1.0:
            scale_factor_t = torch.Tensor(scale_factor)
            display_shape_t = torch.Tensor(display.shape[1:])
            new_shape_t = (scale_factor_t * display_shape_t).to(torch.uint32)
            #new_shape = new_shape_t.round().to(torch.uint32).numpy().astype(np.uint32)
            if verbose:
                print(f"{scale_factor_t} x {display_shape_t} = {new_shape_t}") #: {new_shape}")
            # new_size = torch.Tensor(scale_factor)*torch.Tensor(display.shape[1:]).round().numpy().astype(np.uint8)
            # print(f"new size: {new_size}")
            display = v2.functional.resize(torch.from_numpy(display),new_shape_t,InterpolationMode.BILINEAR).numpy()

    # stack files
    if stack:
        _,scaled_height,scaled_width = display.shape
        if verbose:
            print(f"shape before: {display.shape}")
        display = display.reshape(-1,scaled_height//2,scaled_width)
        if verbose:
            print(f"shape after: {display.shape}")

    # optional kwargs for viewer.add_* method
    add_kwargs = {"name": file_name}

    # optional layer type argument
    layer_type = "image"

    display = display.squeeze()

    if verbose:
        print(f"Final data shape: {display.shape}")

    layer = Layer.create(display, add_kwargs, layer_type)

    # layer_data_tuples = [(display,add_kwargs,layer_type)]

    # add layers from layer data tuples
    # for ldt in layer_data_tuples:
    #    viewer._add_layer_from_data(*ldt)

    # order in is Row, Column, Depth so post load must reorder to Depth, Row, Column for proper numpy display

    return layer
