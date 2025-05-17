"""Code for importing and exporting napari labels data in a viewable
format for other programs"""

from enum import Enum

import numpy as np
from napari.layers import Image, Labels, Layer
from napari.utils.notifications import show_info


class OutTypes(Enum):
    uint8 = "uint8"
    uint16 = "uint16"
    uint32 = "uint32"
    uint64 = "uint64"


def convert_labels_tiff(
    vol: Labels, dtype: OutTypes = OutTypes.uint8, debug: bool = False
) -> Image:
    """"""
    data = vol.data
    name = f"{vol.name}_{dtype.value}"
    add_kwargs = {"name": name}
    layer_type = "image"

    # case uint8
    if dtype == OutTypes.uint8:
        temp_data = data.astype(np.uint8)
    # case uint16
    elif dtype == OutTypes.uint16:
        temp_data = data.astype(np.uint16)
    # case uint32
    elif dtype == OutTypes.uint32:
        temp_data = data.astype(np.uint32)
    # case uint64
    elif dtype == OutTypes.uint64:
        temp_data = data.astype(np.uint64)

    max_out = 2 ** (temp_data.dtype.itemsize * 8) - 1

    if debug:
        show_info(f"Max intensity: {max_out}\n")
    else:
        pass

    num_labels = (
        len(np.unique(temp_data)) - 1
    )  # subtract 1 to account for null label 0

    if debug:
        show_info(f"Labels detected: {num_labels}\n")
    else:
        pass

    intensity_multiplier = int(max_out / num_labels)

    if debug:
        show_info(f"Intensity multiplier: {intensity_multiplier}\n")
    else:
        pass

    out_data = temp_data * intensity_multiplier

    if debug:
        show_info(f"Max value of output: {out_data.max()}\n")
    else:
        pass

    layer = Layer.create(out_data, add_kwargs, layer_type)

    return layer


def convert_tiff_labels(vol: Image, debug: bool = False) -> Layer:
    """"""
    data = vol.data
    out_data = np.zeros_like(data)
    name = f"{vol.name}_labels"
    add_kwargs = {"name": name}
    layer_type = "labels"

    label_vals = np.unique(data)

    if debug:
        show_info(f"Label values: {label_vals}\n")
    else:
        pass

    for i, val in enumerate(label_vals):
        mask = data == val
        out_data[mask] = i

    if debug:
        show_info(f"out_data: {np.unique(out_data)}\n")
    else:
        pass

    layer = Layer.create(out_data, add_kwargs, layer_type)

    return layer
