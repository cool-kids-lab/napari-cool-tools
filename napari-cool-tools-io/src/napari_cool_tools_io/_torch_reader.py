import os
import os.path as ospath
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from napari.utils.notifications import show_info

from napari_cool_tools_io import torch


def torch_get_reader(path):
    """Reader for COOL lab .prof file format.

    Args:
        path(str or list of str): Path to file, or list of paths.

    Returns:
        function or None
        If the path is a recognized format, return a function that accepts the
        same path or list of paths, and returns a list of layer data tuples.
    """
    # If format is recogized return reader function
    if isinstance(path, str) and path.endswith(".pt"):
        # calculate file size in bytes
        file_size = os.path.getsize(path)

        return torch_file_reader
    return None


def torch_file_reader(path):
    """Take a path or list of paths to .prof files and return a list of LayerData tuples.

    Args:
        path(str or list of str): Path to file, or list of paths.

    Returns:
        layer_data : list of tuples
            A list of LayerData tuples where each tuple in the list contains
            (data, metadata, layer_type), where data is a numpy array, metadata is
            a dict of keyword arguments for the corresponding viewer.add_* method
            in napari, and layer_type is a lower-case string naming the type of
            layer. Both "meta", and "layer_type" are optional. napari will
            default to layer_type=="image" if not provided
    """

    data,attributes,layer_type = torch.load(path)
    data = data.numpy()
    
    # transpose array so that x and y are switched then flip array
    # to better orient b-scans for manual segmentation

    #display = data.transpose(0, 2, 1)
    # display = display[:,::-1,:]

    display = np.flip(data.transpose(0, 2, 1), 1)
    # display = b_scan

    file_name = attributes["name"]

    # optional layer type argument
    if layer_type is None:
        layer_type = "image"
    else:
        pass

    show_info(
        f"layer_name: {file_name}, shape: {display.shape}, dtype: {display.dtype}, layer type: {layer_type}\n" #data: {data},
    )
    return [(display, attributes, layer_type)]
