import os.path as ospath
from pathlib import Path
from typing import List

from napari.layers import Layer
from napari.qt import thread_worker
from napari.types import FullLayerData
from napari.utils.notifications import show_info
import numpy as np

def npy_get_reader(path):

    if isinstance(path, str) and path.endswith(".npy"):
        return npy_file_reader
    
    return None


def npy_file_reader(path):

    data = np.load(path)

    _, tail = ospath.split(path)
    file_name = tail.split(".")[0]
    add_kwargs = {"name": file_name}
    layer_type = "image"

    return [(data, add_kwargs, layer_type)]