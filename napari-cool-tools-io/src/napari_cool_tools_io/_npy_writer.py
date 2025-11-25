import os
from pathlib import Path
from typing import List

from napari.layers import Layer
from napari.qt import thread_worker
from napari.types import FullLayerData
from napari.utils.notifications import show_info
import numpy as np

def npy_get_writer(path: str, layer_data: list[FullLayerData]) -> List[str]:
    """Saves a napari scene in .MAT file format
    Args:
        path(str or list of str): Path to file, or list of paths.
        layer_data(List[FullLayerData]): list of FullLayerData (Any,Dict,str) -> (data,kwargs,layer_type)
    Returns:
        List[path]: List of paths to .MAT files
    """

    if len(layer_data) == 1:
        data, attributes, layer_type = layer_data[0]
        name = attributes["name"]

        show_info(f"path: {path}, shape: {data.shape}, attributes: {name}\n")

        # # case .prof files should be 3 dimensional
        # if data.ndim == 3:
        worker = save_numpy(path, data)
        worker.start()
        # save_data.tofile(path)

        show_info(f"Saving {path}")
        # else:
        #     raise ValueError(
        #         f"File contains {data.ndim}-dimensional data .prof files only support 3-dimensions."
        #     )
    else:
        p_path = Path(path)
        p_dir = Path(p_path.parent) / p_path.stem
        ext = p_path.suffix
        os.makedirs(p_dir, exist_ok=True)

        for layer in layer_data:
            data, attributes, layer_type = layer
            name = attributes["name"]
            out_path = p_dir / f"{name}{ext}"

            show_info(f"path: {out_path}, shape: {data.shape}, attributes: {name}\n")

            # # case .prof files should be 3 dimensional
            # if data.ndim == 3:
            worker = save_numpy(out_path, data)
            worker.start()

            show_info(f"Saving {out_path}")
            # else:
            #     raise ValueError(
            #         f"File contains {data.ndim}-dimensional data .prof files only support 3-dimensions."
            #     )

    return [path]



@thread_worker(progress=True)  # give us an indeterminate progress bar
def save_numpy(path: str, data: np.ndarray):
    """Thread worker to save numpy data to file
    Args:
        path (str): Path to the file
        data (np.ndarray): Numpy array to save
    """
    np.save(path, data)
    show_info(f"{path} was saved\n")
    return