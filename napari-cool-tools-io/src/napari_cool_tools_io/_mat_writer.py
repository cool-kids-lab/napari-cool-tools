import os
from pathlib import Path
from typing import List

from napari.layers import Layer
from napari.qt import thread_worker
from napari.types import FullLayerData
from napari.utils.notifications import show_info
from scipy.io import savemat


def mat_file_writer(path: str, layer_data: list[FullLayerData]) -> List[str]:
    """Saves a napari scene in .MAT file format
    Args:
        path(str or list of str): Path to file, or list of paths.
        layer_data(List[FullLayerData]): list of FullLayerData (Any,Dict,str) -> (data,kwargs,layer_type)
    Returns:
        List[path]: List of paths to .MAT files
    """

    show_info(f"\nwriting to file: {path}\n\nFullLayerData: {layer_data}\n")

    mat_dict = {"napari": [layer_data]}
    # savemat(path, mat_dict, do_compression=True)
    worker = save_numpy(path, mat_dict, False)
    worker.start()

    show_info(f"\n.MAT file dictionary: {mat_dict}\n\n")

    return [path]


@thread_worker(progress=True)  # give us an indeterminate progress bar
def save_numpy(path, mat_dict, do_compression):
    """Thread wrapper for Scipy savemat function
    Args:
        path(str or list of str): Path to file, or list of paths.
        mat_dict(Dict): dictionary to be saved in .MAT file
        do_compression(Bool): If true uses compression when saving .MAT file
    Returns:
        None: Saves napari data as .MAT file
    """
    savemat(path, mat_dict, do_compression)
    show_info(f"{path} was saved\n")
    return


def compressed_mat_file_writer(
    path: str, layer_data: list[FullLayerData]
) -> List[str]:
    """Saves a napari scene in compressed .MAT file format
    Args:
        path(str or list of str): Path to file, or list of paths.
        layer_data(List[FullLayerData]): list of FullLayerData (Any,Dict,str) -> (data,kwargs,layer_type)
    Returns:
        List[path]: List of paths to .MAT files
    """

    show_info(f"\nwriting to file: {path}\n\nFullLayerData: {layer_data}\n")

    mat_dict = {"napari": [layer_data]}
    # savemat(path, mat_dict, do_compression=True)
    worker = save_numpy(path, mat_dict, True)
    worker.start()

    show_info(f"\n.MAT file dictionary: {mat_dict}\n\n")

    return [path]


def data_mat_file_writer(path: str, layers: list[Layer]) -> List[str]:
    """Saves napari layer data only in .MAT file format
    Args:
        path(str or list of str): Path to file, or list of paths.
        layer_data(List[FullLayerData]): list of FullLayerData (Any,Dict,str) -> (data,kwargs,layer_type)
    Returns:
        List[path]: List of paths to .MAT files
    """

    if len(layers) == 1:
        # layers consist of LayerDataTuples (data,kwargs,layer_type)
        layer = layers[0]

        data = layer[0]
        name = layer[1]["name"]
        layer_type = layer[2]

        show_info(
            f"\nwriting to file: {path}\n\n{name} is a {layer_type} layer containing data:\n{data}\n"
        )

        mat_dict = {
            "napari_data_only": True,
            "data": data,
            "name": name,
            "layer_type": layer_type,
        }
        worker = save_numpy(path, mat_dict, False)
        worker.start()

        show_info(f"\n.MAT file dictionary: {mat_dict}\n\n")

    else:
        path = Path(path)
        p_dir = Path(path.parent) / path.stem
        ext = path.suffix
        os.makedirs(p_dir, exist_ok=True)

        for layer in layers:
            # layers consist of LayerDataTuples (data,kwargs,layer_type)
            data = layer[0]
            name = layer[1]["name"]
            layer_type = layer[2]
            out_path = p_dir / f"{name}{ext}"

            show_info(
                f"\nwriting to file: {out_path}\n\n{name} is a {layer_type} layer containing data:\n{data}\n"
            )

            mat_dict = {
                "napari_data_only": True,
                "data": data,
                "name": name,
                "layer_type": layer_type,
            }
            worker = save_numpy(out_path, mat_dict, False)
            worker.start()

            show_info(f"\n.MAT file dictionary: {mat_dict}\n\n")

    return [path]


def compressed_data_mat_file_writer(
    path: str, layers: list[Layer]
) -> List[str]:
    """Saves napari layer data only in compressed .MAT file format
    Args:
        path(str or list of str): Path to file, or list of paths.
        layer_data(List[FullLayerData]): list of FullLayerData (Any,Dict,str) -> (data,kwargs,layer_type)
    Returns:
        List[path]: List of paths to .MAT files
    """

    if len(layers) == 1:
        # layers consist of LayerDataTuples (data,kwargs,layer_type)
        layer = layers[0]

        data = layer[0]
        name = layer[1]["name"]
        layer_type = layer[2]

        show_info(
            f"\nwriting to file: {path}\n\n{name} is a {layer_type} layer containing data:\n{data}\n"
        )

        mat_dict = {
            "napari_data_only": True,
            "data": data,
            "name": name,
            "layer_type": layer_type,
        }
        worker = save_numpy(path, mat_dict, True)
        worker.start()

        show_info(f"\n.MAT file dictionary: {mat_dict}\n\n")

    else:
        path = Path(path)
        p_dir = Path(path.parent) / path.stem
        ext = path.suffix
        os.makedirs(p_dir, exist_ok=True)

        for layer in layers:
            # layers consist of LayerDataTuples (data,kwargs,layer_type)

            data = layer[0]
            name = layer[1]["name"]
            layer_type = layer[2]
            out_path = p_dir / f"{name}{ext}"

            show_info(
                f"\nwriting to file: {out_path}\n\n{name} is a {layer_type} layer containing data:\n{data}\n"
            )

            mat_dict = {
                "napari_data_only": True,
                "data": data,
                "name": name,
                "layer_type": layer_type,
            }
            worker = save_numpy(out_path, mat_dict, True)
            worker.start()

            show_info(f"\n.MAT file dictionary: {mat_dict}\n\n")

    return [path]
