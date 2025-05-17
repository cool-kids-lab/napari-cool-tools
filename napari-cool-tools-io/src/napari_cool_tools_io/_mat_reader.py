import os.path as ospath

import numpy as np
from napari.utils.notifications import show_info
from scipy.io import loadmat


def is_mat_file(path):
    """Verify that the file the user is attempting to open is a .MAT file.

    Args:
        path(str or list of str): Path to file, or list of paths.

    Returns:
        (is_mat_file(Bool), out(Dict)): If the load is successful True, and an output dictionary containing .MAT file contents is retrurned
        If the load is unsuccessful False, and None are returned
    """
    is_mat_file = False
    out = None
    try:
        loadmat(path)
    # case not MAT file
    except FileNotFoundError:
        show_info(f"\nFile {path} could not be opened as .MAT file\n")
        is_mat_file = False
        out = None
    # case is MAT File
    else:
        is_mat_file = True
        out = loadmat(path)
    # regardless
    finally:
        pass

    return (is_mat_file, out)


def mat_get_reader(path):
    """Reader for Matlab .MAT file format.

    Args:
        path(str or list of str): Path to file, or list of paths.

    Returns:
        function or None
        If the path is a recognized format, return a function that accepts the
        same path or list of paths, and returns a list of layer data tuples.
    """
    # If format is recogized return reader function
    if isinstance(path, str) and path.endswith(".mat"):
        return mat_file_reader
        # return PushButton.get_value()

    # otherwise return None
    return None


def proc_napari_mat_data(mat_napari):
    """Process list of napari layer_data_tuples converting them from .MAT format to python dictionary format

    Args:
        mat_napari(Dict): Dictionary obtained from "napari" key of .MAT file Dictionary loaded from path

    Returns:
        layer_data_tuples(List[LayerDataTuple]) list of napari layer data tuples
    """

    layers = mat_napari[0]

    layer_data_tuples = []

    for _i, layer in enumerate(layers):
        data = layer[0]
        attr = layer[1][0, 0]
        l_type = layer[2].item()

        nap_dict = proc_mat_dict(attr)

        layer_data_tuple = (data, nap_dict, l_type)

        layer_data_tuples.append(layer_data_tuple)

    return layer_data_tuples


def proc_napari_data_only_mat(mat_dict):
    """Converts data only flagged .MAT file dictionary to napari LayerDataTuples

    Args:
        mat_napari(Dict): .MAT file Dictionary loaded from path

    Returns:
        layer_data_tuples(List[LayerDataTuple]) list of napari layer data tuples
    """
    show_info(f"{mat_dict}")
    data = mat_dict["data"]
    name = mat_dict["name"][0]
    layer_type = mat_dict["layer_type"][0]
    kwargs = {"name": name}
    # show_info(f"name: {name}, kwargs: {kwargs} ({type(kwargs)}), layer_type: {layer_type}")
    layer_data_tuple = (data, kwargs, layer_type)
    layer_data_tuples = [layer_data_tuple]
    return layer_data_tuples


def proc_mat_dict(dict_val):
    """Process data stored in .MAT files that represent python Dictionaries

    Args:
        dict_val(dtype): Value of key that is numpy.dtype[void] which indicates dictionary type

    Returns:
        out_dict(Dict): Output dictionary of kwargs for napari LayerDataTuple type
    """

    # keys to exclude
    excluded = "shear"

    # initialize output dictionary
    out_dict = {}

    # case value is of proper type and represents a python dictionary
    if dict_val.dtype.kind == "V":
        attr_key = dict_val.dtype.names
        attr = dict_val.item()

        for j, key in enumerate(attr_key):
            attr_val = attr[j].squeeze()

            # case value is python dictionary
            dtype = attr_val.dtype
            if dtype.kind == "V":
                val = proc_mat_dict(attr_val)

            # case value is empty python ditionary
            elif dtype.kind == "O":
                val = {}

            # case any other type
            else:
                # case scalar value
                if attr_val.ndim < 1:
                    val = attr_val.item()
                # case linear value store as python list
                elif attr_val.ndim == 1:
                    val = list(attr_val)
                # case ndimensional value keep as numpy ndarray
                else:
                    val = attr_val

            # case exclude specific meta data values
            if key in excluded:
                pass
            else:
                # populate dictionary
                out_dict[key] = val
        pass

    # case value is not of proper type
    else:
        print(
            f"value is not of correct kind 'V', instead it is of kind {dict_val.dtype.kind}"
        )
        pass
    pass

    return out_dict


def mat_file_reader(path):  # -> List[T]:
    # load .prof into numpy array
    """Open .MAT matlab files using scipy with special processing flags for dealing with specific COOL lab file structures

    Args:
        path(str or list of str): Path to file, or list of paths.

    Returns:
        display_out(List[LayerDataTuple])/None: if .MAT file is properly formated returns list of napari LayerDataTuples otherwise returns None
    """

    # init display output array
    display_out = []

    # isolate file name from path and .prof extension
    head, tail = ospath.split(path)
    file_name = tail.replace(".", "_")
    show_info(f"\n\nlayer_name: {file_name}\n")

    verbose = True

    # verify that file is valid .MAT file
    out = is_mat_file(path)

    # case is .MAT file
    if out[0] is True:
        # .MAT file Dict
        mat_dict = out[1]

        # case if verbose flag is true
        if verbose:
            show_info(f"Opening .MAT file: {path}\n\nContents:\n{mat_dict}\n")

        # case verbose flag is false
        else:
            show_info(
                f"Opening .MAT file: {path}\n\nHeader: {mat_dict['__header__']}\n"
            )

        # determine if file format napari scene or data only .MAT file
        print(f"'napari' key in mat_dict?, {'napari' in mat_dict}\n\n")
        print(
            f"'napari_data_only' key in mat_dict?, {'napari_data_only' in mat_dict}\n\n"
        )

        # case Napari .MAT file format
        if "napari" in mat_dict:
            mat_napari = mat_dict["napari"]
            display_out = proc_napari_mat_data(mat_napari)
            show_info(f".MAT file {path} has been loaded")
            return display_out

        elif "napari_data_only" in mat_dict:
            display_out = proc_napari_data_only_mat(mat_dict)
            show_info(f".MAT file {path} has been loaded")
            return display_out

        else:
            # process numpy arrays in dictionary
            for key in mat_dict:
                # case is an instance of a numpy array
                if isinstance(mat_dict[key], np.ndarray):
                    # get array name
                    a_name = key

                    # get array and remove extraneous dimensions
                    a = mat_dict[key].squeeze()

                    # dimension and data type flags
                    v_dim = False
                    v_dtype = False

                    # whether array has 2+ dimensions or not
                    v_dim = a.dim > 0

                    # case array is of basic numeric type
                    # i-int, u-uint, f-float, c-complex
                    if any(char in a.dtype.str for char in r"iufc"):
                        v_dtype = True

                    # case not basic numeric type
                    else:
                        v_dtype = False

                    # case both dimension and data type are valid
                    if v_dim and v_dtype:
                        # optional kwargs for viewer.add_* method
                        add_kwargs = {"name": f"{a_name}_{file_name}"}

                        # optional layer type argument
                        layer_type = "image"

                        # append output to display output
                        display_out.append((a, add_kwargs, layer_type))

                    # case dimension, data type or both are invalid
                    else:
                        pass

                # case not  numpy array
                else:
                    pass

        # case display_out contains data
        if len(display_out) > 0:
            show_info(f".MAT file {path} has been loaded")

            return display_out

        # case display_out is empty
        else:
            show_info(
                "File is a properly formatted .MAT file but does not contain multidimensional numpy arrays of basic numerical data types.\n"
            )
            show_info(f"Dictionary of .MAT file contents:\n\n{mat_dict}\n")
            return None

    # case is not .MAT file
    else:
        show_info("File is not a properly formatted .MAT file.\n")
        return None
