import os
import os.path as ospath
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from napari.utils.notifications import show_info

data_element_size = 2  # number of bytes per data element uint16 == 2 bytes

def unp_get_reader(path):
    """Reader for COOL lab .unp file format.

    Args:
        path(str or list of str): Path to file, or list of paths.

    Returns:
        function or None
        If the path is a recognized format, return a function that accepts the
        same path or list of paths, and returns a list of layer data tuples.
    """
    # If format is recogized return reader function
    if isinstance(path, str) and path.endswith(".unp"):
        # calculate file size in bytes
        file_size = os.path.getsize(path)

        # calculate number of data entries
        # in this case we are using 32 bit floating point
        # aka 4 bytes  as there are 8 bits per byte
        num_entries = file_size / data_element_size

        meta = unp_proc_meta(path, ".unp")

        # case meta data is valid
        if meta is not None:
            print(
                f"h,w,d,BMscan {meta}, size(bytes): {file_size}, data entries: {num_entries}"
            )
            # calculate width of data volume using height and depth info
            # from meta data file and calculated number of data entries
            h, w, d, bmscan, w_param, dtype, layer_type = meta

            globals()["unp_width"] = w
            globals()["unp_height"] = h
            globals()["unp_depth"] = d
            globals()["unp_bmscan"] = bmscan
            globals()["unp_width_param"] = w_param
            globals()["dtype"] = dtype
            globals()["layer_type"] = layer_type

        # case meta data is not valid
        else:
            return None

        return unp_file_reader
    return None


def unp_proc_meta(path, ext: str):
    """Process .unp file metadata.

    Args:
        path(str or list of str): Path to file, or list of paths.
        ext(str): extension of source file

    Returns:
        If .ini metafile is valid returns tuple(height(int),width(int),depth(int),bmscan(int),width_param(int),dtype(None/dtype),layer_type(None/layer_type))
        else if .xml metafile is valid returns tuple(height(int),width(int),depth(int),bmscan(int),width_param(int),dtype(None/dtype),layer_type(None/layer_type))
        else returns None

        If both .ini and .xml metafiles exist the .ini file will be used and the .xml will be ingnored
    """

    height = None
    depth = None

    show_info(f"\nOpening file: {path}")

    head, tail = ospath.split(path)

    # isolate file name from path and .unp extension
    # file_name = ospath.basename(path)
    file_name = tail

    # remove .unp extenstion
    file_no_ext = file_name.replace(ext, "")

    # remove common .unp specifiers _OCTA and _Struc
    file_base = file_no_ext.replace("_OCTA", "").replace("_Struc", "")

    # constuct path to metafile assumed to be in same directory
    meta_path = ospath.join(head, file_base + ".xml")
    show_info(f"Associated .xml meta data file: {meta_path}")
    meta_path2 = ospath.join(head, file_base + ".ini")
    show_info(f"Associated .ini meta data file: {meta_path2}")

    # verify whether meta file exists or not
    # if isinstance(meta_path, str):

    if Path(meta_path2).is_file():
        show_info(".ini Meta Data exists:")
        width_param, height, width, depth, bmscan = (
            None,
            None,
            None,
            None,
            None,
        )

        with open(meta_path2) as file:
            for line in file:
                if "WIDTH=" in line and width_param is None:
                    words = line.split("=")
                    index = words.index("WIDTH")
                    if index + 1 < len(words):
                        width_param = int(words[index + 1])
                if "HEIGHT=" in line and height is None:
                    words = line.split("=")
                    index = words.index("HEIGHT")
                    if index + 1 < len(words):
                        height = int(words[index + 1])
                        # print(height)
                if "BScanWidth=" in line and width is None:
                    words = line.split("=")
                    index = words.index("BScanWidth")
                    if index + 1 < len(words):
                        width = int(words[index + 1])
                        # print(width)
                if "FRAMES=" in line and depth is None:
                    words = line.split("=")
                    index = words.index("FRAMES")
                    if index + 1 < len(words):
                        depth = int(words[index + 1])
                        # print(depth)
                if "BMScan=" in line and bmscan is None:
                    words = line.split("=")
                    index = words.index("BMScan")
                    if index + 1 < len(words):
                        bmscan = int(words[index + 1])
                        print(bmscan)

        dtype = None
        layer_type = None

        # Case no valid values obtained from metafile return None
        if (
            depth is not None
            and height is not None
            and width is not None
            and bmscan is not None
            # and width_param is not None
        ):
            return (
                height,
                width,
                depth,
                bmscan,
                width_param,
                dtype,
                layer_type,
            )
        else:
            return None

    if Path(meta_path).is_file():
        show_info(".xml Meta Data exists:")

        tree = ET.parse(meta_path)
        root = tree.getroot()
        volume_size = root.find(".//Volume_Size")
        volume_size_attrib = volume_size.attrib
        if "Width" in volume_size_attrib:
            width_param = int(volume_size_attrib["Width"])
        else:
            width_param = None
        height = int(volume_size_attrib["Height"])
        width = int(volume_size_attrib["BscanWidth"])
        depth = int(volume_size_attrib["Number_of_Frames"])

        scanning_params = root.find(".//Scanning_Parameters")
        if scanning_params is not None:
            scanning_params_attrib = scanning_params.attrib
            bmscan = int(scanning_params_attrib["Number_of_BM_scans"])
        else:
            bmscan = None

        layer_info = root.find(".//Layer_Info")

        if layer_info is not None:
            layer_info_attrib = layer_info.attrib
            dtype = layer_info_attrib["Dtype"]
            layer_type = layer_info_attrib["Layer_Type"]
        else:
            dtype = None
            layer_type = None

        # Case no valid values obtained from metafile return None
        if (
            depth is not None
            and height is not None
            and width is not None
            # and bmscan is not None
            # and width_param is not None
        ):
            return (
                height,
                width,
                depth,
                bmscan,
                width_param,
                dtype,
                layer_type,
            )
        else:
            return None

    # case no metadata request path to metadata or cancel file load
    else:
        return None


def unp_file_reader(path):
    """Take a path or list of paths to .unp files and return a list of LayerData tuples.

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

    h = globals()["unp_height"]
    w = globals()["unp_width_param"]
    d = globals()["unp_depth"]
    bmscan = globals()["unp_bmscan"]
    dtype = globals()["dtype"]
    layer_type = globals()["layer_type"]

    # isolate file name from path and .unp extension
    # file_name = ospath.basename(path)
    head, tail = ospath.split(path)
    file_name = tail.replace(".", "_")
    #file_name = str(file_name)

    # # define chuncks as little endian f32 4 byte floats with HEIGHT values
    # # per row and WIDTH values per column
    # if dtype is None:
    #     dot_unp = np.dtype(("<f4", (h, w)))
    # else:
    #     dot_unp = np.dtype((dtype, (h, w)))

    # #load the library
    import PyQt5.QtCore
    import ctypes

    dirname = os.path.dirname(PyQt5.QtCore.__file__)
    plugin_path = os.path.join(dirname, 'plugins', 'platforms')
    os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = plugin_path

    dll_filename = os.path.dirname(__file__) + "/UNPImporter.dll"
    print(dll_filename)
    dll_lib = ctypes.WinDLL(dll_filename)

    char_array = ctypes.create_string_buffer(path.encode('utf-8'))

    dll_lib.newMainWindow.argtypes = [ctypes.c_char_p,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_void_p]
    dll_lib.newMainWindow.restypes = [ctypes.c_bool]
    
    display = np.zeros((d,h,int(w/2)),dtype=np.float32)
    buffer = display.ctypes.data_as(ctypes.c_void_p)

    dll_lib.newMainWindow(char_array,ctypes.c_int(w),
                             ctypes.c_int(h),
                             ctypes.c_int(d),
                             ctypes.c_int(bmscan),buffer)

    del dll_lib


    display = np.flip(display.transpose(0, 2, 1),1)

    # optional kwargs for viewer.add_* method
    add_kwargs = {"name": file_name}

    # optional layer type argument
    if layer_type is None:
        layer_type = "image"
    else:
        pass

    show_info(
        f"layer_name: {file_name}, shape: {display.shape}, bmscan: {bmscan}, dtype: {display.dtype}, layer type: {layer_type}\n"
    )
    return [(display, add_kwargs, layer_type)]
