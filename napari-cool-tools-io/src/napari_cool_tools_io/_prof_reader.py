import os
import os.path as ospath
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from napari.utils.notifications import show_info

data_element_size = 4  # number of bytes per data element f32 == 4 bytes

def ini_proc_word(line,target_str):
    """"""
    words = line.split("=")
    index = words.index(target_str)
    if index + 1 < len(words):
        val = int(words[index + 1])
        return val
    else:
        print("ERROR in ini_proc_word function")
        return None

def prof_get_reader(path):
    """Reader for COOL lab .prof file format.

    Args:
        path(str or list of str): Path to file, or list of paths.

    Returns:
        function or None
        If the path is a recognized format, return a function that accepts the
        same path or list of paths, and returns a list of layer data tuples.
    """
    # If format is recogized return reader function
    if isinstance(path, str) and path.endswith(".prof"):
        # calculate file size in bytes
        file_size = os.path.getsize(path)

        # calculate number of data entries
        # in this case we are using 32 bit floating point
        # aka 4 bytes  as there are 8 bits per byte
        num_entries = file_size / data_element_size

        meta = prof_proc_meta(path, ".prof")

        # case meta data is valid
        if meta is not None:
            print(
                f"h,w,d,BMscan {meta}, size(bytes): {file_size}, data entries: {num_entries}"
            )
            # calculate width of data volume using height and depth info
            # from meta data file and calculated number of data entries
            h, w, d, bmscan, w_param, dtype, layer_type = meta

            globals()["prof_width"] = w
            globals()["prof_height"] = h
            globals()["prof_depth"] = d
            globals()["prof_bmscan"] = bmscan
            globals()["prof_width_param"] = w_param
            globals()["dtype"] = dtype
            globals()["layer_type"] = layer_type

        # case meta data is not valid
        else:
            return None

        return prof_file_reader
    return None


def prof_proc_meta(path, ext: str):
    """Process .prof file metadata.

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

    # isolate file name from path and .prof extension
    # file_name = ospath.basename(path)
    file_name = tail

    # remove .prof extenstion
    file_no_ext = file_name.replace(ext, "")

    # remove common .prof specifiers _OCTA and _Struc
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

        data = {"section":"", "content":""}
        settings = []

        with open(meta_path2) as file:
            for i,line in enumerate(file):
                if "[" not in line:
                    data["content"] = f"{data['content']}{line}"

                    if data["section"] == "General" and "WIDTH=" in line:
                        width_param = ini_proc_word(line,"WIDTH")
                    if data["section"] == "General" and "HEIGHT=" in line:
                        height = ini_proc_word(line,"HEIGHT")
                    if data["section"] == "General" and "FRAMES=" in line:
                        depth = ini_proc_word(line,"FRAMES")
                    if data["section"] == "OCT" and "BScanWidth=" in line:
                        width = ini_proc_word(line,"BScanWidth")
                    if data["section"] == "OCTA" and "BMScan=" in line:
                        bmscan = ini_proc_word(line,"BMScan")
                else:
                    if i != 0:
                        settings.append(data)
                    data = {"section":"", "content":""}
                    data["section"] = line.replace("[","").replace("]","").replace("\n","")
                    settings
            settings.append(data)        

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


def prof_file_reader(path):
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

    h = globals()["prof_height"]
    w = globals()["prof_width"]
    bmscan = globals()["prof_bmscan"]
    dtype = globals()["dtype"]
    layer_type = globals()["layer_type"]

    # isolate file name from path and .prof extension
    # file_name = ospath.basename(path)
    head, tail = ospath.split(path)
    file_name = tail.replace(".", "_")

    # define chuncks as little endian f32 4 byte floats with HEIGHT values
    # per row and WIDTH values per column
    if dtype is None:
        dot_prof = np.dtype(("<f4", (h, w)))
    else:
        dot_prof = np.dtype((dtype, (h, w)))

    # generate numpy array by loading 400 * 496 * f32 sized data chunks
    # and stacking them until end of file is reached
    b_scan = np.fromfile(path, dtype=dot_prof, count=-1)

    # transpose array so that x and y are switched then flip array
    # to better orient b-scans for manual segmentation
    display = b_scan.transpose(0, 2, 1)
    # display = display[:,::-1,:]
    display = np.flip(b_scan.transpose(0, 2, 1), 1)
    # display = b_scan

    # # Determine if volume is octa
    # if bmscan is not None and bmscan > 1:
    #     show_info("This is an OCTA Volume!!")
    #     sign = 1
    #     fix_octa = np.empty_like(display)
    #     for i in range(len(fix_octa)):
    #         if sign == -1:
    #             fix_octa[i] = np.flip(display[i], axis=1)
    #             #fix_octa[i] = display[i]
    #         else:
    #             fix_octa[i] = display[i]
    #         if (i + 1) % bmscan == 0:
    #             sign = sign * -1

    #     display = fix_octa

    # optional kwargs for viewer.add_* method
    add_kwargs = {"name": file_name}

    # optional layer type argument
    if layer_type is None:
        layer_type = "image"
    else:
        pass

    show_info(
        f"layer_name: {file_name}, shape: {display.shape}, dtype: {display.dtype}, layer type: {layer_type}\n" #bmscan: {bmscan},
    )
    return [(display, add_kwargs, layer_type)]
