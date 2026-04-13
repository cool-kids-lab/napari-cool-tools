#import os
from enum import Enum
from pathlib import Path
from typing import List

from napari.layers import Layer
from napari.qt import thread_worker
from napari.types import FullLayerData
from napari.utils.notifications import show_info
import numpy as np

from napari_cool_tools_img_proc import DType
from napari_cool_tools_img_proc._normalization_funcs import convert_dtype_and_rescale

def npz_get_writer(path: str, layer_data: list[FullLayerData]) -> List[str]:
    """Saves a napari scene in .npz file format
    Args:
        path(str or list of str): Path to file, or list of paths.
        layer_data(List[FullLayerData]): list of FullLayerData (Any,Dict,str) -> (data,kwargs,layer_type)
    Returns:
        List[path]: List of paths to .npz files
    """

    if len(layer_data) == 1:
        data, attributes, layer_type = layer_data[0]
        name = attributes["name"]
        user_metadata = attributes.get('metadata', {})
        if "motor_position" in user_metadata:
            save_dict = {"name":name,"layer_type":layer_type,"shape":data.shape,"motor_position":user_metadata["motor_position"]}
        else:
            save_dict = {"name":name,"layer_type":layer_type,"shape":data.shape}

        #print(f"attributes: {attributes}\n")
        show_info(f"path: {path}, shape: {data.shape}, name: {name}\n")

        worker = save_npz_thread(path, data, save_dict)
        worker.start()

        show_info(f"Saving {path}")

    else:
        p_path = Path(path)
        p_dir = Path(p_path.parent) / p_path.stem
        ext = p_path.suffix
        p_dir.mkdir(parents=True,exist_ok=True)
        #os.makedirs(p_dir, exist_ok=True)

        for layer in layer_data:
            data, attributes, layer_type = layer
            name = attributes["name"]
            save_dict = {"name":name,"layer_type":layer_type,"shape":data.shape}
            out_path = p_dir / f"{name}{ext}"

            show_info(f"path: {path}, shape: {data.shape}, name: {name}\n")

            # # case .prof files should be 3 dimensional
            # if data.ndim == 3:
            worker = save_npz_thread(out_path, data, save_dict)
            worker.start()

            show_info(f"Saving {out_path}")
            # else:
            #     raise ValueError(
            #         f"File contains {data.ndim}-dimensional data .prof files only support 3-dimensions."
            #     )

    return [path]

def pack_nbits(data, n_bits):
    """
    Packs an array of integers into an array of n-bit values.
    Returns a uint8 array of packed bytes.
    """
    # 1. Convert each number to its n-bit binary representation
    # We shift each value by [n-1, n-2, ..., 0] and bitwise-AND with 1
    shifts = np.arange(n_bits - 1, -1, -1)
    bits = (data[:, np.newaxis] >> shifts) & 1
    
    # 2. Flatten and pack into uint8 bytes
    # packbits by default uses big-endian bit order
    return np.packbits(bits)

def remap_values(value_array:np.ndarray,value_map:np.ndarray,verbose:bool=False):
    """
    """
    if verbose:
        print(f"value array min,max: {value_array.min(),value_array.max()}\nvalue map: {value_map}\n")
    for value in value_map:
        mask = value_array == value
        idx = (value_map == value).nonzero()[0][0]
        value_array[mask] = idx

    return value_array

@thread_worker(progress=True)  # give us an indeterminate progress bar
def save_npz_thread(path: str, data: np.ndarray, save_dict:dict, verbose:bool=True, debug:bool=False):
    """Thread worker to save numpy data to file in custom mapped .npz format
    Args:
        path (str): Path to the file
        data (np.ndarray): Numpy array to save
    """

    if save_dict["layer_type"] == "labels":
        n_bits = 8
        bit_mask = data > 0
        packed_bit_mask = np.packbits(bit_mask)
        save_dict["bit_mask"] = packed_bit_mask

        label_map = np.unique(data)
        label_map = label_map[label_map != 0]
        unique_labels = len(label_map)
        save_dict["label_map"] = label_map


        match unique_labels:
            case num_vals if num_vals == 1:
                print(f"Label values are not required the bit mask is sufficient for a single label, {label_map}")
                n_bits = 0
            case num_vals if num_vals <= 2:
                n_bits = 1
                print(f"Label values {label_map} can be stored in 1 bit")
            case num_vals if num_vals <= 4:
                n_bits = 2
                print(f"Label values {label_map} can be stored in 2 bits")
            case num_vals if num_vals <= 16:
                n_bits = 4                
                print(f"Label values {label_map} can be stored in 4 bits")
            case num_vals if num_vals <= 256:
                print(f"Label values {label_map} can be stored in 8 bits")
            case num_vals if num_vals > 256:
                assert ValueError(f"There are {num_vals} unique labels which exceeds the 256 unique labels that are supported\n")

        save_dict["n_bits"] = n_bits

        remapped_values = remap_values(data[bit_mask],label_map)

        if n_bits > 0:
            pack_remapped_values = pack_nbits(remapped_values,n_bits=n_bits)
        else:
            pack_remapped_values = "single label bitmask only"

        save_dict["packed_remapped_values"] = pack_remapped_values

        if verbose:
            print(f"{save_dict['name']} stores {unique_labels} labels:\n{label_map} packed into {n_bits} bits per value")
            # print(f"label map: {label_map}\nunique labels: {unique_labels}\n")
            # print(f"remapped values: {remapped_values}, shape: {remapped_values.shape}\n")
            # print(f"packed remapped values: {pack_remapped_values}, shape: {pack_remapped_values.shape}\n")

        # unpacked_values = unpack_nbits(pack_remapped_values,n_bits=n_bits,count=remapped_values.size)
        # #mapped_back_values = label_map[remapped_values]
        # mapped_back_values = label_map[unpacked_values]
        # reloaded_data = np.zeros_like(data,dtype="uint8")
        # reloaded_data[bit_mask] = mapped_back_values
        # reloaded_equal_original = np.array_equal(data,reloaded_data)

        # print(f"values: {data[bit_mask]}\nmapping: {remapped_values}\nmapped back: {mapped_back_values}\n")
        # print(f"data == reloaded_data: {reloaded_equal_original}\n")

        np.savez_compressed(path,**save_dict)
        print(f"{path} was saved\n")
        return
    
    elif save_dict["layer_type"] == "image":

        # convert data to byte scale
        converted_data = convert_dtype_and_rescale(data,datatype=DType.NP_UINT8)
        # generate bit mask for nonzero values
        bit_mask = converted_data > 0
        packed_bit_mask = np.packbits(bit_mask)

        # get array of nonzero values
        values = converted_data[bit_mask]

        # calculate new size in GB and compare to uint8 size
        bit_mask_gb = ((bit_mask.nonzero()[0].shape[0]) / 8) / (1043**3)
        values_gb = values.shape[0] / (1043**3)
        new_gb = bit_mask_gb + values_gb
        old_gb = data.flatten().shape[0] / (1043**3)
        gb_ratio = new_gb/old_gb

        if verbose or debug:
            print(f"New (min,max) values (dtype): {converted_data.min(),converted_data.max()} ({converted_data.dtype})\n")
            print(f"bitmask shape, nonzero bitmask shape, values shape: {bit_mask.shape},{bit_mask.nonzero()[0].shape},{values.shape}\n")
        if debug:
            print(f"new v old size: {new_gb} / {old_gb} = {gb_ratio}\n")
            print(f"Size savings in GB: {1.-gb_ratio}\n")

        print("Saving .npz format\n")

        # prepare data to save bitmask and values
        save_dict["bit_mask"] = packed_bit_mask
        save_dict["masked_values"] = converted_data[bit_mask].flatten()
        np.savez_compressed(path,**save_dict)
        # else:
        #     print("Saving .png byte format\n")


    #np.save(path, data)
    print(f"{path} was saved\n")
    return

def save_npz(path: str, data: np.ndarray, save_dict:dict, verbose:bool=True, debug:bool=False):
    """Thread worker to save numpy data to file in custom mapped .npz format
    Args:
        path (str): Path to the file
        data (np.ndarray): Numpy array to save
    """

    if save_dict["layer_type"] == "labels":
        n_bits = 8
        bit_mask = data > 0
        packed_bit_mask = np.packbits(bit_mask)
        save_dict["bit_mask"] = packed_bit_mask

        label_map = np.unique(data)
        label_map = label_map[label_map != 0]
        unique_labels = len(label_map)
        save_dict["label_map"] = label_map


        match unique_labels:
            case num_vals if num_vals == 1:
                print(f"Label values are not required the bit mask is sufficient for a single label, {label_map}")
                n_bits = 0
            case num_vals if num_vals <= 2:
                n_bits = 1
                print(f"Label values {label_map} can be stored in 1 bit")
            case num_vals if num_vals <= 4:
                n_bits = 2
                print(f"Label values {label_map} can be stored in 2 bits")
            case num_vals if num_vals <= 16:
                n_bits = 4                
                print(f"Label values {label_map} can be stored in 4 bits")
            case num_vals if num_vals <= 256:
                print(f"Label values {label_map} can be stored in 8 bits")
            case num_vals if num_vals > 256:
                assert ValueError(f"There are {num_vals} unique labels which exceeds the 256 unique labels that are supported\n")

        save_dict["n_bits"] = n_bits

        remapped_values = remap_values(data[bit_mask],label_map)

        if n_bits > 0:
            pack_remapped_values = pack_nbits(remapped_values,n_bits=n_bits)
        else:
            pack_remapped_values = "single label bitmask only"

        save_dict["packed_remapped_values"] = pack_remapped_values

        if verbose:
            print(f"{save_dict['name']} stores {unique_labels} labels:\n{label_map} packed into {n_bits} bits per value")
            # print(f"label map: {label_map}\nunique labels: {unique_labels}\n")
            # print(f"remapped values: {remapped_values}, shape: {remapped_values.shape}\n")
            # print(f"packed remapped values: {pack_remapped_values}, shape: {pack_remapped_values.shape}\n")

        # unpacked_values = unpack_nbits(pack_remapped_values,n_bits=n_bits,count=remapped_values.size)
        # #mapped_back_values = label_map[remapped_values]
        # mapped_back_values = label_map[unpacked_values]
        # reloaded_data = np.zeros_like(data,dtype="uint8")
        # reloaded_data[bit_mask] = mapped_back_values
        # reloaded_equal_original = np.array_equal(data,reloaded_data)

        # print(f"values: {data[bit_mask]}\nmapping: {remapped_values}\nmapped back: {mapped_back_values}\n")
        # print(f"data == reloaded_data: {reloaded_equal_original}\n")

        np.savez_compressed(path,**save_dict)
        print(f"{path} was saved\n")
        return
    
    elif save_dict["layer_type"] == "image":

        # convert data to byte scale
        converted_data = convert_dtype_and_rescale(data,datatype=DType.NP_UINT8)
        # generate bit mask for nonzero values
        bit_mask = converted_data > 0
        packed_bit_mask = np.packbits(bit_mask)

        # get array of nonzero values
        values = converted_data[bit_mask]

        # calculate new size in GB and compare to uint8 size
        bit_mask_gb = ((bit_mask.nonzero()[0].shape[0]) / 8) / (1043**3)
        values_gb = values.shape[0] / (1043**3)
        new_gb = bit_mask_gb + values_gb
        old_gb = data.flatten().shape[0] / (1043**3)
        gb_ratio = new_gb/old_gb

        if verbose or debug:
            print(f"New (min,max) values (dtype): {converted_data.min(),converted_data.max()} ({converted_data.dtype})\n")
            print(f"bitmask shape, nonzero bitmask shape, values shape: {bit_mask.shape},{bit_mask.nonzero()[0].shape},{values.shape}\n")
        if debug:
            print(f"new v old size: {new_gb} / {old_gb} = {gb_ratio}\n")
            print(f"Size savings in GB: {1.-gb_ratio}\n")

        print("Saving .npz format\n")

        # prepare data to save bitmask and values
        save_dict["bit_mask"] = packed_bit_mask
        save_dict["masked_values"] = converted_data[bit_mask].flatten()
        np.savez_compressed(path,**save_dict)
        # else:
        #     print("Saving .png byte format\n")


    #np.save(path, data)
    print(f"{path} was saved\n")
    return