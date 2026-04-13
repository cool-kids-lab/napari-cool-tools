import os.path as ospath
from pathlib import Path
from typing import List

from napari.layers import Layer
from napari.qt import thread_worker
from napari.types import FullLayerData
from napari.utils.notifications import show_info
import numpy as np

def unpack_nbits(packed_data, n_bits, count):
    """
    Unpacks a uint8 array of bytes back into an array of n-bit integers.
    'count' is the original number of elements before packing.
    """
    # 1. Unpack bytes back into raw bits
    bits = np.unpackbits(packed_data)
    
    # 2. Reshape bits to (count, n) and trim potential trailing zero-padding
    # packbits adds zeros to the end to reach a full byte
    bits = bits[:count * n_bits].reshape(-1, n_bits)
    
    # 3. Convert bit sequences back to integers
    # We multiply bits by [2^(n-1), 2^(n-2), ..., 2^0] and sum them
    powers = 2 ** np.arange(n_bits - 1, -1, -1)
    return np.sum(bits * powers, axis=1)

def npz_get_reader(path):

    if isinstance(path, str) and path.endswith(".npz"):
        return npz_file_reader
    
    return None

def npz_file_reader(path: str, return_layer:bool=True,verbose:bool=True):
    """
    """
    # ['name', 'layer_type', 'shape', 'bit_mask', 'masked_values']
    # ['name', 'layer_type', 'shape', 'bit_mask', 'label_map', 'n_bits', 'packed_remapped_values']
    npzfile = np.load(path,allow_pickle=True)
    bit_mask = np.unpackbits(npzfile["bit_mask"])

    if npzfile["layer_type"] == "image":
        value_indices = bit_mask.nonzero()[0]
        data = np.zeros(bit_mask.shape,dtype="uint8")
        data[value_indices] = npzfile["masked_values"]
        data = data.reshape(npzfile['shape'])

        if verbose:
            print(f"File {npzfile['name']} contains a bitmask of shape:\n{npzfile['shape']} with values at flat indicies:\n{value_indices}\n")

        if not return_layer:
            return data
        else:
            if "motor_position" in npzfile.files:
                return [(data,{"name":str(npzfile["name"]),"metadata":{"motor_position":int(npzfile["motor_position"])}},str(npzfile["layer_type"]))]
            else:
                return [(data,{"name":str(npzfile["name"])},str(npzfile["layer_type"]))]
    
    elif npzfile["layer_type"] == "labels":
        n_bits = npzfile["n_bits"]
        label_map = npzfile["label_map"]
        value_indices = bit_mask.nonzero()[0]
        reloaded_data = np.zeros(bit_mask.shape,dtype="uint8")

        if n_bits > 0:
            packed_remapped_values = npzfile["packed_remapped_values"]

            number_labeled_pixels = bit_mask.nonzero()[0].size
            
            unpacked_values = unpack_nbits(packed_remapped_values,n_bits=n_bits,count=number_labeled_pixels)
            mapped_back_values = label_map[unpacked_values]
            reloaded_data[value_indices] = mapped_back_values
            reloaded_data = reloaded_data.reshape(npzfile["shape"])
        else:
            reloaded_data[value_indices] = label_map[0]
            reloaded_data = reloaded_data.reshape(npzfile["shape"])

        if verbose:
            print(f"File {npzfile['name']} contains a bitmask of shape:\n{npzfile['shape']} with mapped labels: {label_map} values at flat indicies:\n{value_indices}\n")

        if not return_layer:
            return reloaded_data
        else:
            if "motor_position" in npzfile.files:
                return [(reloaded_data,{"name":str(npzfile["name"]),"metadata":{"motor_position":int(npzfile["motor_position"])}},str(npzfile["layer_type"]))]
            else:
                return [(reloaded_data,{"name":str(npzfile["name"])},str(npzfile["layer_type"]))]