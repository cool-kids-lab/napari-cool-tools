"""
Pytorch Batch Processing of UNP files
"""
import gc
from pathlib import Path
import threading
from typing import Literal

# from qtpy.QtWidgets import QApplication
import numpy as np
import napari
from magicgui import magicgui

from napari_cool_tools_img_proc import DType
from napari_cool_tools_img_proc._equalization_funcs import init_bscan_preproc, init_bscan_preproc_pt
from napari_cool_tools_img_proc._normalization_funcs import normalize_data_in_range_func, standardize_data_func, convert_dtype_and_rescale
from napari_cool_tools_io import unp_meta, device
#from napari_cool_tools_io._npz_writer import save_npz
from napari_cool_tools_io._unp_reader import unp_batch_proc_meta
from napari_cool_tools_io.process_unp import process_unp, set_dispersion_coefficients_torch
from napari_cool_tools_oct_preproc._oct_preproc_func import auto_contrast, desine
from napari_cool_tools_segmentation._label_cleaning_funcs import clean_labels,find_indices_after
from napari_cool_tools_segmentation._segmentation_funcs import  bscan_onnx_seg_func, bscan_onnx_deconj_func,enface_onnx_seg_func
from napari_cool_tools_vol_proc._averaging_tools_funcs import average_per_bscan,average_bscans
import torch
from tqdm import tqdm

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

def save_clean_labels(name:str,output_dir:Path,ret_chor_labels:np.ndarray,imaging_range:float,save_clean_ret_chor:bool=True,viewer:napari.Viewer=None):
    # clean retchor labels
    label_values = np.unique(ret_chor_labels)
    non_zero_labels = label_values > 0
    label_values = label_values[non_zero_labels]
    
    clean_ret_chor_labels = np.zeros_like(ret_chor_labels)
    for label_value in tqdm(label_values[:1],desc="processing labels in layer"):
        processed_labels = {}
        (
            processed_labels["depth_map"],
            processed_labels["ret_surf_coords"],
            processed_labels["rpe_surf_coords"],
            processed_labels["thick_map"],
            processed_labels["difference_map"],
            processed_labels["outlier_coordinate_mask"],
            processed_labels["clean_label"],
            processed_labels["percent_outliers"],
        ) = clean_labels(
            label_data=ret_chor_labels == label_value,
            imaging_range=imaging_range, #12.0,
            refractive_index=1.33,
            gap_threshold=8,
            thickness_threshold=600,
            incedence_allowance=1 / np.sin(np.pi / 4),
            component_to_pixel_thickness_ratio = (1 / 3),
            dust_threshold=1.0e6,            
        )
        clean_ret_chor_labels = (processed_labels["clean_label"] > 0)*label_value + clean_ret_chor_labels
    
    clean_ret_chor_name = f"{name}_clean_retchor"

    if viewer:
        viewer.add_labels(clean_ret_chor_labels,name=f"clean_{clean_ret_chor_name}")

    if save_clean_ret_chor:
        clean_ret_chor_path = output_dir/f"{clean_ret_chor_name}.npz"
        save_dict = {"name":clean_ret_chor_name,"layer_type":'labels',"shape":clean_ret_chor_labels.shape}
        save_npz(path=clean_ret_chor_path,data=clean_ret_chor_labels,save_dict=save_dict,verbose=False)
        save_clean_retchor_thread = threading.Thread(target=save_npz,kwargs={"path":clean_ret_chor_path,"data":clean_ret_chor_labels,"save_dict":save_dict,"verbose":False})
        save_clean_retchor_thread.start()

@magicgui(
    unp_dir={"label": "Fold Directory", "mode": "d"},
    output_dir={"label": "Output Directory", "mode": "d"},
    call_button="Batch Process UNPs",
)
def clean_ret_chor_labels(
    unp_dir: Path = Path(r"F:\_temp_test_data"),
    output_dir: Path = Path(r"F:\_temp_test_data"),
    unp_full_range:bool=True,
    save_clean_ret_chor:bool = True,
    bscan_use_cpu:bool = False,
    display_in_napari:bool=False,
    debug:bool=False
):
    """"""
    # TODO remove temp globals
    oct_type = "OCT"
    use_accelerator = not bscan_use_cpu

    if unp_full_range:
        imaging_range = 12.0
    else:
        imaging_range = 6.0

    if use_accelerator:
        current_device = device
    else:
        current_device = "cpu"

    file_paths = list(unp_dir.rglob("*_retchor.npz"))
    print(f"File list:\n{file_paths}\n")

    if display_in_napari:
        viewer = napari.Viewer(show=False)

    



            # clean_segmentations_thread = threading.Thread(target=save_clean_labels,kwargs={"ret_chor_labels":ret_chor_labels,"imaging_range":imaging_range,"save_clean_ret_chor":save_clean_ret_chor})
            # clean_segmentations_thread.start()

        if debug:
            print("Saving Preproc Data")

        if display_in_napari:
            pass

    print("Batch Processing Complete.")
    if display_in_napari:
        viewer.show()


if __name__ == "__main__":
    generate_enface_with_labels.show(run=True)