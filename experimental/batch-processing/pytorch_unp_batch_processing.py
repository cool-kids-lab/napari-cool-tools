"""
Pytorch Batch Processing of UNP files
"""
import gc
from pathlib import Path
import threading
# from typing import Literal

# from qtpy.QtWidgets import QApplication
# import numpy as np
import napari
from magicgui import magicgui
from tqdm import tqdm

from napari_cool_tools_img_proc import DType
from napari_cool_tools_img_proc._equalization_funcs import init_bscan_preproc_pt
from napari_cool_tools_io import device
from napari_cool_tools_io._npz_writer import save_npz
from napari_cool_tools_io._unp_reader import unp_batch_proc_meta
from napari_cool_tools_io.process_unp import process_unp
from napari_cool_tools_oct_preproc._oct_preproc_func import desine
from napari_cool_tools_segmentation._label_cleaning_funcs import find_indices_after
from napari_cool_tools_segmentation._segmentation_funcs import  bscan_onnx_seg_func, bscan_onnx_deconj_func
from napari_cool_tools_vol_proc._averaging_tools_funcs import average_per_bscan,average_bscans,average_bscans_torch
import torch

@magicgui(
    unp_dir={"label": "Fold Directory", "mode": "d"},
    output_dir={"label": "Output Directory", "mode": "d"},
    call_button="Batch Process UNPs",
)
def batch_proc_unps(
    # unp_dir: Path = Path(r"F:\_temp_test_data"),
    # output_dir: Path = Path(r"F:\_temp_test_data"),
    # unp_dir: Path = Path(r"E:\_rebatch_conjugate_test_12_01_2025\UNPs"),
    # output_dir: Path = Path(r"E:\_rebatch_conjugate_test_12_01_2025\output3"),
    unp_dir: Path = Path(r"\\192.168.1.3\coolkid\Beth Roti\Ridge Project\Ridge_UNPs_To_Process"),
    output_dir: Path = Path(r"\\192.168.1.3\coolkid\Beth Roti\Ridge Height Output"),
    ret_chor_suffix:str = "ret_chor_seg",
    structure_suffix:str = "structure",
    unp_dc_subtract:bool=True,
    unp_desine:bool=True,
    unp_double_side:bool=True,
    unp_full_range:bool=True,
    unp_log_scale:bool=False,
    unp_max_projection:bool=False,
    unp_disp_coeff_range:float=100.0,
    unp_auto_dispersion_correction:bool=True,
    unp_flip_disp_coeffs:bool=True,
    deconjugate:bool = True,
    segmentation_bscan_batch_size:int=32,
    # clean_segmentations:bool = False,
    correct_choroid_artefact:bool = False,
    save_structure:bool = True,
    save_retchor:bool = True,
    # save_clean_ret_chor:bool = True,
    overwrite:bool = False,
    bscan_use_cpu:bool = False,
    display_in_napari:bool=False,
    verbose:bool=False,
    debug:bool=False,
):
    """"""
    # TODO remove temp globals
    # oct_type = "OCT"
    use_accelerator = not bscan_use_cpu

    # if unp_full_range:
    #     imaging_range = 12.0
    # else:
    #     imaging_range = 6.0

    if use_accelerator:
        current_device = device
    else:
        current_device = "cpu"

    file_paths = list(unp_dir.rglob("*.unp"))
    print(f"File list:\n{file_paths}\n")

    if save_structure:
        existing_structure_paths = list(output_dir.rglob(f"*{structure_suffix}.npz"))
    if save_retchor:
        existing_ret_chor_paths = list(output_dir.rglob(f"*{ret_chor_suffix}.npz"))

    if display_in_napari:
        viewer = napari.Viewer(show=False)

    unp_pbar = tqdm(file_paths[:])
    for unp_file in unp_pbar:
        print(unp_pbar.set_description(f"Processing {unp_file.stem}"))

        if not overwrite:
            structure_is_present = any(unp_file.stem in str(path) for path in existing_structure_paths)
            ret_chor_is_present = any(unp_file.stem in str(path) for path in existing_ret_chor_paths)

            if verbose:
                print(f"structure present: {structure_is_present}, ret_chor: {ret_chor_is_present}\n")

            if structure_is_present and ret_chor_is_present:
                #if verbose:
                print(f"{unp_file} has already been processed.")
                continue

        meta = unp_batch_proc_meta(unp_file)
        name = unp_file.stem

        # apply user settings
        meta.dcSubtract = unp_dc_subtract
        meta.desine = unp_desine
        meta.double_side = unp_double_side
        meta.full_range = unp_full_range
        meta.log_scale = unp_log_scale
        meta.max_projection = unp_max_projection
        meta.coefRange = unp_disp_coeff_range

        if verbose:
            print(f"{unp_file} metadata:\n{meta}\n")
        try:
            processed_unp = process_unp(unp_file_path=unp_file,meta=meta,auto_dispersion=unp_auto_dispersion_correction,flip_coeffs=unp_flip_disp_coeffs)
            raw_unp = processed_unp
        except Exception as e:
            print(f"An unexpected error occured while processing file:\n{unp_file}")
            print(f"Error Type: {type(e).__name__}")
            print(f"Full Repr: {repr(e)}")
            print(f"Skipping {unp_file}\n")
            continue

        # deconjugate prof
        if deconjugate:
            if debug:
                print("Deconjugating Data")

        processed_unp,_ = bscan_onnx_deconj_func(processed_unp,verbose=False)

        if debug:
            print("Deconjugating complete")

        # if oct_type == "OCT":
        if meta.bmscan == 1:
            processed_unp = average_per_bscan(processed_unp,scans_per_avg=3,axis=0,trim=False)
        elif meta.bmscan > 1:
            #processed_unp = average_bscans_torch(processed_unp,scans_per_avg=meta.bmscan)
            processed_unp = average_bscans(processed_unp,scans_per_avg=meta.bmscan)
            #processed_unp = average_bscans(processed_unp,scans_per_avg=3)
        else:
            print(f"Invalid bmscan value {meta.bmscan} in configuration")
            print(f"Skipping {unp_file}\n")

        pre_seg_proc_unp = init_bscan_preproc_pt(processed_unp,dtype=DType.NP_FLOAT32,use_accelerator=use_accelerator,numpy_out=True,verbose=False)


        # segment prof bscans
        if debug:
            print("Segmenting Data")

        ret_chor_labels,_ = bscan_onnx_seg_func(pre_seg_proc_unp,batch_size=segmentation_bscan_batch_size,use_cpu=bscan_use_cpu,verbose=False)[0]

        if not unp_desine:
            if verbose:
                print(f"retchor min,max,dtype,size: {ret_chor_labels.min()},{ret_chor_labels.max()},{ret_chor_labels.dtype},{ret_chor_labels.nbytes}\n")
            
            ret_chor_labels = torch.tensor(ret_chor_labels,device="cpu",dtype=torch.float32)
            
            if verbose:
                print(f"retchor min,max,dtype,size: {ret_chor_labels.min()},{ret_chor_labels.max()},{ret_chor_labels.dtype},{ret_chor_labels.nbytes}\n")
            ret_chor_labels = ret_chor_labels.to(current_device)
            ret_chor_labels = desine(ret_chor_labels,mode="nearest",transpose=False)
            ret_chor_labels = ret_chor_labels.cpu()
            gc.collect()
            torch.cuda.empty_cache()

            if correct_choroid_artefact:
                ret_chor_labels = ret_chor_labels.to(current_device)
                choroid_artefact_indices = find_indices_after(ret_chor_labels,val_a=1,val_b=2,axis=1,device=current_device)
                ret_chor_labels = ret_chor_labels.cpu().numpy().astype("uint8")
                gc.collect()
                torch.cuda.empty_cache()
                
                if verbose:
                    print([(arr.min(),arr.max()) for arr in choroid_artefact_indices]) 
                    print(f"ret_chor_labels shape: {ret_chor_labels.shape}, ret_chor_labels dtype: {ret_chor_labels.dtype}\n")
                ret_chor_labels[*choroid_artefact_indices] = 0
            else:
                ret_chor_labels = ret_chor_labels.numpy().astype("uint8")

            if verbose:
                print(f"pre_seg_proc_unp min,max,dtype,size: {pre_seg_proc_unp.min()},{pre_seg_proc_unp.max()},{pre_seg_proc_unp.dtype},{pre_seg_proc_unp.nbytes}\n")
            pre_seg_proc_unp = desine(torch.tensor(pre_seg_proc_unp,device=current_device,dtype=torch.float32),mode="bilinear",transpose=False)
            pre_seg_proc_unp = pre_seg_proc_unp.cpu().numpy()
            gc.collect()
            torch.cuda.empty_cache()

        structure_name = f"{name}_{structure_suffix}"
        # save in thread
        if (save_structure and not structure_is_present) | overwrite:
            structure_path = output_dir/f"{structure_name}.npz"
            if meta.motor_position is not None:    
                save_dict = {"name":structure_name,"layer_type":'image',"shape":pre_seg_proc_unp.shape,"motor_position":meta.motor_position}
            else:
                save_dict = {"name":structure_name,"layer_type":'image',"shape":pre_seg_proc_unp.shape}
            save_structure_thread = threading.Thread(target=save_npz,kwargs={"path":structure_path,"data":pre_seg_proc_unp,"save_dict":save_dict,"verbose":False})
            save_structure_thread.start()

        ret_chor_name = f"{name}_{ret_chor_suffix}"
        # save in thread
        if (save_retchor and not ret_chor_is_present) | overwrite:
            ret_chor_path = output_dir/f"{ret_chor_name}.npz"
            if meta.motor_position is not None:
                save_dict = {"name":ret_chor_name,"layer_type":'labels',"shape":ret_chor_labels.shape,"motor_position":meta.motor_position}
            else:
                save_dict = {"name":ret_chor_name,"layer_type":'labels',"shape":ret_chor_labels.shape}
            save_retchor_thread = threading.Thread(target=save_npz,kwargs={"path":ret_chor_path,"data":ret_chor_labels,"save_dict":save_dict,"verbose":False})
            save_retchor_thread.start()

        if debug:
            print("Saving Preproc Data")

        if display_in_napari:
            viewer.add_image(raw_unp,name=name)
            viewer.add_image(pre_seg_proc_unp,name=f"{name}_pre_seg")
            #viewer.add_image(output_unp,name=f"{name}_out")
            viewer.add_labels(ret_chor_labels,name=f"{ret_chor_name}")

    print("Batch Processing Complete.")
    if display_in_napari:
        viewer.show()

if __name__ == "__main__":
    batch_proc_unps.native.setWindowTitle("UNP Batch Processing")
    batch_proc_unps.show(run=True)