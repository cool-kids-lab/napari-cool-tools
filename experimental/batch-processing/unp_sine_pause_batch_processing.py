
from multiprocessing import Pool
from pathlib import Path
from typing import Literal
import torch

# from qtpy.QtWidgets import QApplication
import numpy as np
import napari
from magicgui import magicgui

from napari_cool_tools_io._unp_reader import unp_batch_proc_meta, process_unp_sine_pause
from napari_cool_tools_io import unp_meta
import tifffile
from helper_function import optical_flow_registration, phase_correlation_registration

from skimage.transform import resize
import time

from napari_cool_tools_registration._bidirectional_ascan_registration_funcs import auto_bidirectional_ascan_registration

@magicgui(
    unp_dir={"label": "UNP Directory", "mode": "d"},
    output_dir={"label": "Output Directory", "mode": "d"},
    desine={"label": "Desine", "widget_type": "CheckBox", "value": False}
)
def batch_proc_unps(
    
    unp_dir: Path = Path(r"Z:\Haoshen\super resolution\original\2026-06-17"),
    output_dir: Path = Path(r"C:\Users\TEAMROP\Desktop\Test"),
    # unp_dir: Path = Path(r"."),
    # output_dir: Path = Path(r"."),
    desine: bool = False
):

    print(f"Processing UNPs in directory: {unp_dir}")
    print(f"Output will be saved to: {output_dir}")

    #list all unp files in the directory
    unp_files = list(unp_dir.glob("**/*.unp"))

    print("total unp files")
    print(len(unp_files))

    #return if unp_files is empty or cancelled (None)
    if not unp_files:
        print("No UNP files found in the directory.")
        return

    start_time = time.time()

    for file_num, unp_file in enumerate(unp_files):
        print(f"Processing UNP Num: {file_num}")
        print(f"Processing UNP file: {unp_file}")

        #read metadata from unp file
        meta = unp_meta()
        meta = unp_batch_proc_meta(unp_file)
        print(meta)

        if meta is None:
            print(f"Failed to read metadata for {unp_file}, skipping.")
            continue

        if meta.pattern != "Sine_Pause":
            print(f"UNP file {unp_file} does not match pattern 'Sine_Pause', skipping.")
            continue

        meta.desine = desine

        indices = meta.sine_frame_indices
        pause_index = indices[0::2]

        hires_ratio = meta.sine_hires_ratio
        hires_d = (indices[1] - indices[0])/hires_ratio
        hires_d = int(hires_d)

        print(f"hires_d: {hires_d}, hires_ratio: {hires_ratio}")

        if hires_d != 6:
            print(f"Warning: Expected hires_d to be 6, but got {hires_d}. Skip processing.")
            continue

        #read unp file
        low_res, high_res = process_unp_sine_pause(Path(unp_file), meta, include_hires_in_lowres=False,auto_dispersion=True)


        #AScan Registration for the low_res
        high_res = auto_bidirectional_ascan_registration(high_res,
                                reference_frame_index = 0, # the index of the reference frame in the volume to be registered against
                                bmscan = 1, # whether the volume is a B-M scan (True) or a B-scan (False)
                                init_coeffs = [0.0, 0.0, 0.0, 0.0], # initial coefficients for the polynomial unwarp must be size of 4
                                ranges = 20,
                                step_size = 1.0,
                                search_c0 = False,
                                search_c1 = True,
                                search_c2 = False,
                                search_c3 = False,
                                mode = "bilinear",
                                dual_edge = True,
                                inverse = False, #inverse the coefficient directions
                                flipAB = False, #flip the AA and BB interleave indices
                                double_side = True, #whether the volume is double-sided (alternating) B-M scan acquisition
                                frequency_domain = True,
        )

        low_res_pre = np.zeros((len(pause_index),low_res.shape[1],low_res.shape[2]),dtype=low_res.dtype)#fill it with empty frame
        low_res_post = np.zeros((len(pause_index),low_res.shape[1],low_res.shape[2]),dtype=low_res.dtype)#fill it with empty frame

        for i in range(len(pause_index)):
            idx = pause_index[i] - i*hires_d*hires_ratio + i

            low_res_pre[i] = low_res[idx-1]#get the low res frame at the pause index
            low_res_post[i] = low_res[idx]#get the low res frame at the pause index

            temp_frame = np.zeros_like(low_res[0])#fill it with empty frame
            temp_frame = np.expand_dims(temp_frame, axis=0)
            low_res = np.concatenate((low_res[:idx], temp_frame, low_res[idx:]), axis=0)#this will append the high-res frame to the low-res volume at the correct index


        print("Registering high res frames using phase_correlation_registration...")

        #prepare tasks for parallel processing of optical flow registration
        tasks = []

        for i in range(len(pause_index)):#5 pause index location
            ref = high_res[i*hires_d+1].copy()  # reference frame is the second frame

            frame = high_res[i*hires_d].copy() #take the first frame as moving frame
            #TODO remove the first 250 pixels
            row_coord, col_coord = np.meshgrid(np.arange(ref.shape[0]), np.arange(ref.shape[1]), indexing="ij")
            tasks.append((ref, frame, row_coord, col_coord))

            for j in range(hires_d-2):#4
                frame = high_res[i*hires_d+2+j].copy()  # Start from the third frame
                row_coord, col_coord = np.meshgrid(np.arange(ref.shape[0]), np.arange(ref.shape[1]), indexing="ij")
                tasks.append((ref, frame, row_coord, col_coord))

        # # Run registrations in parallel
        with Pool(processes=16) as pool:
            # results = pool.starmap(optical_flow_registration, tasks)
            results = pool.starmap(phase_correlation_registration, tasks)

        #rearrange the results to match the original high res frame order
        high_res_reg = np.zeros((len(pause_index)*(hires_d),high_res.shape[1],high_res.shape[2]),dtype=high_res.dtype)#fill it with empty frame
        for i in range(len(pause_index)):#5

            high_res_reg[i*hires_d+1] = high_res[i*hires_d+1].copy()#the second frame is the reference frame
            high_res_reg[i*hires_d] = results[i*(hires_d-1)] #results size is 25 (every 5)

            for j in range(hires_d-2):#4
                idx = i*(hires_d-1) + j
                high_res_reg[i*hires_d+2+j] = results[idx+1]


        # #average the registered high res frames for every 5 frame
        # high_res_ave_reg = np.zeros((len(pause_index),high_res.shape[1],high_res.shape[2]),dtype=high_res.dtype)#fill it with empty frame

        # for i in range(len(pause_index)):
        #     high_res_ave_reg[i] = np.mean(high_res_reg[i*(hires_d-1):(i+1)*(hires_d-1)], axis=0)

        # print("Finished registering high res frames using phase correlation.")

        # curr_time = time.time()
        # print(f"Time elapsed for phase correlation registration: {curr_time - start_time} seconds")
        # start_time = curr_time


        # print("Registering low res frames to high res frames using optical flow...")

        # #resize the low res to match the high res size in the x and y dimensions
        # low_res_pre = resize(low_res_pre, (low_res_pre.shape[0], high_res.shape[1], high_res.shape[2]), anti_aliasing=True)
        # low_res_post = resize(low_res_post, (low_res_post.shape[0], high_res.shape[1], high_res.shape[2]), anti_aliasing=True)

        # #prepare the tasks
        # tasks = []

        # for i in range(len(pause_index)):
        #     low_res_frame_pre = low_res_pre[i].copy()  # low res frame
        #     low_res_frame_post = low_res_post[i].copy()  # low res frame
        #     high_res_frame_pre = high_res_ave_reg[i].copy()  # high res frame
        #     high_res_frame_post = high_res_ave_reg[i].copy()  # high res frame

        #     row_coord, col_coord = np.meshgrid(np.arange(high_res_frame_pre.shape[0]), np.arange(high_res_frame_pre.shape[1]), indexing="ij")
        #     tasks.append((low_res_frame_pre, high_res_frame_pre, row_coord, col_coord))

        #     row_coord, col_coord = np.meshgrid(np.arange(high_res_frame_post.shape[0]), np.arange(high_res_frame_post.shape[1]), indexing="ij")
        #     tasks.append((low_res_frame_post, high_res_frame_post, row_coord, col_coord))

        # # # Run registrations in parallel
        # with Pool(processes=8) as pool:
        #     # results = pool.starmap(optical_flow_registration, tasks)
        #     results = pool.starmap(phase_correlation_registration, tasks)



        # registered_hi_res_pre =  np.zeros_like(high_res_ave_reg)
        # registered_hi_res_post =  np.zeros_like(high_res_ave_reg)

        # for i in range(len(pause_index)):
        #     registered_hi_res_pre[i] = results[i*2]
        #     registered_hi_res_post[i] = results[i*2+1]

        # print("Finished registering low res frames to high res frames using optical flow.")

        #saving file

        # # #save low res and high res as tiff files
        low_res_output_path_pre = output_dir / f"{unp_file.stem}_{file_num}_low_res_pre.tiff"
        low_res_output_path_post = output_dir / f"{unp_file.stem}_{file_num}_low_res_post.tiff"
        high_res_output_path = output_dir / f"{unp_file.stem}_high_res.tiff"
        # high_res_output_ave_path = output_dir / f"{unp_file.stem}_{file_num}_high_res_ave.tiff"
        # high_res_output_ave_reg_path_pre = output_dir / f"{unp_file.stem}_{file_num}_high_res_ave_reg_pre.tiff"
        # high_res_output_ave_reg_path_post = output_dir / f"{unp_file.stem}_{file_num}_high_res_ave_reg_post.tiff"

        tifffile.imwrite(low_res_output_path_pre, low_res_pre.astype(np.float32))
        tifffile.imwrite(low_res_output_path_post, low_res_post.astype(np.float32))
        tifffile.imwrite(high_res_output_path, high_res_reg.astype(np.float32))
        # tifffile.imwrite(high_res_output_ave_path, high_res_ave_reg.astype(np.float32))
        # tifffile.imwrite(high_res_output_ave_reg_path_pre, registered_hi_res_pre.astype(np.float32))
        # tifffile.imwrite(high_res_output_ave_reg_path_post, registered_hi_res_post.astype(np.float32))

        print("Finished saving registered frames.")


if __name__ == "__main__":
    batch_proc_unps.native.setWindowTitle("UNP Batch Processing")
    batch_proc_unps.show(run=True)

    