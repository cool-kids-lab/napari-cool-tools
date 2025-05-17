"""
"""

import sys
import math
import pathlib
import napari
import numpy as np
from magicgui import magicgui
from pathlib import Path
from tqdm import tqdm
from napari_cool_tools_io._prof_reader import prof_proc_meta
  
@magicgui(
    unp_file_path={"label": ".unp file", "mode": "r"},
    call_button="Process .unp file",
)
def save_unp_slice(
    unp_file_path: pathlib.Path = Path("D:\\John\\Yakub\\Shuibin\\Dispersion Correction\\SimpleDispersionCorrection\\14_09_04.unp"),
    start:int = 1197,
    stop:int = 1205
    ):
    """
    """
    file_name = unp_file_path.name
    folder = unp_file_path.parent
    file_type = unp_file_path.suffix
    file_out_path = folder / f"{unp_file_path.stem}_{start}_{stop}.unp"

    if file_type == ".unp":
        print(f"{file_name}\n{folder}\n{unp_file_path}\n")

        # Read the xml file
        meta = prof_proc_meta(Path(unp_file_path),".unp")

        print(meta)

        if meta is not None:
            h, w, d, bmscan, w_param, dtype, layer_type = meta

            if bmscan > 1:
                bidir = "false"
                bidir_a = "false"
            else:
                bidir = "true"
                bidir_a = "true"

            meta_params = {
                "width": w_param,
                "height": h,
                "frames": d,
                "bidir": bidir,
                "bidir_a": bidir_a,
                "bscan_width":w,
                "bmscan": bmscan
            }

            print(f"meta params {meta_params}\n")

        oct_vol_array = []

        # open file
        with open(unp_file_path,'rb',buffering=0) as byte_reader:

            data_size_bytes = 2*meta_params["width"]*meta_params["height"]
            frames_to_read = data_size_bytes*((stop+1)-start)
            print(f"stop + 1 - start ?: {(stop+1)-start}\ntype: {type(stop-start)}\n")
            
            current_idx = data_size_bytes*(start)
            byte_reader.seek(current_idx,0)
            #raw_data = np.frombuffer(byte_reader.read(frames_to_read),dtype=np.uint16)
            raw_data = byte_reader.read(frames_to_read)

            '''
            data_size_bytes = 2*meta_params["width"]*meta_params["height"]
            
            # Main OCT Volume process
            
            for frame_num in tqdm(range(start,stop+1),desc="Processing Bscans"):
            #for frame_num in range(1199,1200):
            #for frame_num in range(1199,1202):

                current_idx = data_size_bytes*(frame_num)
                byte_reader.seek(current_idx,0)
                raw_data = np.frombuffer(byte_reader.read(data_size_bytes),dtype=np.uint16)
                #temp_raw = raw_data.reshape((-1,meta_params["height"],meta_params["width"]))
                #raw = temp_raw.astype(np.uint64)
                
                # store image to oct array
                oct_vol_array.append(raw_data)
                #print("Here")

        volOCT = np.concatenate(oct_vol_array,axis=0)
        
        #print(f"OCT volume: {volOCT[1199,:,:]}\n")

        bytes_out = volOCT.tobytes()
        '''
        
        #bytes_out = raw_data.tobytes()

        with open(file_out_path,'wb',buffering=0) as byte_writer:

            byte_writer.write(raw_data)   

        print(f"{unp_file_path} file processing is finished.")
    else:
        print(f"File must be of type '.unp'. {file_name} is not the proper type.")


save_unp_slice.show(run=True)