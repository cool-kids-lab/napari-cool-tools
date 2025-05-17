import pathlib
import xml.etree.ElementTree as ET
import os
import os.path as ospath
import subprocess
import napari
import numpy as np
import torch
#import napari_cool_tools_io
from pickle import HIGHEST_PROTOCOL
from tqdm import tqdm
from napari.utils.notifications import show_info
from magicgui import magicgui
from pathlib import Path
from napari.qt.threading import thread_worker
from napari_cool_tools_io import _prof_reader, _prof_writer
from napari_cool_tools_oct_preproc._oct_preproc_utils_funcs import preproc_bscan
from napari_cool_tools_vol_proc._averaging_tools_funcs import average_per_bscan
#from napari_cool_tools_oct_preproc import _oct_preproc_utils #generate_enface_image

from jj_nn_framework.nn_transforms import (NCHWFormat,PadToTargetM)

DEVICE = 'cpu'

pttm_params2 = {
        'h': 992, #256 512, 992, 864, 800,
        'w': 800, #224 416, 800, 864, 800,
        'X_data_format': 'NCHW', #'HW','NHW','NCHW',
        'y_data_format': 'NCHW', #'HW','NHW', 'NCHW',
        'mode': 'constant',
        'value': None,
        'device': DEVICE
    }

ncwh = NCHWFormat()

pttm = PadToTargetM(**pttm_params2)

same_shape = torch.nn.Sequential(
    ncwh,
    pttm,
)


@magicgui(
    image_volume={"label": "Image File", "mode": "r"},
    label_volume={"label": "Label File", "mode": "r"},
    output_dir={"label": "Output Directory", "mode": "d"},
    vol_slice={"label": "Image/Label Slice"},
    call_button="Generate B-scan Variants",
)
def generate_bscan_variants(
    image_volume: pathlib.Path = Path(r"D:\Mani\BScan Labels 24\Done\231\OCT 899 averaged x 3 08905231.mat"),
    label_volume: pathlib.Path = Path(r"D:\Mani\BScan Labels 24\Done\231\08906231_LabelsTemp.mat"),
    output_dir: pathlib.Path = Path(r"D:\Mani\BScan Labels 24\Done\Pytorch_Unet\Base_Data"),
    output_filename: str = "output.pt",
    use_slice: bool = True,
    vol_slice:slice=slice(None),
    use_gpu: bool = True
    ):
    """
    """
    if use_gpu:
        processor = 'cuda'
    else:
        processor = 'cpu'

    # do something with the folders
    print(f"Image file: {image_volume}, Label file: {label_volume}\n")

    # create headless napari viewer
    viewer = napari.Viewer(show=False)
    #viewer = napari.Viewer()
    #viewer = napari.components.ViewerModel()

    viewer.open(image_volume,plugin="napari-cool-tools-io")
    viewer.open(label_volume,plugin="napari-cool-tools-io")

    names = []
    variations = []

    full_image = viewer.layers[0].data
    full_labels = viewer.layers[-1].data

    if use_slice:
        edit_slice = slice(vol_slice.start-2,vol_slice.stop+2,vol_slice.step)
        edit = full_image[edit_slice]
        image = edit[2:-2]
    else:
        image = full_image
        

    img_name = viewer.layers[0].name
    variations.append(image)
    names.append(img_name)

    if use_slice:
        labels = full_labels[vol_slice]
    else:
        labels = full_labels
    lbl_name = viewer.layers[-1].name
    
    
    # generate sliding window average bscans of 3 and 5
    
    if use_slice:
        #avg3 = average_per_bscan(edit[1:-1],3,0,True)
        avg5 = average_per_bscan(edit,5,0,True)
    else:
        #avg3 = average_per_bscan(image,3,0,True)
        avg5 = average_per_bscan(image,5,0,True)

    #avg3_name = f"{img_name}_3_per"
    avg5_name = f"{img_name}_5_per"

    #variations.append(avg3)
    variations.append(avg5)

    #names.append(avg3_name)
    names.append(avg5_name)
    

    # generate preproc variations for image, avg3, avg5
    proc = preproc_bscan(image,ascan_corr=False,vol_proc=False,processor=processor)
    #proc = preproc_bscan(image,ascan_corr=False,Vol_proc=False,processor=processor)
    #avg3_proc = preproc_bscan(avg3,ascan_corr=False,Vol_proc=False,processor=processor)
    #avg5_proc = preproc_bscan(avg5,ascan_corr=False,Vol_proc=False,processor=processor)
    proc_name = f"{img_name}_proc"
    #avg3_proc_name = f"{avg3_name}_proc"
    #avg5_proc_name = f"{avg5_name}_proc"

    variations.append(proc)
    #variations.append(avg3_proc)
    #variations.append(avg5_proc)

    names.append(proc_name)
    #names.append(avg3_proc_name)
    #names.append(avg5_proc_name)

    # generate vol preproc variations for image, avg3, avg5
    vol_proc = preproc_bscan(image,ascan_corr=False,vol_proc=True,processor=processor)
    #vol_proc = preproc_bscan(image,ascan_corr=False,Vol_proc=True,processor=processor)
    #avg3_vol_proc = preproc_bscan(avg3,ascan_corr=False,Vol_proc=True,processor=processor)
    avg5_vol_proc = preproc_bscan(avg5,ascan_corr=False,vol_proc=True,processor=processor)
    #avg5_vol_proc = preproc_bscan(avg5,ascan_corr=False,Vol_proc=True,processor=processor)
    vol_proc_name = f"{img_name}_vol_proc"
    #avg3_vol_proc_name = f"{avg3_name}_vol_proc"
    avg5_vol_proc_name = f"{avg5_name}_vol_proc"

    variations.append(vol_proc)
    #variations.append(avg3_vol_proc)
    variations.append(avg5_vol_proc)

    names.append(vol_proc_name)
    #names.append(avg3_vol_proc_name)
    names.append(avg5_vol_proc_name)

    img_var = np.stack(variations,axis=0)
    names.append(lbl_name)

    out_dict = {}

    # out_dict["images"] = img_var
    # out_dict["masks"] = labels
    # out_dict["names"] = names

    img_var_t, labels_t = torch.tensor(img_var.copy()), torch.tensor(labels.copy()) #,dtype=torch.uint8)
    img_var_t, labels_t = same_shape((img_var_t,labels_t))

    out_dict["images"] = img_var_t #torch.Tensor(img_var.copy())
    out_dict["labels"] = labels_t.squeeze().to(torch.float16) #(labels_t.squeeze()*2).to(torch.uint8) #torch.Tensor(labels.copy())
    out_dict["meta"] = names

    torch.save(out_dict,Path(f"{output_dir}\{output_filename}"),pickle_protocol=HIGHEST_PROTOCOL)

    print(f"B-scan variations have been saved\n")

    '''
    for i,data in tqdm(enumerate(variations),desc="Adding layers to Napari Viewer"):
        if i > 0:
            viewer.add_image(data,name=names[i])
            print(f"\nAdding layer {names[i]} to Napari viewer.\n")

    print(f"Layer names: {names}\n")
    '''

#generate_bscan_variants.changed.connect(print)
generate_bscan_variants.show(run=True)