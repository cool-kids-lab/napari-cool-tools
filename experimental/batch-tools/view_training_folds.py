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

@magicgui(
    data={"label": "B-scan variations", "mode": "r"},
    call_button="Display B-scan Variants",
)
def view_training_folds(data: pathlib.Path = Path(r"D:\Mani\BScan Labels 24\Done\Pytorch_Unet\Folds\Combined\output.pt")):
    """"""
    in_dict = torch.load(data)

    viewer = napari.Viewer()

    images = in_dict["images"]
    labels = in_dict["masks"]
    names = in_dict["names"]
    test_fold_path = in_dict["test_fold_path"]

    print(f"images shape: {images.shape}, len names: {len(names)}\n")
    print(f"Associated test fold: {test_fold_path}\n")

    for image in tqdm(images,desc="Load B-scan variations"):
        viewer.add_image(image)

    viewer.add_labels(labels)

#view_bscan_variants.changed.connect(print)
view_training_folds.show(run=True)