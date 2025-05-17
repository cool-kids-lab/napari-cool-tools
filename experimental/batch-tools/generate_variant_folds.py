import pathlib
import xml.etree.ElementTree as ET
import os
import os.path as ospath
import subprocess
import napari
import numpy as np
import torch
#import napari_cool_tools_io
from typing import List, Dict
from pickle import HIGHEST_PROTOCOL
from itertools import combinations
from tqdm import tqdm
from napari.utils.notifications import show_info
from magicgui import magicgui
from pathlib import Path
from napari.qt.threading import thread_worker
from napari_cool_tools_io import _prof_reader, _prof_writer
from napari_cool_tools_oct_preproc._oct_preproc_utils_funcs import preproc_bscan
from napari_cool_tools_vol_proc._averaging_tools_funcs import average_per_bscan
#from napari_cool_tools_oct_preproc import _oct_preproc_utils #generate_enface_image


def combine_variant_dicts(a:Dict,b:Dict):
    """"""

    #print(f"\nDict keys:\na: {a.keys()}, b: {b.keys()}\n\n")
    out_names = a["names"]  + b["names"]
    out_images = np.concatenate([a['images'],b['images']],axis=1)
    #print(f"out_images shape: {out_images.shape}\n")
    out__labels = np.concatenate([a['masks'],b['masks']],axis=0)
    #print(f"out_images shape: {out__labels.shape}\n")
    out_dict = {"images":out_images,"masks":out__labels,"names":out_names}

    return out_dict

@magicgui(
    fold_dir={"label": "Fold Directory", "mode": "d"},
    output_dir={"label": "Output Directory", "mode": "d"},
    call_button="Generate Training Folds",
)
def generate_variant_folds(fold_dir: pathlib.Path = Path(r"D:\Mani\BScan Labels 24\Done\Pytorch_Unet\Folds"),
                        output_dir: pathlib.Path = Path(r"D:\Mani\BScan Labels 24\Done\Pytorch_Unet\Folds\Combined"),
):
    """"""
    
    fold_names = list(fold_dir.glob("*.pt"))
    print(f"Folds to combine:\n{fold_names}\n")

    folds = []

    for fold_path in tqdm(fold_names,desc="Loading Folds"):

        print(F"{fold_path}\n")
        folds.append(torch.load(fold_path))

    print(f"Folds have been loaded\n")

    idx_list = range(len(folds))

    #print(f"{idx_list}\n")

    combs = list(combinations(idx_list,4))

    print(f"Possible combinations:\n{combs}\n")

    for i,comb in tqdm(enumerate(combs),desc="Combinations"):

        comb_dict = {}
        train_file_name = f"train_folds_{i}"
        test_fold_idx = np.setdiff1d(idx_list,comb)[0]
        test_fold_path = fold_names[test_fold_idx]
        #print(f"\n\nAssociated test fold:\nidx: {test_fold_idx}\npath: {test_fold_path}\n\n")
        
        for i,f in tqdm(enumerate(comb),desc="Fold in Combination"):

            #print(f"\n\nFold number {i}: {folds[f]['names'][0]}\n\n")

            if i == 0:
                comb_dict = folds[i]
            else:
                comb_dict =combine_variant_dicts(comb_dict,folds[i])
        
        comb_dict["test_fold_path"] = test_fold_path

        print(f"comb_dict image shape: {comb_dict['images'].shape}\n")
        print(f"comb_dict label shape: {comb_dict['masks'].shape}\n")
        print(f"comb_dict name length: {len(comb_dict['names'])}\n")
        print(f"comb_dict associated testfold: {test_fold_path}\n")

        torch.save(comb_dict,Path(f"{output_dir}\{train_file_name}.pt"),pickle_protocol=HIGHEST_PROTOCOL)

        print("Training fold saved\n")


#view_bscan_variants.changed.connect(print)
generate_variant_folds.show(run=True)