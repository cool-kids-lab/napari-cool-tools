"""
"""
import os
import gc
import napari
import numpy as np
from enum import Enum
from tqdm import tqdm
from pathlib import Path
from magicgui import magic_factory

# Implement threading for this

@magic_factory(
        target_file={"label": "Target Directory","mode":"r","tooltip": "Directory Containing files or subdirectories with OCT .prof files to process"},
        save_dir={"label": "Save Directory","mode":"r","tooltip": "Directory Containing files or subdirectories with OCT .prof files to process"}
)
def batch_proc_profs(
    target_file:Path=Path(r"D:\JJ\Projects\Desktop_UWF_OCT\OCTA_Test\New_Hotness\2024-09-12T16_00_49_071775Z-07_00_UNP.prof"), #r"D:\Mani\To Segment\08851930-2023_08_23-13_32_03"), #r"D:\JJ\Development\OCTA\Batch_Test"), #(r"D:\Dir_to_process"),
    save_dir:Path=Path(r"D:\JJ\Projects\Desktop_UWF_OCT\OCTA_Test\New_Hotness\processed"),
    save_selected:bool=False,
    open_napari:bool=True,
    pow:float=1.0 
):
    """
    """

    viewer = napari.Viewer(show=False)
    viewer.open(target_file)
    data = viewer.layers[-1].data
    zero = data[::4]
    one = data[1::4]
    two = data[2::4]
    three = data [3::4]
    print(f"zero shape: {zero.shape}\n")
    zero_calc = zero.reshape(-1,3,zero.shape[-2],zero.shape[-1]).std(axis=1)**pow
    one_calc = one.reshape(-1,3,one.shape[-2],one.shape[-1]).std(axis=1)**pow
    two_calc = two.reshape(-1,3,two.shape[-2],two.shape[-1]).std(axis=1)**pow
    three_calc = three.reshape(-1,3,three.shape[-2],three.shape[-1]).std(axis=1)**pow
    print(f"zero_calc shape: {zero_calc.shape}\n")
    octa_fixed = np.zeros((int(data.shape[-3]/3),data.shape[-2],data.shape[-1]))
    print(f"octa_fixed shape: {octa_fixed.shape}\n")
    octa_fixed[::4] = zero_calc
    octa_fixed[1::4] = one_calc
    octa_fixed[2::4] = two_calc
    octa_fixed[3::4] = three_calc

    viewer.add_image(octa_fixed)

    del (
        data,
        zero,one,two,three,
        zero_calc,one_calc,two_calc,three_calc
    )
    gc.collect()

    if open_napari:
        viewer.show()




batch_proc_profs_gui = batch_proc_profs()
batch_proc_profs_gui.show(run=True)