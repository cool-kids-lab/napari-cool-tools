"""
Pytorch Batch Processing of UNP files
"""
from pathlib import Path
from typing import Literal

# from qtpy.QtWidgets import QApplication
import numpy as np
import napari
from magicgui import magicgui

from napari_cool_tools_io import unp_meta
from napari_cool_tools_io._unp_reader import unp_batch_proc_meta
from napari_cool_tools_io.process_unp import process_unp, set_dispersion_coefficients_torch
import torch
from tqdm import tqdm

@magicgui(
    unp_dir={"label": "Fold Directory", "mode": "d"},
    output_dir={"label": "Output Directory", "mode": "d"},
    call_button="Batch Process UNPs",
)
def generate_enface_with_labels(
    unp_dir: Path = Path(r"F:\_temp_test_data"),
    output_dir: Path = Path(r"F:\_temp_test_data"),
    unp_dc_subtract:bool=True,
    unp_desine:bool=True,
    unp_double_side:bool=True,
    unp_full_range:bool=False,
    unp_log_scale:bool=False,
    unp_max_projection:bool=False,
    unp_disp_coeff_range:float=100.0,
    unp_auto_dispersion_correction:bool=True,
    unp_flip_disp_coeffs:bool=True,
    display_in_napari:bool=False,
):
    """"""
    file_paths = list(unp_dir.rglob("*.unp"))
    print(f"File list:\n{file_paths}\n")
    print("Batch Processing Complete.")

    if display_in_napari:
        viewer = napari.Viewer(show=False)

    for unp_file in file_paths:
        meta = unp_batch_proc_meta(unp_file)

        # apply user settings
        meta.dcSubtract = unp_dc_subtract
        meta.desine = unp_desine
        meta.double_side = unp_double_side
        meta.full_range = unp_full_range
        meta.log_scale = unp_log_scale
        meta.max_projection = unp_max_projection

        print(f"{unp_file} metadata:\n{meta}\n")

        processed_unp = process_unp(unp_file_path=unp_file,meta=meta,auto_dispersion=unp_auto_dispersion_correction,flip_coeffs=unp_flip_disp_coeffs)
        #dispersion_coefficient_range = np.arange(unp_disp_coeff_min,unp_disp_coef_max+1,1) 
        #processed_unp = set_dispersion_coefficients_torch(torch.tensor(processed_unp),maxDispOrders=3,coefRange=unp_disp_coef_max)
        processed_unp = processed_unp

        if display_in_napari:
            viewer.add_image(processed_unp,name=unp_file.stem)

    if display_in_napari:
        viewer.show()


if __name__ == "__main__":
    generate_enface_with_labels.show(run=True)