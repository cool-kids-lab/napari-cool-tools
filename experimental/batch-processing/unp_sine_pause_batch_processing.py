
from pathlib import Path
from typing import Literal

# from qtpy.QtWidgets import QApplication
import numpy as np
import napari
from magicgui import magicgui

from napari_cool_tools_io._unp_reader import unp_batch_proc_meta
from napari_cool_tools_io import unp_meta
import tifffile
from helper_function import process_unp_sine_pause_batch_haoshen


@magicgui(
    unp_dir={"label": "UNP Directory", "mode": "d"},
    output_dir={"label": "Output Directory", "mode": "d"},
    desine={"label": "Desine", "widget_type": "CheckBox", "value": True}
)
def batch_proc_unps(
    # unp_dir: Path = Path(r"F:\_temp_test_data"),
    # output_dir: Path = Path(r"F:\_temp_test_data"),
    # unp_dir: Path = Path(r"E:\_rebatch_conjugate_test_12_01_2025\UNPs"),
    # output_dir: Path = Path(r"E:\_rebatch_conjugate_test_12_01_2025\output3"),
    unp_dir: Path = Path(r"."),
    output_dir: Path = Path(r"."),
    desine: bool = True
):

    print(f"Processing UNPs in directory: {unp_dir}")
    print(f"Output will be saved to: {output_dir}")

    #list all unp files in the directory
    unp_files = list(unp_dir.glob("*.unp"))

    #return if unp_files is empty or cancelled (None)
    if not unp_files:
        print("No UNP files found in the directory.")
        return

    for unp_file in unp_files:
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

        #read unp file
        low_res_pre, low_res_post, high_res = process_unp_sine_pause_batch_haoshen(Path(unp_file), meta)

        #save low res and high res as tiff files
        low_res_output_path = output_dir / f"{unp_file.stem}_low_res_pre.tiff"
        low_res_output_path_post = output_dir / f"{unp_file.stem}_low_res_post.tiff"
        high_res_output_path = output_dir / f"{unp_file.stem}_high_res.tiff"

        tifffile.imwrite(low_res_output_path, low_res_pre.astype(np.float32))
        tifffile.imwrite(low_res_output_path_post, low_res_post.astype(np.float32))
        tifffile.imwrite(high_res_output_path, high_res.astype(np.float32))


if __name__ == "__main__":
    batch_proc_unps.native.setWindowTitle("UNP Batch Processing")
    batch_proc_unps.show(run=True)