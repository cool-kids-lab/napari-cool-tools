
from pathlib import Path
from typing import Literal

# from qtpy.QtWidgets import QApplication
import numpy as np
import napari
from magicgui import magicgui

from napari_cool_tools_io._unp_reader import unp_batch_proc_meta
from napari_cool_tools_io import unp_meta
import tifffile
from napari_cool_tools_io.process_unp import process_unp_sine_pause, process_unp


@magicgui(
    unp_dir={"label": "UNP Directory", "mode": "d"},
    output_dir={"label": "Output Directory", "mode": "d"},
    desine={"label": "Desine", "widget_type": "CheckBox", "value": True}
)
def batch_proc_unps(
    unp_dir: Path = Path(r"."),
    output_dir: Path = Path(r"."),
    desine: bool = True
):

    print(f"Processing UNPs in directory: {unp_dir}")
    print(f"Output will be saved to: {output_dir}")

    #list all unp files in the directory
    unp_files = list(unp_dir.glob("**/*.unp"))

    #return if unp_files is empty or cancelled (None)
    if not unp_files:
        print("No UNP files found in the directory.")
        return

    for i, unp_file in enumerate(unp_files):
        
        print(f"Processing UNP file: {unp_file}")

        #read metadata from unp file
        meta = unp_meta()
        meta = unp_batch_proc_meta(unp_file)
        print(meta)

        if meta is None:
            print(f"Failed to read metadata for {unp_file}, skipping.")
            continue

        #include desine option in the processing
        meta.desine = desine 

        if meta.pattern == "Sine_Pause":
            volume, _ = process_unp_sine_pause(Path(unp_file), meta)
        else:
            volume = process_unp(Path(unp_file), meta)

        #save volume as tiff file in the output directory with the same name as the unp file
        #also include the parent directory name of the unp file in the output directory
        relative_path = unp_file.relative_to(unp_dir).parent

        #make sure the output directory exists
        (output_dir / relative_path).mkdir(parents=True, exist_ok=True)

        volume_path = output_dir / relative_path / f"{i}_processed.tiff"

        tifffile.imwrite(volume_path, volume.astype(np.float32))


if __name__ == "__main__":
    batch_proc_unps.native.setWindowTitle("UNP Batch Processing")
    batch_proc_unps.show(run=True)