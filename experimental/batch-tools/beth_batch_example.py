"""
"""

from pathlib import Path

from napari_cool_tools_img_proc._equalization_funcs import init_bscan_preproc,DTYPE


image_path = Path("")

init_bscan_preproc( num_std=16,max_intensity=1.0,dtype=DTYPE.NP_FLOAT)