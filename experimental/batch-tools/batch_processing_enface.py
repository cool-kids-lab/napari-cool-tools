import configparser
from pathlib import Path
from typing import Literal
import os.path as ospath
import numpy as np
import napari
from magicgui import magicgui
import xml.etree.ElementTree as ET
from tqdm import tqdm
from napari_cool_tools_io.process_unp import process_unp, process_unp_sine_pause
from napari_cool_tools_oct_preproc._oct_preproc_func import desine
from napari_cool_tools_io import unp_meta
import torch
import tifffile
from scipy.ndimage import uniform_filter1d, gaussian_filter1d

# class unp_meta:
#     width: int = 0
#     height: int = 0
#     depth: int = 0
#     bmscan: int = 0
#     vista: int = 0
#     packed: bool = False
#     double_side: bool = True
#     pattern: str = "Sine"
#     full_range: bool = False
#     desine: bool = False
#     dcSubtract: bool = True
#     log_scale: bool = False
#     max_projection: bool = False
#     delay: int = 0
#     sine_frame_indices: list[int] = None
#     sine_hires_ratio: int = 0
#     c2: int = 0
#     c3: int = 0
#     octa: str = "none"
#     structure: bool = True

def unp_proc_meta(path) -> unp_meta:

    head, tail = ospath.split(path)
    file_no_ext = tail.split(".")[0]

    # constuct path to metafile assumed to be in same directory
    meta_path_xml = ospath.join(head, file_no_ext + ".xml")

    meta_path_ini = ospath.join(head, file_no_ext + ".ini")

    meta = unp_meta()

    meta.double_side = True

    if Path(meta_path_ini).is_file():
        print(".ini Meta Data exists:")

        config = configparser.ConfigParser()
        config.read(meta_path_ini)

        meta.width = config.getint('General', 'WIDTH')
        meta.height = config.getint('General', 'HEIGHT')
        meta.depth = config.getint('General', 'FRAMES')
        meta.bmscan = config.getint('OCTA', 'BMScan')
        meta.vista = config.getint('Scanning', 'VISTA_Num')
        meta.packed = config.getboolean('Acquisition', 'PACKED12')
        meta.double_side = config.getboolean('Scanning', 'Bidirectional')
        meta.pattern = config['Scanning']['Pattern']
        meta.delay = config.getint('Scanning', 'XDelay')

        if meta.pattern == "Sine_Pause":

            if config.has_option('Scanning', 'Sine_Pause_Frame_Index'):
                meta.sine_frame_indices = list(map(int, config['Scanning']['Sine_Pause_Frame_Index'].split()))
                meta.sine_hires_ratio = config.getint('Scanning', 'Sine_Pause_X_Rate_Reduction')
            else:
                meta.sine_frame_indices = [236, 254, 282, 300, 330, 348, 378, 396, 426, 444]
                meta.sine_hires_ratio = 3

        print("File Info")
        print(f"width: {meta.width}")
        print(f"height: {meta.height}")
        print(f"depth: {meta.depth}")
        print(f"bmscan: {meta.bmscan}")
        print(f"vista: {meta.vista}")
        print(f"packed: {meta.packed}")
        print(f"double_side: {meta.double_side}")
        print(f"pattern: {meta.pattern}")
        print(f"delay: {meta.delay}")
        print(f"full_range: {meta.full_range}")
        print(f"desine: {meta.desine}")
        print(f"dcSubtract: {meta.dcSubtract}")
        print(f"log_scale: {meta.log_scale}")
        print(f"max_projection: {meta.max_projection}")
        print(f"c2: {meta.c2}")
        print(f"c3: {meta.c3}")
        print(f"octa: {meta.octa}")


    if Path(meta_path_xml).is_file():
        print(".xml Meta Data exists:")

        tree = ET.parse(meta_path_xml)
        root = tree.getroot()
        volume_size = root.find(".//Volume_Size")
        volume_size_attrib = volume_size.attrib # type: ignore
        meta.height = int(volume_size_attrib["Height"])
        meta.width = int(volume_size_attrib["Width"])
        meta.depth = int(volume_size_attrib["Number_of_Frames"])

        scanning_params = root.find(".//Scanning_Parameters")
        scanning_params_attrib = scanning_params.attrib # type: ignore
        meta.bmscan = int(scanning_params_attrib["Number_of_BM_scans"])

        print("File Info")
        print(f"width: {meta.width}")
        print(f"height: {meta.height}")
        print(f"depth: {meta.depth}")
        print(f"bmscan: {meta.bmscan}")
        print(f"vista: {meta.vista}")
        print(f"packed: {meta.packed}")
        print(f"double_side: {meta.double_side}")
        print(f"pattern: {meta.pattern}")
        print(f"delay: {meta.delay}")
        print(f"full_range: {meta.full_range}")
        print(f"desine: {meta.desine}")
        print(f"dcSubtract: {meta.dcSubtract}")
        print(f"log_scale: {meta.log_scale}")
        print(f"max_projection: {meta.max_projection}")
        print(f"c2: {meta.c2}")
        print(f"c3: {meta.c3}")
        print(f"octa: {meta.octa}")

    return meta


# @magicgui(
#     input_dir={"label": "Input Directory", "mode": "d"},
#     output_dir={"label": "Output Directory", "mode": "d"},
#     call_button="Generate Enface",
# )
# def generate_enface(
#     input_dir: Path = Path(r"."),
#     output_dir: Path = Path(r"."),
# ):
#     """"""

input_dirs = []
# input_dirs.append(Path(r"Z:\ROP data\2024.08.07"))
# input_dirs.append(Path(r"Z:\ROP data\2023.01.04"))
# input_dirs.append(Path(r"Z:\ROP data\2023.01.11"))
# input_dirs.append(Path(r"Z:\ROP data\2023.01.18"))
# input_dirs.append(Path(r"Z:\ROP data\2023.01.25"))
# input_dirs.append(Path(r"Z:\ROP data\2023.02.08"))
# input_dirs.append(Path(r"Z:\ROP data\2023.02.15"))
# input_dirs.append(Path(r"Z:\ROP data\2023.02.21"))
# input_dirs.append(Path(r"Z:\ROP data\2023.03.01"))
# input_dirs.append(Path(r"Z:\ROP data\2023.03.07"))
# input_dirs.append(Path(r"Z:\ROP data\2023.03.15"))
# input_dirs.append(Path(r"Z:\ROP data\2023.03.21"))
# input_dirs.append(Path(r"Z:\ROP data\2023.03.27"))
# input_dirs.append(Path(r"Z:\ROP data\2023.04.05"))
# input_dirs.append(Path(r"Z:\ROP data\2023.05.24"))
# input_dirs.append(Path(r"Z:\ROP data\2023.05.31"))
# input_dirs.append(Path(r"Z:\ROP data\2023.06.08"))
# input_dirs.append(Path(r"Z:\ROP data\2023.09.06"))
# input_dirs.append(Path(r"Z:\ROP data\2023.09.13"))
# input_dirs.append(Path(r"Z:\ROP data\2023.06.14"))
# input_dirs.append(Path(r"Z:\ROP data\2023.09.20"))
# input_dirs.append(Path(r"Z:\ROP data\2023.06.21"))
# input_dirs.append(Path(r"Z:\ROP data\2023.09.27"))
# input_dirs.append(Path(r"Z:\ROP data\2023.10.04"))
# input_dirs.append(Path(r"Z:\ROP data\2023.11.01"))
# input_dirs.append(Path(r"Z:\ROP data\2023.06.28"))
# input_dirs.append(Path(r"Z:\ROP data\2023.11.08"))
# input_dirs.append(Path(r"Z:\ROP data\2023.07.05"))
# input_dirs.append(Path(r"Z:\ROP data\2023.11.15"))
# input_dirs.append(Path(r"Z:\ROP data\2023.07.12"))
# input_dirs.append(Path(r"Z:\ROP data\2023.11.22"))
# input_dirs.append(Path(r"Z:\ROP data\2023.11.29"))
# input_dirs.append(Path(r"Z:\ROP data\2023.12.06"))
# input_dirs.append(Path(r"Z:\ROP data\2023.12.13"))
# input_dirs.append(Path(r"Z:\ROP data\2023.12.20"))
# input_dirs.append(Path(r"Z:\ROP data\2023.12.27"))
# input_dirs.append(Path(r"Z:\ROP data\2023.07.17"))
# input_dirs.append(Path(r"Z:\ROP data\2023.07.26"))
# input_dirs.append(Path(r"Z:\ROP data\2023.08.02"))
# input_dirs.append(Path(r"Z:\ROP data\2023.08.09"))
# input_dirs.append(Path(r"Z:\ROP data\2023.08.16"))
input_dirs.append(Path(r"Z:\ROP data\2024.11.20"))
input_dirs.append(Path(r"Z:\ROP data\2025-11-25"))

file_paths = []

for input_dir in input_dirs:
    file_paths.extend(list(input_dir.rglob("*.unp")))

output_dir = Path(r"C:\test_batch_output")

for _, file_path in tqdm(enumerate(file_paths), desc="Processing OCT Volumes"): 
    print(f"Processing: {file_path}")

    # meta_file_path  = Path(r"C:\ROP data\2025-08-13\1\26d39229f8ec919a")
    file_meta = unp_proc_meta(file_path)

    # print(file_meta)

    if file_meta.pattern == "Sine_Pause":
        display, display_hires = process_unp_sine_pause(Path(file_path), file_meta, include_hires_in_lowres=True)
    else:
        display = process_unp(Path(file_path), file_meta, auto_dispersion=False)


    enface_mean = np.mean(display[:, 10:-10, :], axis=1)
    enface_mean = desine(torch.tensor(enface_mean), scale_fac=1, transpose=False)
    enface_mean = enface_mean.numpy()

    
    # display = uniform_filter1d(display, size=3, axis=1)
    display = gaussian_filter1d(display, sigma=3, axis=1)

    #generate enface projection

    display = display[:, 10:-10, :].copy()
    
    # indices = np.argmax(display, axis=1)

    # enface_max = np.zeros_like(enface_mean)

    # for i in range(-2,3):
    #     idx = indices + i
    #     idx = np.clip(idx, 0, display.shape[1]-1)
    #     enface_max = enface_max +  np.take_along_axis(display,
    #                                     idx[:, None, :],  # expand axis 1
    #                                     axis=1).squeeze(axis=1)
        
    # enface_max = enface_max / 5.0
    enface_max = np.max(display, axis=1)
    enface_max = desine(torch.tensor(enface_max), scale_fac=1, transpose=False)
    enface_max = enface_max.numpy()
    enface_max = np.log10(enface_max + 1e-6)

    # create output directory structure
    file_parts = file_path.parts
    file_dir = output_dir / file_parts[-3]
    file_dir = file_dir / file_parts[-2]
    file_dir.mkdir(parents=True, exist_ok=True)

    # tifffile.imwrite(
    #     ospath.join(file_dir, f"{file_path.stem}_enface_max_raw.tiff"),
    #     enface_max.astype(np.float32),
    # )

    # tifffile.imwrite(
    #     ospath.join(file_dir, f"{file_path.stem}_enface_mean_raw.tiff"),
    #     enface_mean.astype(np.float32),
    # )

    enface_max_auto_contrast = enface_max.copy()
    vmin, vmax = np.percentile(enface_max_auto_contrast, (5, 99))
    enface_max_auto_contrast = np.clip(enface_max_auto_contrast, vmin, vmax)
    enface_max_auto_contrast = (enface_max_auto_contrast - vmin) / (vmax - vmin)
    enface_max_auto_contrast = enface_max_auto_contrast * 255  # scale to int16
    enface_max_auto_contrast = enface_max_auto_contrast.astype(np.uint8)

    enface_mean_auto_contrast = enface_mean.copy()
    vmin, vmax = np.percentile(enface_mean_auto_contrast, (5, 99))
    enface_mean_auto_contrast = np.clip(enface_mean_auto_contrast, vmin, vmax)
    enface_mean_auto_contrast = (enface_mean_auto_contrast - vmin) / (vmax - vmin)
    enface_mean_auto_contrast = enface_mean_auto_contrast * 255  # scale to int16
    enface_mean_auto_contrast = enface_mean_auto_contrast.astype(np.uint8)

    B = np.zeros_like(enface_max_auto_contrast)
    G = enface_max_auto_contrast
    R = enface_mean_auto_contrast

    rgb = np.stack([R, G, B], axis=-1)   # (H, W, 3)

    tifffile.imwrite(
        ospath.join(file_dir, f"{file_path.stem}_enface_mean_rgb_log.tiff"),
        rgb,
        photometric="rgb",
        compression="none"
    )

print("Finished processing all files.")

# generate_enface.show(run=True)

