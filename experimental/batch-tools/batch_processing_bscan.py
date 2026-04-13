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
from napari_cool_tools_io import unp_meta
from napari_cool_tools_oct_preproc._oct_preproc_func import desine, auto_contrast
import torch
import tifffile

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


def post_process(data, desine_fact=1, auto_contrast_percentile=[1,99], log_scale=False):
    
    #do desine
    input_data = torch.Tensor(data).cuda()
    processed_torch = desine(input_data, scale_fac=desine_fact,transpose=False)

    #do log scale
    if log_scale:
        processed_torch = torch.log10(processed_torch + 1e-6)
    
    #do auto contrast
    center = data.shape[0] // 2
    temp_frame_torch = processed_torch[center-1 : center+1]#average of 3
    temp_frame_torch = torch.mean(temp_frame_torch, axis=0)
    temp_frame = temp_frame_torch.cpu().numpy()

    processed = processed_torch.cpu().numpy()
    del input_data, processed_torch, temp_frame_torch
    torch.cuda.empty_cache()

    vmin, vmax = np.percentile(temp_frame, (auto_contrast_percentile[0], auto_contrast_percentile[1]))
    processed = np.clip(processed, vmin, vmax)
    processed = (processed - vmin) / (vmax - vmin)

    return processed#output is normalized to 0-1




#main batch processing loop
input_dirs = []
# input_dirs.append(Path(r"Z:\ROP data\2025-11-25"))
input_dirs.append(Path(r"Z:\ROP data\2024.11.20"))

file_paths = []

for input_dir in input_dirs:
    file_paths.extend(list(input_dir.rglob("*.unp")))

output_dir = Path(r"C:\test_batch_output_bscan")

for _, file_path in tqdm(enumerate(file_paths), desc="Processing OCT Volumes"): 
    print(f"Processing: {file_path}")

    # meta_file_path  = Path(r"C:\ROP data\2025-08-13\1\26d39229f8ec919a")
    file_meta = unp_proc_meta(file_path)

    if file_meta.pattern == "Sine_Pause":
        display, display_hires = process_unp_sine_pause(Path(file_path), file_meta, include_hires_in_lowres=True)
    else:
        display = process_unp(Path(file_path), file_meta, auto_dispersion=False)
    
    
    # create output directory structure
    file_parts = file_path.parts
    file_dir = output_dir / file_parts[-3]
    file_dir = file_dir / file_parts[-2]
    file_dir.mkdir(parents=True, exist_ok=True)

    log_scales = [True, False]
    desine_facts = [1,2]
    auto_contrast_percentile_mins = np.arange(1,6)
    auto_contrast_percentile_maxs = np.arange(95,100)

    for log_scale in log_scales:
        for desine_fact in desine_facts:
            for ac_min in auto_contrast_percentile_mins:
                for ac_max in auto_contrast_percentile_maxs:
                    print(f"Processing with log_scale={log_scale}, desine_fact={desine_fact}, auto_contrast_percentile=[{ac_min},{ac_max}]")
                    
                    post_processed = post_process(display, desine_fact=desine_fact, auto_contrast_percentile=[ac_min, ac_max], log_scale=log_scale)

                    center = post_processed.shape[0] // 2
                    post_process_ave = post_processed[center-1 : center+1]#average of 3
                    post_process_ave = np.mean(post_process_ave, axis=0)
                    post_processed_ave_uint16 = (post_process_ave * 65535.0).astype(np.uint16)
                    tifffile.imwrite(
                        ospath.join(file_dir, f"{file_path.stem}_ave_log{int(log_scale)}_desine{desine_fact}_ac{int(10*ac_min)}-{int(10*ac_max)}.tiff"),
                        post_processed_ave_uint16,
                    )

                    post_processed_uint16 = (post_processed[center] * 65535.0).astype(np.uint16)
                    tifffile.imwrite(
                        ospath.join(file_dir, f"{file_path.stem}_log{int(log_scale)}_desine{desine_fact}_ac{int(10*ac_min)}-{int(10*ac_max)}.tiff"),
                        post_processed_uint16,
                    )

                    enface_max_auto_contrast = np.mean(post_processed[:, 10:-10, :], axis=1)
                    vmin, vmax = np.percentile(enface_max_auto_contrast, (1, 99))
                    enface_max_auto_contrast = np.clip(enface_max_auto_contrast, vmin, vmax)
                    enface_max_auto_contrast = (enface_max_auto_contrast - vmin) / (vmax - vmin)
                    enface_max_auto_contrast = enface_max_auto_contrast * 65535.0  # scale to int16
                    enface_max_auto_contrast = enface_max_auto_contrast.astype(np.uint16)

                    tifffile.imwrite(
                        ospath.join(file_dir, f"{file_path.stem}_enface_log{int(log_scale)}_desine{desine_fact}_ac{int(10*ac_min)}-{int(10*ac_max)}.tiff"),
                        enface_max_auto_contrast,
                    )

print("Finished processing all files.")




    