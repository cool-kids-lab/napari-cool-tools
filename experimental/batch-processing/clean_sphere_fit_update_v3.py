"""
Sphere Fit Code has become to messy cleanup code and document to assure accuracy
"""
from dataclasses import dataclass
from pathlib import Path
import threading

import cv2
import jax
import jax.numpy as jnp
from jax import grad
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import napari
import numpy as np
import polars as pl
from scipy.optimize import fmin_bfgs
from scipy.spatial import KDTree
import torch
from tqdm import tqdm

from napari_cool_tools_registration._fitting_funcs import sphere_fit_thick_map_corrected_v2
from napari_cool_tools_io._npz_reader import npz_file_reader

from napari_cool_tools_registration._fitting_funcs import (
    extract_surfaces_and_curve_correct_coordinates,
    generate_angle_of_incidence_map,
)
from curve_correct_utils import (
    # CurvCorrectSettings,
    equidistant_loss,
    generate_noisy_ellipsoid_sample_data,
    get_incidence_angle_torch,
    get_pixel_spacing_and_padding,
)

jax.config.update("jax_enable_x64", True)

@dataclass
class CurvCorrectSettings():
    pivot_point:float = 19.2
    imaging_range:float = 12.0
    reference_motor_position:float = 85.0
    imaging_motor_position: float = 85.0
    imaging_motor_position_delta: float = 0.0
    refractive_index: float = 1.33
    scan_angle: float = 100

def main():
    #output_dir = Path(r"\\192.168.1.3\coolkid\Beth Roti\Stage 0 Output")
    # output_dir = Path(r"\\192.168.1.3\coolkid\Beth Roti\Ridge_Height_Troubelshoot")

    # input_dir = Path(r"\\192.168.1.3\coolkid\Beth Roti\Ridge Height Output\Clean_Labels_Topo_Maps")
    # output_dir = Path(r"\\192.168.1.3\coolkid\Beth Roti\Ridge Height Output\NN_Topo_Maps")

    # input_dir = Path(r"\\192.168.1.3\coolkid\Beth Roti\TestforJohn\OutputForBeth")
    # output_dir = Path(r"\\192.168.1.3\coolkid\Beth Roti\TestforJohn\OutputForBeth")
    
    # input_dir = Path(r"\\192.168.1.3\coolkid\Beth Roti\Stage 0 Output v2")
    # output_dir = Path(r"\\192.168.1.3\coolkid\Beth Roti\Stage 0 Output v2\nn_topo_maps")

    # input_dir = Path(r"\\192.168.1.3\coolkid\Beth Roti\Ridge Project\Ridge Height Output\Clean_RetChor_Topo_Preview")
    # output_dir = Path(r"\\192.168.1.3\coolkid\Beth Roti\Ridge Project\Ridge Height Output\Clean_Labels_Topo_Maps\Measurement_Maps")

    input_dir = Path(r"\\192.168.1.3\coolkid\Beth Roti\Ridge Project\All ridge project Outputs\Clean_RetChor_Topo_Preview")
    output_dir = Path(r"\\192.168.1.3\coolkid\Beth Roti\Ridge Project\All ridge project Outputs\Measurement_Topo_Maps")

    #output_dir = Path(r"\\192.168.1.3\coolkid\Beth Roti\Ridge Height Output")
    #output_dir = Path(r"E:\_rebatch_conjugate_test_12_01_2025\Temp_Thickness_Output")

    # set flags
    cc_thick_map_suffix = "cc_nn_micro_topo_map"
    raw_thick_map_suffix = "raw_micro_topo_map"
    nn_incidence_map_suffix = "nn_incidence_map"
    tissue_map:tuple[str,str] = ("retina","choroid")
    tissue_values:tuple[int,int] = (1,2)
    verbose =  False #False #True
    display_in_napari = False #False True
    perform_calculations = False #False True
    save_plots = False #False True
    show_plots = False #False True
    save_nn_topo_map = True # False

    # initialize data paths

    output_dataframe_filepath = output_dir/"tissue_data.csv"

    retchor_path_generator = input_dir.glob("*_ret_chor_seg_clean.npz")
    #retchor_path_generator = Path(r"E:\_rebatch_conjugate_test_12_01_2025\output2").glob("*_ret_chor_seg_clean.npz")
    
    # TODO figure out efficient method for checking for new files
    #available_file_ids = {path.stem.replace("_ret_chor_seg_clean","") for path in retchor_path_generator}

    # check for existing ids and skip them
    if output_dataframe_filepath.exists():
        existing_dataframe = pl.read_csv(output_dataframe_filepath)
        processed_label_ids = set(existing_dataframe["id_number"].to_list())
    else:
        processed_label_ids = set()

    # initiallize curve correction settings
    cc_settings = CurvCorrectSettings(
        imaging_range=12., #6., #12.,
        imaging_motor_position_delta=6., #0., #6.,
    )

    retchor_progress_bar = tqdm(retchor_path_generator)
    output_dataframe = None
    id_number_list = []
    tissue_list = []
    radius_list = []
    min_thickness_list = []
    mean_thickness_list = []
    max_thickness_list = []
    eccentricity_error_values = []

    if display_in_napari:
        viewer = napari.Viewer(show=False)

    for retchor_path in retchor_progress_bar:
        retchor_progress_bar.set_description(f"Processing: {retchor_path.stem}")

        name = retchor_path.stem
        id_number = name.replace("_ret_chor_seg_clean","")

        if id_number in processed_label_ids:
            print(f"Skipping {name} thickness has already been processed.")
            continue

        label_data,attributes,layer_type = npz_file_reader(retchor_path,return_layer=True,verbose=False)[0]
        #name = attributes["name"]
        #id_number = name.replace("_ret_chor_seg_clean","")
        if "metadata" in attributes:
            if "motor_position" in attributes["metadata"]:
                imaging_motor_position = attributes["metadata"]["motor_position"]
                cc_settings.imaging_motor_position = imaging_motor_position/1000

        # if xml_path.exists():

            line_segments = []

            if verbose:
                print(f"Imaging motor position: {cc_settings.imaging_motor_position}\n")

            label_pbar = tqdm(tissue_values)

            for label_val in label_pbar:
                label_pbar.set_description(f"Processing label {label_val} for tissue {tissue_map[label_val-1]}")

                tissue_data = label_data == label_val

                id_number_list.append(id_number)
                tissue_list.append(tissue_map[label_val-1])


                # get pixel spacing and padding for the volume
                pixel_spacing, padding_pixel = get_pixel_spacing_and_padding(
                    cc_settings=cc_settings, axial_data_shape=tissue_data.shape[1], verbose=False
                )

                # get surface points and curve correct there positions
                if tissue_data.sum() > 0:
                    (
                        cc_micro_thickness_map,
                        cc_pixel_thickness_map,
                        curv_ret_points,
                        curv_rpe_points,
                        retina_points,
                        rpe_points,
                        cc_retina_nn_points,
                        raw_micro_thickness_map,
                        raw_pixel_thickness_map,
                    ) = extract_surfaces_and_curve_correct_coordinates(
                        tissue_data,
                        pixel_spacing=pixel_spacing,
                        padding_pixel=padding_pixel,
                        refractive_index=cc_settings.refractive_index,
                    )
                else:
                    return

                curv_points_of_interest = curv_rpe_points
                points_of_interest = rpe_points

                nearest_neighbor_dir = curv_ret_points - cc_retina_nn_points

                nn_angles_of_incidence = get_incidence_angle_torch(
                    torch.as_tensor(curv_points_of_interest, dtype=torch.bfloat16),
                    torch.as_tensor(nearest_neighbor_dir, dtype=torch.bfloat16),
                    use_degrees=True,
                )

                nn_angle_of_incidence_map = generate_angle_of_incidence_map(
                    points_of_interest, nn_angles_of_incidence, (840, 800)
                )
                
                if save_nn_topo_map:
                    nn_topo_path = output_dir/f"{id_number}_{tissue_map[label_val-1]}_{cc_thick_map_suffix}.npy"
                    if verbose:
                        print(f"Saving {nn_topo_path}\n")
                    save_topology_thread = threading.Thread(target=np.save,kwargs={"file":nn_topo_path,"arr":cc_micro_thickness_map.astype(np.float32)})
                    # save_topology_thread = threading.Thread(target=np.save,kwargs={"file":nn_topo_path,"arr":thickness_map.astype(np.float64)})
                    save_topology_thread.start()

                if save_nn_topo_map:
                    nn_topo_path = output_dir/f"{id_number}_{tissue_map[label_val-1]}_{raw_thick_map_suffix}.npy"
                    if verbose:
                        print(f"Saving {nn_topo_path}\n")
                    save_topology_thread = threading.Thread(target=np.save,kwargs={"file":nn_topo_path,"arr":raw_micro_thickness_map.astype(np.float32)})
                    # save_topology_thread = threading.Thread(target=np.save,kwargs={"file":nn_topo_path,"arr":thickness_map.astype(np.float64)})
                    save_topology_thread.start()
                if save_nn_topo_map:
                    nn_topo_path = output_dir/f"{id_number}_{tissue_map[label_val-1]}_{nn_incidence_map_suffix}.npy"
                    if verbose:
                        print(f"Saving {nn_topo_path}\n")
                    save_topology_thread = threading.Thread(target=np.save,kwargs={"file":nn_topo_path,"arr":nn_angle_of_incidence_map.astype(np.float32)})
                    # save_topology_thread = threading.Thread(target=np.save,kwargs={"file":nn_topo_path,"arr":thickness_map.astype(np.float64)})
                    save_topology_thread.start()
                
main()

if __name__ == "__main__":
    main()
