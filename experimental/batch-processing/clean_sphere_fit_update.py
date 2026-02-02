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
from scipy.spatial import cKDTree,KDTree
from tqdm import tqdm

from napari_cool_tools_registration._fitting_funcs import sphere_fit_thick_map_corrected_v2
from napari_cool_tools_io._npz_reader import npz_file_reader
# from batch_tools_utils import (
#     get_motor_pos_from_xml,
#     load_bits_labels_v2,
#     sphere_fit_thick_map,
#     sphere_fit_thick_map_corrected,
#     sphere_fit_thick_map_corrected_v2,
#     generate_height_map_from_3D_points,
#     spherical_to_cartesian,
# )

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

def equidistant_loss(center_point_3D:np.ndarray, points_3D:np.ndarray):
    """
    """
    distances = jnp.sqrt(jnp.sum((points_3D.astype("float64") - center_point_3D.astype("float64"))**2,axis=1))
    return jnp.var(distances)

def equidistant_pixel_error_by_degree(error_map:np.ndarray,scan_angle:float=100):
    """
    """
    from scipy import ndimage
    # replace nan values with zero
    data = np.nan_to_num(error_map)
    # get mask of nonzero values
    mask = data.astype(bool)
    # create coordinate grid
    slow_axis,fast_axis = data.shape[:2]
    slow_coords,fast_coords = np.ogrid[0:slow_axis,0:fast_axis]
    # get centroid
    #center_slow,center_fast = ndimage.center_of_mass(mask)
    center_slow,center_fast = slow_axis//2,fast_axis//2
    # shift coordinates for the center
    slow_coords = slow_coords - center_slow
    fast_coords = fast_coords - center_fast
    #TODO add in support for rotations
    # ellipsoid adjustment parameters
    major_axis = slow_axis
    minor_axis = fast_axis
    # calculate distances
    # radii_map = np.sqrt(slow_coords**2 + fast_coords**2)
    elliptical_distance_squared = (slow_coords/major_axis)**2 + (fast_coords/minor_axis)**2
    radii_map = np.sqrt(elliptical_distance_squared)*major_axis
    # define boundaries
    max_distance = np.max(radii_map)
    num_rings = int(scan_angle/2.0)
    zone_boundaries = np.linspace(0,max_distance, num_rings+1)

    #initial_ring_mask = radii_map < zone_boundaries[0]
    #iniitial_pixels_in_ring = data[initial_ring_mask]

    sampled_pixels = []
    #sampled_pixels = {"Ring_0": data[int(center_slow),int(center_fast)]}
    #sampled_pixels = [data[int(center_slow),int(center_fast)]]
    #sampled_pixels = [iniitial_pixels_in_ring.mean()]


    for i in range(num_rings):
        inner_radius = zone_boundaries[i]
        outer_radius = zone_boundaries[i+1]
        # create mask for actively selected pixels
        ring_mask = (radii_map >= inner_radius) & (radii_map < outer_radius)
        # extract pixels
        pixels_in_ring = data[ring_mask]
        if pixels_in_ring.sum() > 0:
            nonzero_mask = pixels_in_ring > 0
            #sampled_pixels[f"Ring_{i+1}"] = pixels_in_ring[nonzero_mask].mean()
            sampled_pixels.append(pixels_in_ring[nonzero_mask].mean())
        else:
            #sampled_pixels.append(0.0)
            pass

    return sampled_pixels

    #  Flatten data and distances
    # flat_data = data.flatten()
    # flat_distances = radii_map.flatten()
    # # sort by distance
    # sorted_indicies = np.argsort(flat_distances)
    # sorted_distances = flat_distances[sorted_indicies]
    # sorted_pixels = flat_data[sorted_indicies]
    # # sample pixels by approx equidistant intervals
    # sampling_indicies = np.linspace(0,len(sorted_indicies)-1,int(scan_angle/2.0),dtype=int)
    # equidistant_samples = sorted_pixels[sampling_indicies]
    # return equidistant_samples, sorted_distances[sampling_indicies]



def main():
    output_dir = Path(r"\\192.168.1.3\coolkid\Beth Roti\Ridge Height Output")
    #output_dir = Path(r"E:\_rebatch_conjugate_test_12_01_2025\Temp_Thickness_Output")

    # set flags
    nn_thick_map_suffix = "nn_topo_map"
    tissue_map:tuple[str,str] = ("retina","choroid")
    tissue_values:tuple[int,int] = (1,2)
    verbose =  False #False #True
    display_in_napari = False #False True
    perform_calculations = True #False True
    save_plots = False #False True
    show_plots = False #False True
    save_nn_topo_map = True # False

    # initialize data paths
    data_path = Path(r"E:\_rebatch_conjugate_test_12_01_2025\output2\08830821-2023_05_31-13_09_12_ret_chor_seg_clean.npz")
    #xml_path = Path(r"E:\_Beth_Thickness_Calculations\All xmls\08977490-2024_05_15-14_53_55.xml")

    #output_dir = Path(r"\\192.168.1.3\coolkid\Beth Roti\Ridge Height Output")
    #output_dir = Path(r"E:\_rebatch_conjugate_test_12_01_2025\Temp_Thickness_Output")
    output_dataframe_filepath = output_dir/"tissue_data.csv"
    #output_dataframe_filepath = output_dir/"retina_radii.csv"

    # data_path = Path(r"E:\_Beth_Thickness_Calculations\processed2\UNPs_08977490-2024_05_15-14_53_55_processed_ret_chor_seg.npz")
    # xml_path = Path(r"E:\_Beth_Thickness_Calculations\All xmls\08977490-2024_05_15-14_53_55.xml")

    # data_path = Path(r"F:\_Complex_Conjugate_Problem\6_mm_test\processed2\08983292-2024_05_01-14_37_16_ret_chor_seg.npz")
    # xml_path = Path(r"F:\_Complex_Conjugate_Problem\_Complex_Conjugate_Problem_08983292-2024_05_01-14_37_16_processed.xml")

    # data_path = Path(r"E:\_Beth_Thickness_Calculations\processed2\UNPs_08983292-2024_05_01-14_37_16_processed_ret_chor_seg.npz")
    # xml_path = Path(r"E:\_Beth_Thickness_Calculations\All xmls\08983292-2024_05_01-14_37_16.xml")

    # data_path = Path(r"E:\_Beth_Thickness_Calculations\processed2\UNPs_08977490-2024_05_09-14_35_14_processed_ret_chor_seg.npz")
    # xml_path = Path(r"E:\_Beth_Thickness_Calculations\All xmls\08977490-2024_05_09-14_35_14.xml")

    # data_path = Path(r"F:\_6-v-12mm-and-scan-v-scan\trial_one\6_mm_segmentations\processed2\08977490-2024_05_09-14_35_14_ret_chor_seg.npy.npz")
    # xml_path = Path(r"F:\_6-v-12mm-and-scan-v-scan\trial_one\08977490-2024_05_09-14_35_14.xml")

    retchor_path_generator = [data_path]

    retchor_path_generator = Path(r"\\192.168.1.3\coolkid\Beth Roti\Ridge Height Output").glob("*_ret_chor_seg_clean.npz")
    #retchor_path_generator = Path(r"E:\_rebatch_conjugate_test_12_01_2025\output2").glob("*_ret_chor_seg_clean.npz")
    
    # TODO figure out efficient method for checking for new files
    #available_file_ids = {path.stem.replace("_ret_chor_seg_clean","") for path in retchor_path_generator}

    # check for existing ids and skip them
    if output_dataframe_filepath.exists():
        existing_dataframe = pl.read_csv(output_dataframe_filepath)
        processed_label_ids = set(existing_dataframe["id_number"].to_list())
    else:
        processed_label_ids = set()

    # check peak retchor 38 values
    #retchor_path_generator = Path(r"E:\_Beth_Thickness_Calculations\Problematic_Retchor").glob("*.npz")
    #retchor_path_generator = Path(r"E:\_Beth_Thickness_Calculations\Peak_38_Retchor").glob("*.npz")
    #retchor_path_generator = Path(r"E:\_Beth_Thickness_Calculations\processed2").glob("*.npz")
    #retchor_path_generator = Path(r"E:\38 peak stage ret_chor crop\test_data").glob("*.npz")


    # initiallize curve correction settings
    cc_settings = CurvCorrectSettings(
        imaging_range=12., #6., #12.,
        imaging_motor_position_delta=6., #0., #6.,
    )

    retchor_progress_bar = tqdm(retchor_path_generator)
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
        # get id
        # prefix = "UNPs_"
        # suffix = "_processed"
        # id_num_start = str(retchor_path).find(prefix) + len(prefix)
        # id_num_end = str(retchor_path).find(suffix)
        # id_number = str(retchor_path)[id_num_start:id_num_end]
        # #print(id_num_start,id_num_end,id_number)
        # xml_path = Path(f"E:\_Beth_Thickness_Calculations\All xmls\{id_number}.xml")
        # #print(f"{xml_path} exists? {xml_path.exists()}\n")
        # #break

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

            ############################################################ Added into loop to test Refine Later

            # load data from files
            #label_data,metadata,layer_type = load_bits_labels_v2(data_path)
            # label_data,metadata,layer_type = load_bits_labels_v2(retchor_path)
            # imaging_motor_position = get_motor_pos_from_xml(xml_path=xml_path)



            # # check imaging motor position
            # if imaging_motor_position:
            #     cc_settings.imaging_motor_position = imaging_motor_position/1000
            # else:
            #     print("skipping...")
            #     continue

            if verbose:
                print(f"Imaging motor position: {cc_settings.imaging_motor_position}\n")

            # # genterate center point guess
            # center_point_guess = np.array([0.0,cc_settings.pivot_point/2.,0.0])
            # if verbose:
            #     print(f"Center point guess: {center_point_guess}\n")

            # isolate retina data
            #tissue_data = label_data == 1

            label_pbar = tqdm(tissue_values)

            for label_val in label_pbar:
                label_pbar.set_description(f"Processing label {label_val} for tissue {tissue_map[label_val-1]}")

                tissue_data = label_data == label_val

                id_number_list.append(id_number)
                tissue_list.append(tissue_map[label_val-1])


                # calculate values for curve correction
                # TODO clairify 
                imaging_range_in_water = cc_settings.imaging_range / cc_settings.refractive_index
                pixel_spacing = imaging_range_in_water / tissue_data.shape[1]
                if verbose:
                    print(f"imaging range in water / A-scan pixels = pixel spacing: {imaging_range_in_water} / {tissue_data.shape[1]} = {pixel_spacing}\n")
                    #print(f"pixel_spacing: {pixel_spacing}\n")

                base_padding = cc_settings.pivot_point-imaging_range_in_water

                reference_arm_shift = ( cc_settings.imaging_motor_position - cc_settings.imaging_motor_position_delta) - cc_settings.reference_motor_position
                reference_arm_shift_in_water = (reference_arm_shift * 0.5) / cc_settings.refractive_index 
                if verbose:
                    print(f"(imaging motor position - imaging motor position delta) - reference motor postition = raw refereence arm shift: ({cc_settings.imaging_motor_position} - {cc_settings.imaging_motor_position_delta}) - {cc_settings.reference_motor_position} = {reference_arm_shift}\n")
                    print(f"(raw reference arm shift / 2) / refractive index = refereence arm shift in water: ({reference_arm_shift} * 0.5) / {cc_settings.refractive_index} = {reference_arm_shift_in_water}\n")

                padding = base_padding + reference_arm_shift_in_water
                if verbose:
                    print(f"base_padding + reference arm shift in air = padding: {base_padding} + {reference_arm_shift_in_water} = {padding}\n")

                padding_pixel = int(padding / pixel_spacing)
                if verbose:
                    print(f"padding / pixel spacing = padding pixels: {padding / pixel_spacing} = {padding_pixel}\n")

                # get surface points and curve correct there positions
                if tissue_data.sum() > 0:
                    (
                        thickness_map, retina_points, rpe_points, curv_ret_points, curv_rpe_points, raw_pixel_thickness_map, pixel_thickness_map
                    ) = sphere_fit_thick_map_corrected_v2(tissue_data,pixel_spacing=pixel_spacing,padding_pixel=padding_pixel,refractive_index=cc_settings.refractive_index)
                    #) = sphere_fit_thick_map_corrected(tissue_data,imaging_motor_position=cc_settings.imaging_motor_position-cc_settings.imaging_motor_position_delta,pivot_point=cc_settings.pivot_point,scan_angle=cc_settings.scan_angle)
                else:
                    return
                
                if save_nn_topo_map:
                    nn_topo_path = output_dir/f"{id_number}_{tissue_map[label_val-1]}_{nn_thick_map_suffix}.npy"
                    if verbose:
                        print(f"Saving {nn_topo_path}\n")
                    save_topology_thread = threading.Thread(target=np.save,kwargs={"file":nn_topo_path,"arr":thickness_map.astype(np.float32)})
                    save_topology_thread.start()
                
                # get pivot points for labeling
                pivot_point_vector = np.array([0.,0.,0.])

                # calculate conversion factor from pixels to microns
                #conv_factor = pixel_spacing * 1000 / cc_settings.refractive_index # mm/pixel * 1000 um/mm / refractive index = um/pixel
                conv_factor = 1.0
                #conv_factor = pixel_spacing * 1000 # mm/pixel * 1000 um/mm = um/pixel

                # maximum distance down axial axis
                max_axial = curv_rpe_points[:,1].max()

                # generate inital guess for center of ellipsoid
                center_guess_slow = curv_rpe_points[:,0].mean()
                center_guess_fast = curv_rpe_points[:,2].mean()
                #center_guess_axial = max_axial - ((16.5/2.0) / pixel_spacing)
                #center_guess_axial = max_axial - ((15.1/2.0) / pixel_spacing)
                center_guess_axial = max_axial - ((cc_settings.pivot_point/2.0) / pixel_spacing)
                center_point_guess = np.array([center_guess_slow,center_guess_axial,center_guess_fast])
                if verbose:
                    print(f"Center point guess: {center_point_guess}\n")

                curv_rpe_points_microns = curv_rpe_points*conv_factor
                curv_ret_points_microns = curv_ret_points*conv_factor

                if verbose:
                    print(f"Axial min/mean/max in microns: {curv_rpe_points_microns[:,1].min()}/{curv_rpe_points_microns[:,1].mean()}/{curv_rpe_points_microns[:,1].max()}")

                # do nearest neighbor calculation
                tree = KDTree(curv_rpe_points_microns)
                curve_correct_height, nearest_indicies = tree.query(curv_ret_points_microns, k=1)
                curve_correct_height2 = np.linalg.norm(curv_rpe_points_microns-curv_ret_points_microns,axis=1)

                if verbose:
                    print(f"curved rpe points shape: {curv_rpe_points.shape}\n") 
                    print(f"curve correct height: {curve_correct_height}\n")
                    print(f"Thickness min/mean/max: {curve_correct_height.min()}/{curve_correct_height.mean()}/{curve_correct_height.max()}\n")
                    print(f"Thickness 2 min/mean/max: {curve_correct_height2.min()}/{curve_correct_height2.mean()}/{curve_correct_height2.max()}\n")
                    print(f"nearest indicies: {nearest_indicies}\n")

                # get min, mean, and max thicknesses
                min_thickness_list.append(curve_correct_height.min())
                mean_thickness_list.append(curve_correct_height.mean())
                max_thickness_list.append(curve_correct_height.max())

                # find heights above theshold
                thickness_threshold = 200 #300 #250 #350 400
                threshold_mask = curve_correct_height > thickness_threshold
                threshold_indicies = threshold_mask.nonzero()

                # get start and end points for thickness measurement
                # start_points = curv_ret_points_microns[nearest_indicies]
                # end_points = curv_rpe_points_microns[nearest_indicies]

                #start_points = curv_ret_points_microns[nearest_indicies[threshold_indicies]]
                start_points = curv_ret_points_microns[threshold_indicies]
                end_points = curv_rpe_points_microns[nearest_indicies[threshold_indicies]]
                #end_points = curv_rpe_points_microns[threshold_indicies]

                if verbose:
                    print(f"number of start points: {len(start_points)}\n")
                    print(f"number of end points: {len(end_points)}\n")
                    print(f"number of unique endpoints: {len(np.unique(end_points,axis=0))}\n")

                # generarte linesegments to display
                if start_points.shape == end_points.shape:
                    #line_segments = []
                    min_seg = 1000.0
                    max_seg = 0.0
                    #for p1,p2 in tqdm(zip(start_points,end_points),desc="Generating Line Segments"):
                    for p1,p2 in zip(start_points,end_points):
                        distance = np.linalg.norm(p2-p1) #np.sqrt(np.sum((p1-p2)**2))
                        if distance < min_seg:
                            min_seg = distance
                        if distance > max_seg:
                            max_seg = distance
                        line_segments.append(np.array([p1,p2]))

                    if verbose:
                        print(f"min seg/max seg: {min_seg}/{max_seg}\n")
                        print(f"nearest min/mean/max: {curve_correct_height[threshold_indicies].min()} / {curve_correct_height[threshold_indicies].mean()} / {curve_correct_height[threshold_indicies].max()}\n")
                        #print(f"Nearest min/mean/max: {curve_correct_height.min()} / {curve_correct_height.mean()} / {curve_correct_height.max()}\n")

                    #scaled_heights = (curve_correct_height-min_seg) / (max_seg-min_seg)
                    #colormap = 

                slow_shape,axial_shape,fast_shape = tissue_data.shape
                rpe_slow, rpe_axial, rpe_fast = rpe_points.T
                pixel_thickness_map2 = np.full((slow_shape, fast_shape), 0.0)
                pixel_thickness_map3 = np.full((slow_shape, fast_shape), 0.0)

                # Fill thickness map at (y, x) positions
                pixel_thickness_map2[rpe_slow, rpe_fast] = curve_correct_height
                pixel_thickness_map3[rpe_slow, rpe_fast] = curve_correct_height2

                # # calculate mimimal distortion samples
                # minimal_distortion_curv_rpe = get_nearest_neigbors_to_target(target_point=curv_fovea_point[None,:],point_distribution=curv_rpe_points,ratio=ratio_near_target)
                # print(f"minimal_distortion_curv_rpe shape: {minimal_distortion_curv_rpe.shape}\n")

                # # choose to use full point distribution or minimal distortion sample
                # if not use_min_sample:
                #     rpe_point_input = curv_rpe_points
                # else:
                #     rpe_point_input = minimal_distortion_curv_rpe

                rpe_point_input = curv_rpe_points

                # perform sphere fitting on data
                output = fmin_bfgs(
                    f=equidistant_loss,
                    x0=center_point_guess.astype("float64"),
                    fprime=grad(equidistant_loss),
                    norm=2.0,
                    args=(rpe_point_input.astype("float64"),),
                    #gtol=1e-17,
                    maxiter=None,
                    full_output=True,
                    disp=False, #True False
                    retall=False,
                    callback=None,
                )

                if verbose:
                    print(f"output: {output}\n")
                std_from_center = jnp.sqrt(output[1])

                min_std = np.sqrt(output[1])
                if verbose:
                    print(f"std in pixels: {std_from_center}\n")
                    print(f"std in mm: {std_from_center*pixel_spacing}\n")
                min_center_pt = output[0]
                if verbose:
                    print(f"predicted center point in pixels: {min_center_pt}\n")
                    print(f"predicted center point in mm: {min_center_pt*pixel_spacing}\n")

                radius_calc = np.sqrt(np.sum((rpe_point_input - min_center_pt)**2,axis=1)).mean()
                if verbose:
                    print(f"radius calculation from fitting: {radius_calc} pixels, {radius_calc*pixel_spacing} mm\n")
                radius_list.append(radius_calc*pixel_spacing)

                if perform_calculations:

                    raw_micron_thickness_map = raw_pixel_thickness_map*conv_factor
                    pixel_difference_map = np.round(raw_pixel_thickness_map).astype("uint8") - np.round(pixel_thickness_map).astype("uint8")
                    micron_difference_map = ((raw_pixel_thickness_map-pixel_thickness_map)*conv_factor).astype("float32")
                    #micron_difference_map2 = (raw_micron_thickness_map - thickness_map).astype("float32")
                    epsilon = np.finfo(float).eps
                    micron_error_map = (np.abs(raw_micron_thickness_map-pixel_thickness_map)/(pixel_thickness_map + epsilon)).astype("float32")

                    #print(f"micron diff argmin: {np.unravel_index(micron_difference_map.argmin(),pixel_thickness_map.shape)}\n")

                    #sampled_pixels,sampled_distances = equidistant_pixel_error_by_degree(micron_error_map,scan_angle=100.0)
                    sampled_pixels = equidistant_pixel_error_by_degree(micron_error_map,scan_angle=100.0)
                    eccentricity_error_values.append(np.array(sampled_pixels))

                    if save_plots or show_plots:
                        fig,ax = plt.subplots(figsize=(8,5))
                        ax.plot(np.arange(len(sampled_pixels)),sampled_pixels,marker='o',linestyle='-',color='b')

                        ax.set_xlabel("Eccentricity (degrees)")
                        ax.set_ylabel("Precent Error")
                        ax.set_title(f"{id_number} Percent Error vs. Ececentricity(Degrees)")

                        formatter = mtick.PercentFormatter(xmax=100.0,decimals=1)
                        ax.yaxis.set_major_formatter(formatter)

                        ax.grid(True,linestyle='--',alpha=0.6)

                        output_plot_dir = Path(r"E:\_rebatch_conjugate_test_12_01_2025\Temp_Thickness_Output")
                    if save_plots:
                        plt.savefig(output_plot_dir/f"{id_number}_error_plot.png")

                    if show_plots:
                        plt.show()

                    #print(sampled_pixels,sampled_distances)
                    #print(sampled_pixels)

                    # update processed id list
                    processed_label_ids.add(id_number)

                    if display_in_napari:

                        viewer.add_image(raw_pixel_thickness_map,name=f"{id_number}_raw_thick_map")
                        viewer.add_image(raw_micron_thickness_map,name=f"{id_number}_raw_micron_thick_map")
                        viewer.add_image(pixel_thickness_map,name=f"{id_number}_pixel_thick_map")
                        viewer.add_image(pixel_thickness_map2,name=f"{id_number}_pixel_thick_map2")
                        viewer.add_image(pixel_thickness_map3,name=f"{id_number}_pixel_thick_map3")
                        viewer.add_image(thickness_map,name=f"{id_number}_thick_map")
                        viewer.add_image(pixel_difference_map,name=f"{id_number}_pixel_diff_map")
                        viewer.add_image(micron_difference_map,name=f"{id_number}_micron_diff_map")
                        viewer.add_image(micron_error_map,name=f"{id_number}_micron_error_map")
                        viewer.add_points(curv_ret_points_microns,size=0.5,face_color="magenta",border_color="magenta",name="retina",blending="translucent_no_depth")
                        viewer.add_points(curv_rpe_points_microns,size=0.5,face_color="yellow",border_color="yellow",name="rpe",blending="translucent_no_depth")
                        viewer.add_points(start_points,size=0.5,face_color="green",border_color="green",name="retina start",blending="translucent_no_depth")
                        viewer.add_points(end_points,size=0.5,face_color="red",border_color="red",name="rpe stop",blending="additive")
                        #viewer.add_image(micron_difference_map2,name=f"{id_number}_micron_diff_map2")

                        if len(line_segments) > 0:
                            viewer.add_shapes(line_segments[::4],shape_type="line",edge_width=1,name="thickness_segments",blending="translucent_no_depth",edge_color="blue") #edge_colormap="viridis")

    if perform_calculations and (save_plots or show_plots):

        shapes_arr = np.array([arr.shape[0] for arr in eccentricity_error_values])
        end_idx = shapes_arr.min()
        print(f"end idx: {end_idx}\n")
        aligned_eccentricity_values = [arr[:end_idx] for arr in eccentricity_error_values]
        print([arr.shape[0] for arr in eccentricity_error_values])
        print([arr.shape[0] for arr in aligned_eccentricity_values])
        #print(eccentricity_error_values)

        #all_sampled_pixels = np.stack(eccentricity_error_values,axis=0)
        all_sampled_pixels = np.stack(aligned_eccentricity_values,axis=0)
        all_sampled_pixels = all_sampled_pixels.mean(axis=0)

        #if save_plots or show_plots:
        fig,ax = plt.subplots(figsize=(8,5))
        ax.plot(np.arange(len(all_sampled_pixels)),all_sampled_pixels,marker='o',linestyle='-',color='b')

        ax.set_xlabel("Eccentricity (degrees)")
        ax.set_ylabel("Precent Error")
        ax.set_title("Aggregate Percent Error vs. Ececentricity(Degrees)")

        formatter = mtick.PercentFormatter(xmax=100.0,decimals=1)
        ax.yaxis.set_major_formatter(formatter)

        ax.grid(True,linestyle='--',alpha=0.6)

        output_plot_dir = Path(r"E:\_rebatch_conjugate_test_12_01_2025\Temp_Thickness_Output")

        if save_plots:
            plt.savefig(output_plot_dir/"aggregate_error_plot.png")

        if show_plots:
            plt.show()

    if len(radius_list) > 0:
        output_dataframe = pl.DataFrame({
            "id_number": id_number_list,
            "tissue": tissue_list,
            "radii": radius_list,
            "min_thickness": min_thickness_list,
            "mean_thickness": mean_thickness_list,
            "max_thickness": max_thickness_list,
        })

        #output_dir = Path(r"\\192.168.1.3\coolkid\Beth Roti\Ridge Height Output")
        #output_dir = Path(r"E:\_rebatch_conjugate_test_12_01_2025\Temp_Thickness_Output")
        #output_dataframe_filepath = output_dir/"tissue_data.csv"

        if output_dataframe_filepath.exists():
            existing_dataframe = pl.read_csv(output_dataframe_filepath)
            #output_dataframe = pl.concat([existing_dataframe,output_dataframe])
            output_dataframe = existing_dataframe.vstack(output_dataframe).unique(maintain_order=True)
        print(output_dataframe)
        output_dataframe.write_csv(output_dataframe_filepath)

        retina_df = output_dataframe.filter(
            pl.col("tissue").str.contains("retina", literal=True)
        )

        choroid_df = output_dataframe.filter(
            pl.col("tissue").str.contains("choroid", literal=True)
        ) 

        radii_ndarray = np.array(output_dataframe["radii"].to_list())
        diameter_ndarray = radii_ndarray*2.
        retina_min_thick_array = np.array(retina_df["min_thickness"].to_list())
        retina_mean_thick_array = np.array(retina_df["mean_thickness"].to_list())
        retina_max_thick_array = np.array(retina_df["max_thickness"].to_list())
        choroid_min_thick_array = np.array(choroid_df["min_thickness"].to_list())
        choroid_mean_thick_array = np.array(choroid_df["mean_thickness"].to_list())
        choroid_max_thick_array = np.array(choroid_df["max_thickness"].to_list())
        print(f"min radius: {radii_ndarray.min()}, mean radius: {radii_ndarray.mean()}, max radius: {radii_ndarray.max()}\n")
        print(f"min diameter: {diameter_ndarray.min()}, mean diameter: {diameter_ndarray.mean()}, max diameter: {diameter_ndarray.max()}\n")
        print(f"retina min thickness: {retina_min_thick_array.min()}, retina mean thickness: {retina_mean_thick_array.mean()}, retina max thickness: {retina_max_thick_array.max()}\n")
        print(f"choroid min thickness: {choroid_min_thick_array.min()}, choroid mean thickness: {choroid_mean_thick_array.mean()}, choroid max thickness: {choroid_max_thick_array.max()}\n")

        if display_in_napari:
            # if len(line_segments) > 0:
            #     viewer.add_shapes(line_segments[::1],shape_type="line",edge_color="blue",edge_width=1,name="thickness_segments",blending="translucent_no_depth")
                #viewer.add_shapes(line_segments,shape_type="line",edge_color="blue",edge_width=1,name="thickness_segments",blending="translucent_no_depth")
            viewer.show()
            napari.run()

if __name__ == "__main__":
    main()

