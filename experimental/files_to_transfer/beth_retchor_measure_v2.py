import sys
import argparse
from typing import Literal
from pathlib import Path

import numpy as np
import polars as pl
import pandas as pd
from openpyxl import load_workbook
import napari
from numpy.typing import ArrayLike
from magicgui import magicgui
from qtpy.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QPushButton,
    QFileDialog,
    QLabel,
    QMessageBox,
)
from tqdm import tqdm
from scipy.io import loadmat
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET

from batch_tools_utils import sphere_fit_thick_map, generate_circular_mask


def ridge_analysis(
    ridge: ArrayLike,
    ret_chor: ArrayLike,
    scan_angle: float = 106.0,
    imaging_range: float = 6.0,
    reference_motor_position: float = 85.0,
    imaging_motor_position: float = 0.0,
    refractive_index: float = 1.33,
    circular_mask_scale: float = 1.0,
    mode: Literal["center", "optic disk", "fovea"] = "center",
    nearest_neighbor_calc: bool = True,
    incedence_correction: bool = True,
    micrometer_output: bool = True,
    #display_in_napari: bool = False,
    viewer = None,
    key = None,
    verbose: bool = True,
    debug: bool = False,
) -> tuple[float, float]:
    """Calculate mean and peak thickness from ridge and retchor segmentation labels.

    Calculates retinal thickness in pixels (optionally microns) using Bscan retinal slab segmentations and en face ridge segmentations.

    Args:
        ridge: Label array generated from en face UWF-OCT scan that highlighs the location of the fibrovascular ridge with  shape(Fast Axis, Slow Axis)
        retchor: Label array containing retinal and choroidal slabs generated from Bscans of UWF-OCT scan with shape(Fast Axis,Axial,Slow Axis)
        scan_angle: UWF-OCT field of view in degrees SCAN_ANGLE ASSUMES USING OLD CAMERA (circa 2023-2024); ADJUST WITH DIFFERENT DEVICE
        imaging_range: UWF-OCT axial depth in mm
        refractive_index: Refreactive index for calculating refraction within the eye (valid for contact handheld device)
        mode: Option to determine the center for correcting thickness measurements due to the non-orthogonal inciden of Ascans in the periphery assuming a spherical model of the eye
        incedence_correction: Flag for activating Ascan incedence angle correction
        micrometer_output: Falg for converting output to micrometers
        dispaly_in_napari: Dispaly selected retinal depth maps and other visual debugging informatino in napari viewer
        verbose: Print thickness information
        debug: Print statements helpful for debuging
    Returns:
        Tuple containing the arithmetic mean of the ridge thickness as well as the maximum measured thickness of the fibrovascular ridge
    Raises:
        ValueError if retchor is not 3-Dimensional
        ValueError if ridge shape does not match rdm shape

    Assumes retchor shape: (Z, Y, X) and ridge shape matches (Z, X).
    SCANWIDTH ASSUMES USING OLD CAMERA (circa 2023-2024); ADJUST WITH DIFFERENT DEVICE
    ALSO ASSUMES CENTER OF IMAGE IS CLOSE TO OPTICAL AXIS; NOT VAILD FOR NONTEMPORAL OR CENTRAL IMAGES
    """
    if ret_chor.ndim != 3:
        raise ValueError(f"Expected 3D retchor array, got shape {ret_chor.shape}")

    # half_end = int(retchor.shape[1]/2)
    # retchor = retchor[:,half_end:,:]

    def clean_mask(mask):
        slow_axis,axial_axis,fast_axis = mask.shape
        #mask[:,:int(axial_axis*1/4),:] = 0
        #mask[:,:int(940),:] = 0
        mask[:,:int(780),:] = 0

    clean_mask(ret_chor)

    retina_mask = (ret_chor == 1).astype(np.float64)
    rdm = retina_mask.sum(axis=1)  # shape: (Z, X)

    if ridge.shape != rdm.shape:
        raise ValueError(
            f"Shape mismatch: ridge {ridge.shape} vs thickness map {rdm.shape}"
        )

    try: 
        assert ridge.sum() > 0
    except:
        print(f"This volume contains no ridge labels")
        return -1.0,-1.0
    
    reference_arm_shift = reference_motor_position - imaging_motor_position
    reference_arm_shift = (reference_arm_shift * 0.5) / refractive_index

    # generate enface circular mask
    en_face_circular_mask = generate_circular_mask(retina_mask.shape,circular_mask_scale)

    rdm[~en_face_circular_mask] = 0

    # apply mask along depth axis
    en_face_circular_mask = np.broadcast_to(en_face_circular_mask[:,None,:],retina_mask.shape)

    print(f"retina mask shape: {retina_mask.shape}, en face circular mask shape: {en_face_circular_mask.shape}\n")
    retina_mask[~en_face_circular_mask] = 0

    if nearest_neighbor_calc:
        #thickness_map, retina_coords, rpe_coords, curv_ret_coords, curv_rpe_coords = sphere_fit_thick_map(
        thickness_map, *_ = sphere_fit_thick_map(
            mask=retina_mask,
            refractive_index=refractive_index,
            imaging_motor_position=imaging_motor_position,
            reference_motor_position=reference_motor_position,
            imaging_range=imaging_range,
            scan_angle=scan_angle,
        )

    if verbose:
        print(f"reference_arm_shift: {reference_arm_shift}\n")
        

    thickness_vals = rdm[(ridge == 4) & (rdm != 0)]

    if debug:
        print(f"\n\nrdm type: {rdm.dtype}\nthickness type: {thickness_vals.dtype}\n\n")

    if verbose or debug:
        print(
            f"Raw thickness mean:\n{thickness_vals.mean()}\nRaw thickness max: {thickness_vals.max()}\n"
        )

    if incedence_correction:
        x, y = tuple([*rdm.shape[-2:]])

        match mode:
            case "fovea":
                print(
                    "Foveal center is not implemented yet performing calculation using scan center."
                )
                center_x = x // 2
                center_y = y // 2
            case "center":
                center_x = x // 2
                center_y = y // 2
            case "optic disk":
                print(
                    "Optic disk center is not implemented yet performing calculation using scan center."
                )
                center_x = x // 2
                center_y = y // 2

        scan_angle_from_center = (
            scan_angle // 2
        )  # TODO update this in future to account for differences
        min_scan_angle = 90 - scan_angle_from_center
        max_scan_angle = 180 - scan_angle_from_center

        if debug:
            print(f"center of image: {center_x, center_y}")
            print(f"\n\nx,y shapes: {x, y}\n\n")

        thetax, thetay = np.mgrid[
            0 - center_x : x - center_x, 0 - center_y : y - center_y
        ]
        # print(f"\n\ncenter_x,center_y: {center_x,center_y}\n\n")
        if debug:
            print(f"\n\ntheta_x,theta_y: {thetax, thetay},{thetax.shape, thetay.shape}\n\n")

        # x_conv = np.linspace(0-scan_angle_from_center,scan_angle-scan_angle_from_center,num=x)
        # y_conv = np.linspace(0-scan_angle_from_center,scan_angle-scan_angle_from_center,num=y)
        x_conv = np.linspace(-min_scan_angle, min_scan_angle, num=x)
        y_conv = np.linspace(-min_scan_angle, min_scan_angle, num=y)

        if debug:
            print(f"conversion shapes: {x_conv.shape},{y_conv.shape}\n\n")

        x_degree = np.repeat(x_conv[:, None], y, axis=1)
        y_degree = np.repeat(y_conv[None, :], x, axis=0)

        x_rad = x_degree / (2 * np.pi) / 4
        y_rad = y_degree / (2 * np.pi) / 4
        # x_degree = np.tile(x_conv[:,None],(1,y))
        # y_degree = np.tile(y_conv,(x,1))

        # y_degree = np.repeat(y_conv[:,None],y,axis=1)
        # print(f"\n\n\nthetax:{thetax[0],thetax[0].shape}\n\n")
        # print(f"x_conv: {x_conv},{x_conv.shape}\n\n")
        # print(f"y_conv: {y_conv},{y_conv.shape}\n\n")

        if debug:
            print(f"x_degree: {x_degree}\n{x_degree.shape}\n")
            print(f"y_degree: {y_degree}\n{y_degree.shape}\n")

        if debug:
            print(f"x_rad: {x_rad}\n{x_rad.shape}\n")
            print(f"y_rad: {y_rad}\n{y_rad.shape}\n")

        factor = np.cos(np.sqrt(x_rad**2 + y_rad**2))

        if debug:
            print(f"factor:{factor}\nfactor shape: {factor.shape}\n\n")
            print(f"factor at center:{factor[(center_x, center_y)]}\n\n")

        min_factor = np.unravel_index(factor.argmin(), factor.shape)
        max_factor = np.unravel_index(factor.argmax(), factor.shape)

        if debug:
            print(f"factor min: {factor.min(), min_factor}\n\n")
            print(f"factor max: {factor.max(), max_factor}\n\n")

        # thickness_vals_type = thickness_vals.dtype
        # thickness_vals = factor*thickness_vals.astype(thickness_vals_type)

        rdm_type = rdm.dtype
        adjusted_rdm = factor * rdm.astype(rdm_type)
        adjusted_rdm = np.clip(adjusted_rdm,a_min=0,a_max=None)

        corrected_thickness_vals = adjusted_rdm[(ridge == 4) & (adjusted_rdm != 0)]
        
        if verbose or debug:
            print(
                f"Corrected thickness mean:\n{corrected_thickness_vals.mean()}\nCorrected thickness max: {corrected_thickness_vals.max()}\n"
            )

    if thickness_vals.size == 0:
        print("[WARN] No overlapping ridge found. Returning NaN.")

        if nearest_neighbor_calc:
            viewer.add_image(thickness_map,name=f"{key}_thickness_map")
            viewer.add_labels(retina_mask.astype("uint8"),name=f"{key}_retina_mask")
            viewer.add_labels(ridge,name=f"{key}_ridge")

        return float("nan"), float("nan")

    if incedence_correction:
        thickness_vals = corrected_thickness_vals
    if micrometer_output:
        conv_factor = imaging_range / ret_chor.shape[1] * 1000 / refractive_index # imaging range in mm / ascan len in pixels * um/mm * refractive index ratio = um/pixel
        thickness_vals = conv_factor * thickness_vals # um/pixel * pixels = um
        micrometer_rdm = conv_factor * adjusted_rdm

        if verbose or debug:
            print(
                f"Micrometer thickness mean:\n{thickness_vals.mean()}\nMicrometer thickness max: {thickness_vals.max()}\n"
            )
    if nearest_neighbor_calc:
        thickness_vals = thickness_map[(ridge == 4) & (thickness_map != 0)]

        if verbose or debug:
            print(
                f"Nearest neighbor thickness mean:\n{thickness_vals.mean()}\nNearest neighbor thickness max: {thickness_vals.max()}\n"
            )


    # if display_in_napari:
    #     viewer = napari.Viewer()
    if viewer:
        #viewer.add_image(en_face_circular_mask)
        viewer.add_labels(retina_mask.astype("uint8"),name=f"{key}_retina_mask")
        #viewer.add_image(rdm)
        if incedence_correction:
            #viewer.add_image(x_rad)
            #viewer.add_image(y_rad)
            #viewer.add_image(factor)
            viewer.add_image(adjusted_rdm,name=f"{key}_incedence_angle_adjusted_rdm")
        if micrometer_output:
            viewer.add_image(micrometer_rdm,name=f"{key}_micrometer_rdm")
        if nearest_neighbor_calc:
            viewer.add_image(thickness_map,name=f"{key}_thickness_map")
            viewer.add_labels(ridge,name=f"{key}_ridge")

    if verbose:
        print(f"thickness_vals dtype: {thickness_vals.dtype}\n")
        

    return thickness_vals.mean(), thickness_vals.max()


def write_to_csv(
    retchor_path: Path,
    mean_t: float,
    peak_t: float,
    output_file: Path = Path("ridge_analysis_results.csv"),
    display_dataframe: bool = False,
    write_output: bool = True,
):
    name = retchor_path.name
    suffix = "_ret_chor_seg.npy"
    if name.endswith(suffix):
        label = name[: -len(suffix)]
    else:
        label = name

    identifier_start_index = retchor_path.name.find("_") + 1
    indentifier_end_index = retchor_path.name.find("_processed")
    identifier = retchor_path.name[identifier_start_index:indentifier_end_index]
    #print(f"identifier: {identifier}\n")

    current_thickness_df = pl.DataFrame(
        #{"Filename": label, "Mean": mean_t, "Peak": peak_t}
        {"Identifier": identifier, "Mean": mean_t, "Peak": peak_t}
    )

    if output_file.exists():
        prior_thickness_df = pl.read_csv(output_file)
        current_thickness_df = pl.concat(
            [prior_thickness_df, current_thickness_df], how="vertical"
        )
    else:
        output_file.parent.mkdir(parents=True,exist_ok=True)

    if display_dataframe:
        with pl.Config(tbl_cols=-1, tbl_rows=-1, set_tbl_width_chars=2000):
            print(current_thickness_df)

    if write_output:
        current_thickness_df.write_csv(output_file)


def write_to_excel(
    retchor_path: Path,
    mean_t: float,
    peak_t: float,
    excel_file: Path = Path("ridge_analysis_results.xlsx"),
):
    name = retchor_path.name
    suffix = "_ret_chor_seg.npy"
    if name.endswith(suffix):
        label = name[: -len(suffix)]
    else:
        label = name

    row = pd.DataFrame([{"Filename": label, "Mean": mean_t, "Peak": peak_t}])

    if not excel_file.exists():
        row.to_excel(excel_file, index=False)
    else:
        wb = load_workbook(excel_file)
        ws = wb.active
        ws.append([label, mean_t, peak_t])
        wb.save(excel_file)


def run_cli(ridge_path: Path, retchor_path: Path):
    if ridge_path.suffix.lower() == ".npy":
        ridge = np.load(ridge_path)
    elif ridge_path.suffix.lower() == ".mat":
        mat = loadmat(str(ridge_path))
        ridge = np.squeeze(mat.get("ridge"))
    else:
        print(f"Unsupported ridge format: {ridge_path.suffix}")
        sys.exit(1)

    ret_chor = np.load(retchor_path)
    mean_t, peak_t = ridge_analysis(ridge, ret_chor)
    print(
        f"{ridge_path.name} vs {retchor_path.name} → Mean: {mean_t:.2f}, Peak: {peak_t:.2f}"
    )
    write_to_excel(retchor_path, mean_t, peak_t)
    print(f"Results written to '{Path.cwd() / 'ridge_analysis_results.xlsx'}'")


def collect_pairs(ridge_paths, retchor_paths):
    ret_map = {}
    for p in retchor_paths:
        if "_processed_" in p.name:
            prefix = p.name.split("_processed_")[0] + "_processed_"
            if prefix not in ret_map:
                ret_map[prefix] = p
            else:
                print(
                    f"[WARN] Duplicate retchor prefix '{prefix}': {ret_map[prefix]} and {p}. Using the first."
                )
        else:
            print(f"[WARN] RetChor file '{p.name}' missing '_processed_'. Skipping.")

    pairs = []
    for r in ridge_paths:
        if "_processed_" in r.name:
            prefix = r.name.split("_processed_")[0] + "_processed_"
            if prefix in ret_map:
                pairs.append((r, ret_map[prefix]))
            else:
                print(
                    f"[WARN] No matching RetChor file for Ridge '{r.name}' (prefix='{prefix}')."
                )
        else:
            print(f"[WARN] Ridge file '{r.name}' missing '_processed_'. Skipping.")

    return pairs

def collect_file_paths(ridge_paths, ret_chor_paths, xml_paths):
    ret_map = {}
    for ret_chor_path in ret_chor_paths:
        if "_processed_" in ret_chor_path.name:
            prefix = ret_chor_path.name.split("_processed_")[0] #+ "_processed_"
            if "UNPs_" in prefix:
                prefix = prefix.split("UNPs_")[1]
            if prefix not in ret_map:
                ret_map[prefix] = {}
                ret_map[prefix]["ret_chor"] = ret_chor_path
            else:
                print(
                    f"[WARN] Duplicate retchor prefix '{prefix}': {ret_map[prefix]} and {ret_chor_path}. Using the first."
                )
        else:
            print(f"[WARN] RetChor file '{ret_chor_path.name}' missing '_processed_'. Skipping.")

    #pairs = []
    for ridge_path in ridge_paths:
        if "_processed_" in ridge_path.name:
            prefix = ridge_path.name.split("_processed_")[0] #+ "_processed_"
            if "UNPs_" in prefix:
                prefix = prefix.split("UNPs_")[1]
            if prefix in ret_map:
                ret_map[prefix]["ridge"] = ridge_path
                #pairs.append((r, ret_map[prefix]))
            else:
                print(
                    f"[WARN] No matching RetChor file for Ridge '{ridge_path.name}' (prefix='{prefix}')."
                )
        else:
            print(f"[WARN] Ridge file '{ridge_path.name}' missing '_processed_'. Skipping.")

    for xml_path in xml_paths:

        prefix = xml_path.stem
        if "UNPs_" in prefix:
            prefix = prefix.split("UNPs_")[1]

        print(f"path: {xml_path} with prefix {prefix}\n")

        if prefix in ret_map:
            #pairs.append((xml_path, ret_map[prefix]))
            ret_map[prefix]["xml"] = xml_path
        else:
            print(
                f"[WARN] No matching RetChor file for xml '{xml_path.name}' (prefix='{prefix}')."
            )

    return ret_map

def get_motor_pos_from_xml(xml_path:Path):
    """"""
    motor_pos = None
    if xml_path.exists():
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            scanning_params = root.find(".//Scanning_Parameters")
            if scanning_params is not None:
                motor_pos = int(scanning_params.attrib.get("Motor_Pos"))
        except Exception as e:
            print(f"Error parsing {xml_path}: {e}")

    if motor_pos is None:
        print(f"No motor_pos found for {xml_path}, skipping...")
    
    return motor_pos


def run_batch(
    ridge_dir: Path,
    ret_chor_dir: Path,
    xml_dir: Path,
    output: Literal[".xlsx", ".csv", "none"] = ".csv", #".xlsx"
    output_dir_path=Path("ridge_analysis_output"),
    display_dataframe: bool = False,
    scan_angle: float = 105.0,
    imaging_range: float = 12.0, #6.0,
    reference_motor_position: float = 85.0,
    refractive_index: float = 1.33,
    circular_mask_scale: float = 1.0,
    mode: Literal["center", "optic disk", "fovea"] = "center",
    nearest_neighbor_calc: bool = True,
    incedence_correction: bool = True,
    micrometer_output: bool = True,
    #display_in_napari: bool = False,
    verbose: bool = True,
    debug: bool = False,
    viewer: napari.Viewer = None,
):
    # ridge_paths = list(ridge_dir.rglob("*_en_face_ridge_labels.npy")) + \
    #               list(ridge_dir.rglob("*.mat"))
    # retchor_paths = list(retchor_dir.rglob("*_ret_chor_seg.npy"))
    ridge_paths = list(ridge_dir.rglob("*.npy")) + list(ridge_dir.rglob("*.mat"))
    ridge_paths = [path for path in ridge_paths if "ridge" in str(path)]
    
    ret_chor_paths = list(ret_chor_dir.rglob("*.npy")) + list(ridge_dir.rglob("*.mat"))
    ret_chor_paths = [path for path in ret_chor_paths if "ret_chor" in str(path)]
    
    xml_paths = list(xml_dir.rglob("*.xml"))

    pairs = collect_pairs(ridge_paths, ret_chor_paths)

    ret_map = collect_file_paths(ridge_paths,ret_chor_paths,xml_paths)  
    print(f"ret_map: {ret_map}\n")

    # if not pairs:
    #     print("No matching ridge/retchor pairs found. Exiting.")
    #     return
    
    if not ret_map:
        print("No matching ridge/retchor/xml files found. Exiting.")
        return
    
    print()

    #print(f"Found {len(pairs)} matched pairs. Processing...\n")
    print(f"Found {len(ret_map)} matched files. Processing...\n")


    #for ridge_path, retchor_file in tqdm(pairs,desc="Pair"):
    keys = list(ret_map.keys())
    pbar = tqdm(keys)
    for key in pbar:
        pbar.set_description(f"Processing: {key}")


        ret_chor_path = ret_map[key]["ret_chor"] # keys are generated from retchor file so this must exist
        if "ridge" in ret_map[key]:
            ridge_path = ret_map[key]["ridge"]
        else:
            print(f"{key} is missing the accompanying ridge file.")
            continue
        if "xml" in ret_map[key]:
            xml_path = ret_map[key]["xml"]
        else:
            print(f"{key} is missing the accompanying xml file.")
            #continue
            pass

        if ridge_path.suffix.lower() == ".npy":
            ridge_label = np.load(ridge_path)
        else:
            mat = loadmat(str(ridge_path))
            ridge_label = np.squeeze(mat.get("ridge"))


        ret_chor_labels = np.load(ret_chor_path)

        imaging_motor_position = get_motor_pos_from_xml(xml_path=xml_path) / 1000

        if debug:
            #print(f"Motor position from {xml_path}:\n{imaging_motor_position}\nrefrence arm shift: {reference_arm_shift}\n")
            print(f"Imaging motor position from {xml_path}:\n{imaging_motor_position}\n")

        # DEBUG: Print shapes
        if debug:
            print(
                f"[DEBUG] retchor shape: {ret_chor_labels.shape}, ridge shape: {ridge_label.shape}"
            )

        # if display_in_napari:
        #     viewer = napari.Viewer(show=False)

        mean_t, peak_t = ridge_analysis(
            ridge_label,
            ret_chor_labels,
            scan_angle=scan_angle,
            imaging_range=imaging_range,
            reference_motor_position=reference_motor_position,
            imaging_motor_position=imaging_motor_position,
            refractive_index=refractive_index,
            circular_mask_scale=circular_mask_scale,
            mode=mode,
            nearest_neighbor_calc=nearest_neighbor_calc,
            incedence_correction=incedence_correction,
            micrometer_output=micrometer_output,
            #display_in_napari=display_in_napari,
            viewer=viewer,
            key=key,
            verbose=verbose,
            debug=debug,
        )

        if verbose:
            print(
                f"{ridge_path.name} vs {ret_chor_path.name} → Mean: {mean_t:.2f}, Peak: {peak_t:.2f}"
            )

        identifier_start_index = ret_chor_path.name.find("_") + 1
        indentifier_end_index = ret_chor_path.name.find("_processed")
        identifier = ret_chor_path.name[identifier_start_index:indentifier_end_index]
        
        if verbose:
            print(f"identifier: {identifier}\n")

        # if viewer is not None:
        #     pts = np.array([[0, 0]])
        #     viewer.add_points(
        #         pts,
        #         text=[f"{ridge_path.name}\nMean: {mean_t:.2f}\nPeak: {peak_t:.2f}"],
        #         size=0,
        #         name=f"Batch: {ridge_path.stem}",
        #     )

        match output:
            case ".xlsx":
                write_to_excel(
                    ret_chor_path,
                    mean_t,
                    peak_t,
                    excel_file=output_dir_path / "ridge_analysis_results.xlsx",
                )
            case ".csv":
                write_to_csv(
                    ret_chor_path,
                    mean_t,
                    peak_t,
                    output_file=output_dir_path / "ridge_analysis_results.csv",
                    display_dataframe=display_dataframe,
                    write_output=True,
                )
            case "none":
                    write_to_csv(
                    ret_chor_path,
                    mean_t,
                    peak_t,
                    output_file=output_dir_path / "ridge_analysis_results.csv",
                    display_dataframe=display_dataframe,
                    write_output=False,
                )
                    
    if viewer != None:
        viewer.show()
        napari.run()

    # TODO fix this as it is incorrect
    print(
        f"\nBatch complete. Results appended to '{Path.cwd() / 'ridge_analysis_results.xlsx'}'."
    )

@magicgui(
    ridge_dir={"label": "Path to folder containing ridge masks.", "mode": "d"},
    ret_chor_dir={"label": "Path to folder containing retchor masks", "mode": "d"},
    xml_dir={"label": "Path to folder containing .xml metadata files", "mode": "d"},
    output_dir_path={"label": "Path to output results", "mode": "d"},
    call_button="Run Batch Analysis",
)
def generate_enface_with_labels(
    #ridge_dir: Path = Path(r"F:\38 peak stage 2"),
    ridge_dir: Path = Path(r"C:\Beth\TestTrueThickness\Ridge"),
    #ret_chor_dir: Path = Path(r"F:\38 peak stage 2"),
    ret_chor_dir: Path = Path(r"C:\Beth\TestTrueThickness\RetChorMasks"),
    xml_dir: Path = Path(r"F:\xmls"),
    output_dir_path: Path = Path(r"C:\Beth\TestTrueThickness\Output"),
    output: Literal[".xlsx", ".csv", "none"] = ".csv", #.xlsx",
    display_dataframe: bool = True,
    scan_angle: float = 105.0,
    imaging_range: float = 12.0, #6.0,
    reference_motor_position: float = 85.0,
    refractive_index: float = 1.33,
    circular_mask_scale: float = 0.85, #1.0,
    mode: Literal["center", "optic disk", "fovea"] = "center",
    nearest_neighbor_calc: bool = True,
    incedence_correction: bool = False, #True,
    micrometer_output: bool = False, #True,
    display_in_napari: bool = False,
    verbose: bool = False, #True,
    debug: bool = False,
):
    
    if display_in_napari:
        viewer = napari.Viewer(show=False)
    else:
        viewer = None

    run_batch(
        ridge_dir,
        ret_chor_dir,
        xml_dir=xml_dir,
        output=output,
        output_dir_path=output_dir_path,
        display_dataframe=display_dataframe,
        scan_angle=scan_angle,
        imaging_range=imaging_range,
        reference_motor_position=reference_motor_position,
        refractive_index=refractive_index,
        circular_mask_scale=circular_mask_scale,
        mode=mode,
        nearest_neighbor_calc=nearest_neighbor_calc,
        incedence_correction=incedence_correction,
        micrometer_output=micrometer_output,
        #display_in_napari=display_in_napari,
        viewer=viewer,
        verbose=verbose,
        debug=debug,
    )



if __name__ == "__main__":
    generate_enface_with_labels.show(run=True)
