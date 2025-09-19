import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pyvista as pv
import numpy as np
from scipy.linalg import lstsq
from scipy.interpolate import splprep, splev
import pandas as pd
import tkinter as tk
from tkinter import filedialog
from scipy.spatial import cKDTree
import os
import glob
import xml.etree.ElementTree as ET

from pathlib import Path
from typing import Literal


from numpy.typing import ArrayLike
from magicgui import magicgui
import napari



def convert_spherical(coord):
    #coord = [theta_x, theta_y, r]
    #theta_x and theta_y are in radians units
    theta_x = coord[0]
    theta_y = coord[1]
    r = coord[2]
    theta = np.arctan2(theta_x,theta_y)
    phi = np.sqrt(theta_x**2 + theta_y**2)

    x = r*np.sin(phi)*np.cos(theta)
    y = r*np.sin(phi)*np.sin(theta)
    z = r*np.cos(phi)
    return (x,y,z)

def circular_mask_3D(data, center, radius):
    """
    Creates a 3D mask where values outside a circular region in the XY plane are masked out.
    
    Parameters:
        array_3d (np.ndarray): Input 3D array with shape (Z, Y, X).
        center (tuple): (y, x) coordinates of the circle center.
        radius (float): Radius of the circular mask.
    
    Returns:
        np.ndarray: Boolean 3D mask with the same shape as array_3d.
    """
    y_dim, z_dim, x_dim = data.shape  # Extract shape
    Y, X = np.ogrid[:y_dim, :x_dim]  # Create 2D coordinate grids
    dist_from_center = (X - center[1])**2 + (Y - center[0])**2  # Compute squared distance
    mask_2d = dist_from_center <= radius**2  # Create 2D mask
    
    return mask_2d[:, None, :].repeat(z_dim, axis=1)  # Expand to 3D

def RPE_layer_coords(mask):
    mask_r = mask == 1
    interfaceMap = np.where(mask_r.any(axis=1), mask_r.shape[1] -1- np.argmax(mask_r[:, ::-1, :], axis=1), -1)
    coords = valid_coordinates(interfaceMap)
    return coords

def Retinal_surface_coords(mask):
    mask_r = mask ==1
    interfaceMap = np.where(mask_r.any(axis =1), np.argmax(mask_r, axis =1), -1)
    coords = valid_coordinates(interfaceMap)
    return coords

def MIP_coords(data, mask):
    mask_r = mask ==1
    MIP_z = np.argmax(data,axis=1)
    MIP_map = np.where(mask_r.any(axis=1) > 0, MIP_z, -1)
    coords = valid_coordinates(MIP_map)
    return coords

def valid_coordinates(z_array):
    # Find where z is not -1
    mask = z_array != -1

    # Get x and y coordinates
    y_indices, x_indices = np.where(mask)

    # Get the corresponding z values
    z_values = z_array[y_indices, x_indices]

    # Stack into (N, 3) format: (x, y, z)
    coordinates = np.column_stack((y_indices,z_values, x_indices))

    return coordinates

def fit_sphere(points):
    """Fit a sphere to 3D points and return the sphere's center and radius."""
    # Extract x, y, z coordinates
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    # Formulate the linear system
    A = np.c_[2*x, 2*y, 2*z, np.ones(len(points))]
    b = x**2 + y**2 + z**2

    # Solve the linear system using least squares
    c, _, _, _ = lstsq(A, b)
    x_c, y_c, z_c, d = c

    # Calculate the radius
    radius = np.sqrt(x_c**2 + y_c**2 + z_c**2 + d)

    return (x_c, y_c, z_c), radius

def sphere_fit_thick_map(mask, refractive_index=1.33, imaging_motor_position=0.0, reference_motor_position=85.0, imaging_range=6, pivot_point=19.2, scan_angle=105,ret_to_rpe:bool=True):
    """
    Optimized function to fit a sphere to the retinal surface and compute z-difference.
    Returns Cartesian coordinates and color values for visualization.
    """
    
    y_shape, z_shape, x_shape = mask.shape
    print(f"retchor_mask shape: {mask.shape}\n")
    #x_shift, y_shift = [0,0]
    imaging_range /= refractive_index
    pixel_spacing = imaging_range / z_shape
    center = np.array([x_shape // 2 , y_shape // 2])

    # Adjust reference arm shift
    reference_arm_shift = reference_motor_position - imaging_motor_position
    reference_arm_shift = (reference_arm_shift * 0.5) / refractive_index
    print(f"reference_arm_shift: {reference_arm_shift}\n")
    #padding = pivot_point - imaging_range - reference_arm_shift
    padding = pivot_point - imaging_range + reference_arm_shift # this is from fixing Dr. Young curve correction to match Yakub curve correction
    
    padding_pixel = int(padding / pixel_spacing)

    # Circular masking to remove artifacts
    #radius = int(x_shape / rad_clip)
    #circ_mask = circular_mask_3D(mask, [0,0], radius)
    #mask = np.where(circ_mask, mask, 0)
    #data = np.where(circ_mask, data, 0)

    # Extract layer coordinates
    #note to self, this essentially calls valid_coordinates three separate times; for efficiency can probably condense
    rpe_coords = np.array(RPE_layer_coords(mask))  # Shape: (N, 3)
    retina_coords = np.array(Retinal_surface_coords(mask))  # Shape: (M, 3)
    #mip_coords = np.array(MIP_coords(data,mask))
    #MIP = np.max(data, axis=1)
    #MIP = MIP/np.max(MIP)
    #opacity = MIP*2

    print(f"retina_coords len: {len(retina_coords)}\n\nrpe_coords len: {len(rpe_coords)}\n")
    

    # Extract the corresponding MIP opacity values using the valid y, x indices from valid_coords
    #MIP_y, MIP_z, MIP_x = mip_coords.T
    #opacity = opacity[MIP_y, MIP_x]
    # Convert 3D coordinates to 2D indices
    rpe_y, rpe_z, rpe_x = rpe_coords.T
    retina_y, retina_z, retina_x = retina_coords.T
    standard_height = rpe_z - retina_z
    # Vectorized spherical to Cartesian conversion
    def spherical_to_cartesian(y, z, x):
        p_t_x = ((x - center[0]) / (x_shape / 2)) * (scan_angle / 2) * (np.pi / 180)
        p_t_y = ((y - center[1]) / (y_shape / 2)) * (scan_angle / 2) * (np.pi / 180)
        p_r = z + padding_pixel
        #return np.array(convert_spherical([p_t_x, p_t_y, p_r])) * pixel_spacing

        spherical_tuple = convert_spherical([p_t_x,p_t_y,p_r])

        return np.array((spherical_tuple[0],spherical_tuple[2],spherical_tuple[1])) #* pixel_spacing

    # Apply spherical-to-Cartesian conversion vectorized
    #cart_coords= np.array([spherical_to_cartesian(y, z, x) for y, z, x in retina_coords])
    curv_ret_coords= np.array([spherical_to_cartesian(y, z, x) for y, z, x in retina_coords])
    #rpe_coords = np.array([spherical_to_cartesian(y, z, x) for y, z, x in rpe_coords])
    curv_rpe_coords = np.array([spherical_to_cartesian(y, z, x) for y, z, x in rpe_coords])

    print(f"retina_coords len: {len(curv_ret_coords)}\n\nrpe_coords len: {len(curv_rpe_coords)}\n")

    if ret_to_rpe:
        #tree = cKDTree(cart_coords) #switch to avoid mesuring distance to base of curve
        tree = cKDTree(curv_rpe_coords)
    else:
        #tree = cKDTree(rpe_coords)
        tree = cKDTree(curv_ret_coords) #switch to avoid mesuring distance to base of curve
        
    
    
    # Query nearest neighbor for each inner point

    if ret_to_rpe:
        #curve_correct_height, _ = tree.query(rpe_coords, k=1) #switch to avoid mesuring distance to base of curve
        curve_correct_height, _ = tree.query(curv_ret_coords, k=1)
    else:
        #curve_correct_height, _ = tree.query(cart_coords, k=1)
        curve_correct_height, _ = tree.query(curv_rpe_coords, k=1) #switch to avoid mesuring distance to base of curve

    #convert from mm to pixels
    #curve_correct_height = curve_correct_height /((6/(mask.shape[1]*1.33))) # I believe the magic number 6 needs to be the imaging range and 1.33 is the index of refraction of water.
    
    #curve_correct_height = curve_correct_height /((imaging_range/(mask.shape[1]*refractive_index)))

    conv_factor = pixel_spacing * 1000 / refractive_index # mm/pixel * 1000 um/mm / refractive index = um/pixel
    curve_correct_height = curve_correct_height * conv_factor

    #z_diffs = rpe_z-retina_z
    #thickness_map = np.full((y_shape, x_shape), np.nan)  # use NaN for missing pixels
    thickness_map = np.full((y_shape, x_shape), 0.0)  # use NaN for missing pixels

    # Fill thickness map at (y, x) positions
    thickness_map[rpe_y, rpe_x] = curve_correct_height

    #return thickness_map
    #return thickness_map, cart_coords, rpe_coords
    return thickness_map, retina_coords, rpe_coords, curv_ret_coords, curv_rpe_coords

def load_mask():
    """ Open a file dialog to select the data and mask files. """
    root = tk.Tk()
    root.withdraw()  # Hide the root window

    # Prompt user to select the mask file
    mask_file = filedialog.askopenfilename(title="Select Mask File", filetypes=[("NumPy files", "*.npy"), ("All files", "*.*")])
    if not mask_file:
        print("No mask file selected.")
        return None, None

    # Load the NumPy arrays
    mask = np.load(mask_file)

    #print(f"Loaded data shape: {data.shape}")
    print(f"Loaded mask shape: {mask.shape}")

    return mask

def collect_pairs_xml(retchor_paths, xml_paths):
    ret_map = {}
    for p in retchor_paths:
        if "_processed_" in p.name:
            #prefix = p.name.split("_processed_")[0] + "_processed_"
            prefix = p.stem.split("_processed_")[0]
            if "UNPs_" in prefix:
                prefix = prefix.split("UNPs_")[1]

            print(f"path: {p} with prefix: {prefix}\n")

            if prefix not in ret_map:
                ret_map[prefix] = p
            else:
                print(
                    f"[WARN] Duplicate retchor prefix '{prefix}': {ret_map[prefix]} and {p}. Using the first."
                )
        else:
            print(f"[WARN] RetChor file '{p.name}' missing '_processed_'. Skipping.")

    pairs = []
    for r in xml_paths:

        prefix = r.stem

        print(f"path: {r} with prefix {prefix}\n")

        if prefix in ret_map:
            pairs.append((r, ret_map[prefix]))
        else:
            print(
                f"[WARN] No matching RetChor file for Ridge '{r.name}' (prefix='{prefix}')."
            )

    return pairs

def isolate_mask_by_value(mask,value:int=1):
    """"""
    isolated_mask = mask.copy()
    target_mask = isolated_mask == 1
    isolated_mask[~target_mask] = 0
    return isolated_mask

def generate_depth_map(retina_mask,normalized:bool=True,axis=1):
    """"""
    from napari_cool_tools_img_proc._normalization_funcs import normalize_data_in_range_func
    depth = retina_mask.shape[1]
    raw_depth = retina_mask.sum(axis=axis)

    if not normalized:
        return raw_depth
    else:
        normalized_depth = raw_depth.astype("float32") / float(depth)
        normalized_depth = normalize_data_in_range_func(normalized_depth,min_val=0.0,max_val=1.0)
        return normalized_depth

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

# mouse cursor callback function
def display_closest_points_retina_rpe(ret_points_layer,event):
    """"""
    # mouse click
    yield

    # mouse move
    while event.type == "mouse_move":
        value = ret_points_layer.get_value(event.position)
        if value != None:
            print(value)
        yield

    # mouse release
    pass

def run_batch(
    ridge_dir: Path,
    retchor_dir: Path,
    xml_dir: Path,
    output: Literal[".xlsx", ".csv", "none"] = ".xlsx",
    output_dir_path=Path("ridge_analysis_output"),
    display_dataframe: bool = False,
    scan_angle: float = 106.0,
    imaging_range: float = 6.0,
    refractive_index: float = 1.33,
    mode: Literal["center", "optic disk", "fovea"] = "center",
    incedence_correction: bool = True,
    ret_to_rpe:bool=True,
    micrometer_output: bool = False,
    display_in_napari: bool = False,
    verbose: bool = True,
    debug: bool = False,
    viewer: napari.Viewer = None,
):

    retchor_paths = list(retchor_dir.rglob("*.npy"))
    xml_paths = list(xml_dir.rglob("*.xml"))

    pairs = collect_pairs_xml(retchor_paths,xml_paths)

    if not pairs:
        print("No matching ridge/retchor pairs found. Exiting.")
        return

    print(f"Found {len(pairs)} matched pair(s). Processing...\n")

    for xml_path,retchor_path in pairs:
        imaging_motor_position = get_motor_pos_from_xml(xml_path) / 1000
        print(f"imaging_motor_position: {imaging_motor_position}\n")
        retchor_mask = np.load(retchor_path)

        #try:

        #thick_map = sphere_fit_thick_map(retchor_mask, reference_arm_shift=reference_arm_shift,imaging_range=imaging_range)
        scale =  1 / 4.0
        axial_shift = (1178/2) - 164 #166 #172
        radial_shift = 934

        thick_map, ret_surf_coords, rpe_surf_coords, curv_ret_surf_coords, curv_rpe_surf_coords = sphere_fit_thick_map(retchor_mask, refractive_index=refractive_index, imaging_motor_position=imaging_motor_position,imaging_range=imaging_range)
        print(f"ret_surf_coords:\n{ret_surf_coords}\n\nrpe_coords:\n{rpe_surf_coords}\n\n")
        print(f"curv_ret_surf_coords:\n{curv_ret_surf_coords}\n\ncurv_rpe_coords:\n{curv_rpe_surf_coords}\n\n")
        
        curv_ret_surf_shifted_coords = curv_ret_surf_coords.copy() * scale
        curv_ret_surf_shifted_coords[:,0] += radial_shift
        curv_ret_surf_shifted_coords[:,1] -= axial_shift
        curv_ret_surf_shifted_coords[:,2] += radial_shift

        curv_rpe_surf_shifted_coords = curv_rpe_surf_coords.copy() * scale
        curv_rpe_surf_shifted_coords[:,0] += radial_shift
        curv_rpe_surf_shifted_coords[:,1] -= axial_shift
        curv_rpe_surf_shifted_coords[:,2] += radial_shift

        print(f"scaled_curv_ret_surf_coords:\n{curv_ret_surf_coords * scale}\n\nscaled_curv_rpe_coords:\n{curv_rpe_surf_coords * scale}\n\n")

        target = ret_surf_coords == [494,1630,202]
        ret_idxs = [idx for idx in range(len(ret_surf_coords)) if ret_surf_coords[idx][0] == 494 and ret_surf_coords[idx][2] == 202]
        rpe_idxs = [idx for idx in range(len(rpe_surf_coords)) if rpe_surf_coords[idx][0] == 494 and rpe_surf_coords[idx][2] == 202]
        print(ret_idxs,ret_surf_coords[ret_idxs[0]],curv_ret_surf_shifted_coords[ret_idxs[0]])
        #print(rpe_idxs,rpe_coords[rpe_idxs[0]],curv_rpe_surf_coords[rpe_idxs[0]])
        print(rpe_idxs,rpe_surf_coords[rpe_idxs[0]],curv_rpe_surf_shifted_coords[rpe_idxs[0]])

        curv_target_coords = curv_ret_surf_shifted_coords[ret_idxs[0]]
        # tree = cKDTree(curv_ret_surf_shifted_coords[ret_idxs[0]][None,:])
        # result = tree.query(curv_rpe_surf_shifted_coords, k=1)
        tree = cKDTree(curv_rpe_surf_shifted_coords)
        result = tree.query(curv_target_coords[None,:], k=1)
        nearest_coords = tree.data[result[1]].squeeze()
        print(f"tree query result: {result}\ndata at result index: {nearest_coords}\n")

        nearest_neigbor_line_data = np.array([curv_target_coords,nearest_coords])



        print(curv_ret_surf_shifted_coords[ret_idxs[0]])
        print(curv_rpe_surf_shifted_coords[rpe_idxs[0]])

        # generate depth_map
        retina_mask = isolate_mask_by_value(retchor_mask,value=1)
        raw_retina_depth_map = generate_depth_map(retina_mask,normalized=False)
        normalized_retinal_depth_map = generate_depth_map(retina_mask,normalized=True)

        viewer = napari.Viewer(show=False)
        viewer.add_image(raw_retina_depth_map)
        viewer.add_image(normalized_retinal_depth_map)
        viewer.add_image(thick_map)
        #viewer.add_points(ret_surf_coords * 100,size=4,face_color="red",name="retinal_surface")
        #viewer.add_points(rpe_coords * 100,size=4,face_color="white",name="rpe_surface")
        viewer.add_points(ret_surf_coords,size=4,face_color="red",name="retinal_surface")
        viewer.add_points(rpe_surf_coords,size=4,face_color="white",name="rpe_surface")
        # viewer.add_points(curv_ret_surf_coords/scale,size=4,face_color="red",name="retinal_surface")
        # viewer.add_points(curv_rpe_coords/scale,size=4,face_color="white",name="rpe_surface")
        viewer.add_points(curv_ret_surf_shifted_coords,size=4,face_color="red",name="curv_retinal_surface_shifted")
        viewer.add_points(curv_rpe_surf_shifted_coords,size=4,face_color="white",name="curv_rpe_surface_shifted")
        viewer.add_shapes(
            nearest_neigbor_line_data,
            shape_type="line",
            edge_width=1,
            edge_color="green",
            name="nearest_neighbors_link"
        )
        viewer.add_points(nearest_neigbor_line_data,size=10,face_color="yellow",name="nearest_neighbors")

        # Connect the callback to the viewer's mouse move event
        #viewer.mouse_move_callbacks.append(on_mouse_move)
        #viewer.layers["retinal_surface"].mouse_drag_callbacks.append(display_closest_points_retina_rpe)

        viewer.show()
        napari.run()

        # except Exception as e:
        #     print(f"Error running sphere_fit_comp on {retchor_path}: {e}")
        #     result = None


@magicgui(
    ridge_dir={"label": "Path to folder containing ridge masks", "mode": "d"},
    retchor_dir={"label": "Path to folder containing retchor masks", "mode": "d"},
    xml_dir={"label": "Path to folder containing xml files", "mode": "d"},
    output_dir_path={"label": "Path to output results", "mode": "d"},
    call_button="Run Batch Analysis",
)
def generate_enface_with_labels(
    ridge_dir: Path = Path(r"C:\Beth\TestTrueThickness\RetChorMasks"),
    retchor_dir: Path = Path(r"C:\Beth\TestTrueThickness\RetChorMasks"),
    xml_dir: Path = Path(r"C:\Beth\TestTrueThickness\XMLs"),
    output_dir_path: Path = Path(r"F:\Beth_RetChor_Stuff\output"),
    output: Literal[".xlsx", ".csv", "none"] = ".xlsx",
    display_dataframe: bool = True,
    scan_angle: float = 105, #106.0,
    imaging_range: float = 12.0, #6.0,
    refractive_index: float = 1.33,
    mode: Literal["center", "optic disk", "fovea"] = "center",
    incedence_correction: bool = True,
    ret_to_rpe:bool=True,
    micrometer_output: bool = False,
    display_in_napari: bool = False,
    verbose: bool = True,
    debug: bool = False,
):
    run_batch(
        ridge_dir,
        retchor_dir,
        xml_dir,
        output=output,
        output_dir_path=output_dir_path,
        display_dataframe=display_dataframe,
        scan_angle=scan_angle,
        imaging_range=imaging_range,
        refractive_index=refractive_index,
        mode=mode,
        incedence_correction=incedence_correction,
        ret_to_rpe=ret_to_rpe,
        micrometer_output=micrometer_output,
        display_in_napari=display_in_napari,
        verbose=verbose,
        debug=debug,
    )


if __name__ == "__main__":
    generate_enface_with_labels.show(run=True)

# def process_masks_with_xml():
#     # Select folders interactively
#     root = tk.Tk()
#     root.withdraw()
#     mask_folder = filedialog.askdirectory(title="Select folder containing *_processed_ret_chor_seg.npy files")
#     xml_folder = filedialog.askdirectory(title="Select folder containing .xml files")

#     if not mask_folder or not xml_folder:
#         print("No folder(s) selected.")
#         return None

#     results = []

#     # Get all processed .npy files
#     npy_files = glob.glob(os.path.join(mask_folder, "*_processed_ret_chor_seg.npy"))

#     for npy_path in npy_files:
#         base = os.path.basename(npy_path)

#         # Strip suffix to get core name
#         core_name = base.replace("_processed_ret_chor_seg.npy", "")
#         # Remove UNPs_ if present
#         core_name = core_name.replace("UNPs_", "")

#         # Construct XML path in the xml_folder
#         xml_path = os.path.join(xml_folder, core_name + ".xml")

#         motor_pos = None
#         if os.path.exists(xml_path):
#             try:
#                 tree = ET.parse(xml_path)
#                 root = tree.getroot()
#                 scanning_params = root.find(".//Scanning_Parameters")
#                 if scanning_params is not None:
#                     motor_pos = int(scanning_params.attrib.get("Motor_Pos"))
#             except Exception as e:
#                 print(f"Error parsing {xml_path}: {e}")

#         if motor_pos is None:
#             print(f"No motor_pos found for {npy_path}, skipping...")
#             continue

#         # Load mask
#         mask = np.load(npy_path)

#         # Compute reference_arm from motor_pos
#         reference_arm = motor_pos / 10000

#         # Run your analysis
#         try:
#             thick_map = sphere_fit_thick_map(mask, reference_arm=reference_arm)
#         except Exception as e:
#             print(f"Error running sphere_fit_comp on {npy_path}: {e}")
#             result = None

#         results.append({
#             "npy_file": npy_path,
#             "xml_file": xml_path,
#             "motor_pos": motor_pos,
#             "result": thick_map
#         })

#     return pd.DataFrame(results)

####################

# Example usage:
    #mask = load_mask()
    #sphere_fit_thick_map(mask)

#process_masks_with_xml()