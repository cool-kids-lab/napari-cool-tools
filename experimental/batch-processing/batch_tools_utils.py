"""
"""
from pathlib import Path
from typing import Callable

import numpy as np
from scipy.spatial import cKDTree

def scan_angle_fit_func(indices_to_map:int,bb:float=0.7669,cc:float=0.05,dd=0.0063,ee=0.0107):
    """"""
    sign = np.sign
    x = np.linspace(-1.0,1.0,indices_to_map)
    return bb*sign(x)*abs(x)**1+cc*sign(x)*abs(x)**2+dd*sign(x)*abs(x)**3+ee*sign(x)*abs(x)**4

def get_nearest_neigbors_to_target(target_point:np.ndarray,point_distribution:np.ndarray,ratio:float=0.1)->np.ndarray:
    """
    """
    tree = cKDTree(point_distribution)
    num_points = len(point_distribution[:,0])
    distances,indicies = tree.query(target_point,k=int(ratio*num_points))
    return point_distribution[indicies].squeeze() # output distribution

# File handling stuff

#### TODO Review
def get_optimal_uint_dtype(arr:np.ndarray)->np.dtype:
    """
    Determines the smallest numpy uint dtype that can store the maximum value
    in the input array.

    Args:
        arr (np.ndarray): The input NumPy array.

    Returns:
        np.dtype: The optimal unsigned integer NumPy data type.
    """
    if arr.size == 0:
        return np.uint8 # Or handle empty array case as needed

    max_val = arr.max()

    if max_val <= np.iinfo(np.uint8).max:
        return np.uint8
    elif max_val <= np.iinfo(np.uint16).max:
        return np.uint16
    elif max_val <= np.iinfo(np.uint32).max:
        return np.uint32
    elif max_val <= np.iinfo(np.uint64).max:
        return np.uint64
    else:
        # If the max_val exceeds uint64, it indicates an overflow or
        # a need for a custom handling (e.g., raise an error or use object dtype)
        raise ValueError(f"Maximum value {max_val} exceeds the capacity of np.uint64.")

####

#### TODO Review ####
def fill_ellipse(image_array, center, major_axis_len, minor_axis_len, fill_value=1):
    """
    Fills pixels within the border of an ellipse in a NumPy array.

    Args:
        image_array (np.ndarray): The 2D NumPy array representing the image.
        center (tuple): A tuple (cy, cx) representing the center coordinates of the ellipse.
        major_axis_len (float): The length of the major axis.
        minor_axis_len (float): The length of the minor axis.
        fill_value (int or float): The value to fill the pixels with.
    """
    rows, cols = image_array.shape
    cy, cx = center
    
    # Calculate semi-major and semi-minor axes
    a = major_axis_len / 2
    b = minor_axis_len / 2

    # Create a grid of coordinates
    y_coords, x_coords = np.indices(image_array.shape)

    # Translate coordinates so the center is at (0,0)
    translated_x = x_coords - cx
    translated_y = y_coords - cy

    # Apply the ellipse equation: (x^2 / a^2) + (y^2 / b^2) <= 1
    # Pixels satisfying this condition are inside or on the border of the ellipse
    mask = (translated_y**2 / a**2) + (translated_x**2 / b**2) <= 1

    # Apply the mask to the image array
    image_array[mask] = fill_value
####

def load_bits_labels(file_path:Path,key,shape_key="shape",value:int=1):
    """
    """
    npzfile = np.load(file_path)
    bits_label = npzfile[key]
    shape = npzfile[shape_key]
    label = np.unpackbits(bits_label)*value
    return label.reshape(shape)

def load_bits_labels_v2(file_path:Path,verbose:bool=False): #,required_keys:tuple[str]=["name","shape","type"]):
    """
    """
    # load npz file data and get keys
    required_keys = ("name","shape","type")
    npzfile = np.load(file_path)
    keys = tuple(npzfile.keys())

    if not all(key in keys for key in required_keys):
        raise ValueError(
            f"This file is imporperly formated it has keys {keys} but is missing one or more of the required keys {required_keys}.\n"
        )
    label_map = [key.split("_")[1] for key in keys if key not in required_keys and "bitkey" in key]
    label_keys = [key for key in keys if key not in required_keys and "bitkey" in key]
    label_values = [int(npzfile[key]) for key in keys if key not in required_keys and "value_key" in key]

    
    # print(f"label_map: {label_map}\n")
    # print(f"label_keys: {label_keys}\n")
    # print(f"label_values: {label_values}\n")

    optimal_dtype = get_optimal_uint_dtype(np.array(label_values))

    if len(label_keys) != len(label_values):
        raise ValueError(
            f"The number of bit masks ({len(label_keys)}) does not match the number of unique label values ({len(label_values)}).\n"
        )

    # Get file data
    metadata = {}
    metadata["name"] = npzfile["name"]
    metadata["properties"] = {"label_map":label_map}
    shape = (npzfile["shape"][0],npzfile["shape"][1],npzfile["shape"][2])
    file_type = npzfile["type"]
    label_data = np.zeros(shape,dtype=optimal_dtype)
    # Build label from bit data
    for key in label_map:
        bitkey = f"bitkey_{key}"
        value_key= f"value_key_{key}"
        # unpack bits reshape cast to optimal dtype multiply by stored label value add to output label data
        label_data = label_data + np.unpackbits(npzfile[bitkey]).reshape(shape).astype(optimal_dtype)*npzfile[value_key]

    if verbose:
        print(f"Loaded {file_type} data with shape: {label_data.shape} and dtype: {label_data.dtype} containing {len(label_values)} labels with values {label_values} mapped to {label_map}\nmetadata:{metadata}\n")

    return (label_data,metadata,file_type)  

def save_bits_labels(file_path:Path, label_data:np.ndarray, label_map:tuple[str]=[],verbose:bool=False):
    """
    Assume label values are positive and add to exceptions/assertions
    """

    is_integer_dtype_int_array = np.issubdtype(label_data.dtype, np.integer)

    if is_integer_dtype_int_array:

        # generate name from file path
        name = file_path.stem
        # find unique label values
        label_values = np.unique(label_data)
        non_zero_labels = label_values > 0
        if verbose:
            print(f"label values: {label_values}\nnonzero labels mask, masked values: {non_zero_labels,label_values[non_zero_labels]}\nnonzero labels: {len(label_values[non_zero_labels])}\n")
        #if len(non_zero_labels) == 0:
        #if len(non_zero_labels.nonzero()) == 0:
        if len(label_values[non_zero_labels]) == 0:
            print(f"{file_path} has no nonzero labels to store. This file will not be saved.\n")
            return 1
        label_values = label_values[non_zero_labels]

        if verbose:
            print(f"label_values: {label_values}\nlabel_map: {label_map}\n")

        if label_values.min() < 0:
            raise ValueError(f"Label data contains the value: {label_values.min()}. All labels values must be >= 0")
        if label_values.max() == 0:
            print("The label data provided is empty. File will not be saved.\n")
            return
        if len(label_map) > 0:
            if len(label_map) != len(label_values):
                raise ValueError(f"There is a mismatch between the number of labels ({len(label_values)}) and their mapped values ({len(label_map)})\n")
        # get data shape to recover when unpacking bits
        shape = label_data.shape
        # get optimal storage size for values
        optimal_storage_dtype = get_optimal_uint_dtype(label_data)
        label_values =label_values.astype(optimal_storage_dtype)

        # generate save metadata
        save_dict = {"name":name,"shape": shape, "type":"labels"}
        for idx,value in enumerate(label_values):
            #label_dict = {}
            # generate bit mask for label
            if value > 0:
                bit_mask = np.packbits(label_data == value)
                # label_dict["bits"] = bit_mask
                # label_dict["values"] = value
                if len(label_map) == 0:
                    save_dict[f"bitkey_{str(value)}"] = bit_mask
                    save_dict[f"value_key_{str(value)}"] = value
                else:
                    map_key = label_map[idx]
                    save_dict[f"bitkey_{map_key}"] = bit_mask
                    save_dict[f"value_key_{map_key}"] = value

        print(save_dict,file_path)
        print(f"Saving {len(label_values)} labels with value(s) {label_values} and shape {shape}\n")
        np.savez(file_path,**save_dict)
        return 0
    else:
        raise ValueError(f"label data must be of integer dtype: not {label_data.dtype}\n")
    
def get_motor_pos_from_xml(xml_path:Path):
    """"""
    import xml.etree.ElementTree as ET
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
        print(f"No motor_pos found for {xml_path}")
    
    return motor_pos

####
####

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

def generate_circular_mask(data_shape,scale:float=1.0,mask_depth=None,use_input_depth:bool=False,verbose:bool=False):
    """"""
    height,depth,width = data_shape
    height_center = height // 2
    width_center = width // 2 
    radius = (min(height,width) / 2) * scale

    if use_input_depth:
        mask_depth = depth

    # Create a meshgrid of coordinates
    height_at_coord,width_at_coord = np.ogrid[-height_center:height_center,-width_center:width_center]
    if verbose:
        print(f"height and width coordinate shapes: {height_at_coord.shape,width_at_coord.shape}\n")


    distance_squared = height_at_coord**2 + width_at_coord**2
    # compute circular mask
    circular_mask = distance_squared <= radius**2

    if verbose:
        print("circular mask shape",circular_mask.shape, "mask_depth value", mask_depth)

    if mask_depth:
        circular_mask = circular_mask[:,None,:]
        circular_mask = np.repeat(circular_mask,mask_depth,axis=1)

    return circular_mask

####
####
# Segmentation Cleanup Specific Code
from napari.utils.notifications import show_info
from napari_cool_tools_vol_proc._masking_tools_funcs import project_2d_mask

#### TODO review aided by llm ####
def fill_nan_nearest_neighbor_2D(arr):
    """
    Fills NaN values in a 2D array using the nearest non-NaN neighbor.
    """
    filled_arr = np.copy(arr)
    rows, cols = filled_arr.shape

    # Create a mask for NaN values
    nan_mask = np.isnan(filled_arr)

    # Iterate until no more NaNs can be filled
    while np.any(nan_mask):
        new_nan_mask = np.copy(nan_mask)

        for r in range(rows):
            for c in range(cols):
                if nan_mask[r, c]:
                    # Check 8-directional neighbors
                    neighbors = []
                    for dr in [-1, 0, 1]:
                        for dc in [-1, 0, 1]:
                            if dr == 0 and dc == 0:
                                continue  # Skip self

                            nr, nc = r + dr, c + dc
                            if (
                                0 <= nr < rows
                                and 0 <= nc < cols
                                and not np.isnan(filled_arr[nr, nc])
                            ):
                                neighbors.append(filled_arr[nr, nc])

                    if neighbors:
                        # Replace NaN with the first found non-NaN neighbor (or mean/median)
                        filled_arr[r, c] = neighbors[0]
                        new_nan_mask[r, c] = False  # Mark as filled

        # Update the NaN mask for the next iteration
        if np.array_equal(nan_mask, new_nan_mask):  # No more NaNs filled in this pass
            break
        nan_mask = new_nan_mask

    return filled_arr
#####

def generate_maps(label: np.ndarray):
    """A custom function to produce depth, thickness, and difference maps from 3D label data."""
    cylindrical_mask = generate_circular_mask(label.shape,scale=1.0,use_input_depth=True)
    label[:] = label * cylindrical_mask
    raw_depth_map = generate_depth_map(label, normalized=False)
    retina_coords, rpe_coords, thick_map = generate_thick_map(label)
    difference_map = thick_map - raw_depth_map

    return (
        raw_depth_map,
        retina_coords,
        rpe_coords,
        thick_map,
        difference_map,
    )

def get_points_centroid_along_axis(
    points: np.ndarray, points2: np.ndarray, target_axes: list[int] = [1]
):
    """ """
    if points.size == 0 or points2.size == 0:
        print("One or both of the surface points arrays are empty. No centroid points will be produced.")
        return np.array([],dtype="uint8")
    elif len(points) != len(points2):
        print(f"The number of points between the two surfaces do not match {len(points) != {len(points2)}}. No centroid points will be produced.")
        return np.array([],dtype="uint8")
    
    axes = np.arange(len(points[0]))
    out_data = []
    for ax in axes:
        if ax in target_axes:
            out_data.append((points[:, ax] + points2[:, ax]) / 2)
        else:
            out_data.append(points[:, ax])
    return np.column_stack(out_data)

def mask_3D_points_with_2D_boolean_mask(
    points_3D: np.ndarray,
    mask: np.ndarray,
    axis: int = 1,
    invert_mask: bool = False,
    verbose: bool = False,
) -> np.ndarray:
    """Filters a 3D array of points using a 2D boolean mask by projection.

    This function takes a 3D array of points and applies a 2D boolean mask to it.
    The mask is projected along the specified `axis` to generate a 3D boolean mask
    of the same shape as `points_3D`. This mask is then used to filter the
    `points_3D` array. Points where the mask is `True` are kept, while points
    where the mask is `False` are set to 0.

    Args:
        points_3D: A NumPy array of shape `(N, 3, ...)` or similar. The shape
        must conform to the specified checks, particularly that the second
        dimension must have a size of 3.
        mask: A 2D NumPy boolean array of shape `(H, W)`.
        axis: An integer specifying the dimension along which the 2D mask is
        broadcast to create the 3D mask.

    Returns:
        A NumPy array of the same shape as `points_3D` with the filtered points.
        If an error occurs, the original `points_3D` array is returned.

    Raises:
        ValueError: If the second dimension of `points_3D` is not 3.
        ValueError: If the `mask` is not a 2D array.
        ValueError: If the `mask` contains no `True` values.
        ValueError: If the total number of elements in the `mask` does not match
        the number of elements in the `points_3D` dimensions for the specified
        `axis`.
    """
    # --- Error checks ---
    if points_3D.shape[1] != 3:
        print(
            "Error: 'points_3D' must have a second dimension of size 3. "
            "Returning original array."
        )
        return points_3D

    if mask.ndim != 2:
        print("Error: 'mask' must be a 2D array. Returning original array.")
        return points_3D

    if not np.any(mask):
        print("Error: 'mask' contains no positive values. Returning original array.")
        return points_3D

    input_points = points_3D.copy().astype(int)

    # Build voxel grid from points data
    depth_min = int(input_points[:, 1].min())
    depth_max = int(input_points[:, 1].max())
    depth_shape = len(np.arange(depth_max - depth_min)) + 1  # adjust for 0 indexing

    if verbose:
        print(
            f"depth min: {depth_min}, depth max: {depth_max}, depth_shape: {depth_shape}\n"
        )

    voxel_grid = np.zeros((mask.shape[0], depth_shape, mask.shape[1])).astype(bool)
    if verbose:
        print(f"voxel grid shape: {voxel_grid.shape}\n")
    voxel_grid[
        input_points[:, 0], input_points[:, 1] - depth_min, input_points[:, 2]
    ] = 1  # offset depth points by min value to set it to zero

    # Expand mask to cover depth
    mask_3D = np.repeat(mask[:, None, :], depth_shape, axis=axis)
    if invert_mask:
        mask_3D = ~mask_3D
    voxel_grid = voxel_grid * mask_3D
    height, depth, width = voxel_grid.nonzero()
    masked_points_3D = np.column_stack((height, depth + depth_min, width))

    # return voxel_grid,masked_points_3D
    return masked_points_3D

def generate_height_map_from_3D_points(
    points_3D: np.ndarray, shape: tuple, axis=1
) -> np.ndarray:
    """
    shape is the 2D shape of the output map
    """
    # def depth_from_points_3D(points_3D:np.ndarray,shape:tuple)->np.ndarray:
    # """
    # """
    depth_out = np.zeros(shape)
    # print(depth_out.shape)
    coordinates = np.arange(len(points_3D))
    # print(points_3D[coordinates,0],points_3D[coordinates,2],points_3D[coordinates,1])
    if axis == 1:
        depth_out[points_3D[coordinates, 0], points_3D[coordinates, 2]] = points_3D[
            coordinates, 1
        ]
    elif axis == 2:
        depth_out[points_3D[coordinates, 0], points_3D[coordinates, 1]] = points_3D[
            coordinates, 2
        ]
    elif axis == 0:
        depth_out[points_3D[coordinates, 1], points_3D[coordinates, 2]] = points_3D[
            coordinates, 0
        ]
    return depth_out

def mask_components_within_theshold_of_points_distribution(
    binary_mask: np.ndarray,
    point_distirbution: np.ndarray,
    threshold: float,
    verbose: bool = False,
):
    """ """
    import cc3d
    from scipy.spatial import KDTree
    # ensure point distribution is of the proper dimenionality [num_points,dimensions/axes]

    # get connected components
    labeled_components = cc3d.connected_components(binary_mask, binary_image=True)
    # get statistics
    stats = cc3d.statistics(labeled_components)
    # get centroids
    centroids = stats["centroids"]
    # build kdtree for point_distribution
    point_distribution_kdtree = KDTree(point_distirbution)
    # find closest points for centroids
    distance_from_centroid, neighbor_index = point_distribution_kdtree.query(
        centroids, k=1, workers=-1
    )
    # filter centroids beyond the theshold distance
    indices_meet_criteria = (distance_from_centroid < threshold).nonzero()
    # build mask of components that meet the criteria
    if verbose:
        print(
            f"centroids: {centroids}\ndistance from centroid: {distance_from_centroid}\nindices that meet criteria: {indices_meet_criteria}\n"
        )
    out_mask = np.isin(labeled_components, indices_meet_criteria)
    return out_mask

def outlier_correction(points_3D: np.ndarray, outlier_map: np.ndarray) -> np.ndarray:
    """ """
    in_points = points_3D.copy()
    outlier_mask = outlier_map > 0
    outlier_coords = np.column_stack(outlier_mask.nonzero())

    # remove outlier points
    # out_points = remove_outliers_by_xz_coords(out_points,outlier_xz_coords=outlier_coords)
    outlier_free_points = mask_3D_points_with_2D_boolean_mask(
        in_points, mask=outlier_map, axis=1, invert_mask=True
    )

    # Interpolate depth values
    height_map = generate_height_map_from_3D_points(
        points_3D=points_3D, shape=outlier_mask.shape, axis=1
    )
    # print(outlier_coords,outlier_coords.shape)
    height_outlier_nans = height_map.astype(float).copy()
    # height_outlier_nans[outlier_coords] = np.nan
    height_outlier_nans[outlier_mask] = np.nan
    interpolated_nans = fill_nan_nearest_neighbor_2D(height_outlier_nans)
    new_depth_vals = interpolated_nans[outlier_mask].flatten()
    #print(new_depth_vals.shape, outlier_coords.shape)

    # create points from depth values
    interpolated_points = np.column_stack(
        (outlier_coords[:, 0], new_depth_vals, outlier_coords[:, 1])
    )
    # interpolated_points = create_3d_points_from_depth(interpolated_nans,depth_axis_index=1)
    # return height_map,height_outlier_nans,interpolated_nans,create_3d_points_from_depth(interpolated_nans,depth_axis_index=1)
    # filtered_points = filter_points_with_2d_mask(interpolated_points,mask_2d=mask)
    # out_points[outlier_coords,1] = new_depth_vals
    out_points = np.concatenate((outlier_free_points, interpolated_points))
    return out_points, outlier_free_points, interpolated_nans

def find_and_process_outliers(
    label_to_process,
    raw_depth_map: np.ndarray,
    difference_map: np.ndarray,
    ret_surf_coords: np.ndarray,
    rpe_surf_coords: np.ndarray,
    gap_threshold: float,
    pixel_thickness_threshold: float,
    component_to_central_retina_threshold: float,
):
    """ """
    # find outlier values
    show_info("Finding Outlier Values\n")
    diff = (
        difference_map > gap_threshold
    ) * 9  # mismatch outliers indicating gaps in the segmentaion
    diff2 = (
        (raw_depth_map >= pixel_thickness_threshold) & ~(diff)
    ) * 4  # mismatch from overly thick measurments 112 is slightly greater than 500 um
    outlier_map = (diff + diff2).astype(bool)
    # print(f"diff shape: {diff.shape}, label shape: {label.shape}\n")

    # Process surface outliers
    # replace outlier values in retina surface points
    ret_surf_replaced, ret_outlier_free, outlier_filled_height_map = outlier_correction(
        ret_surf_coords, outlier_map=outlier_map
    )
    ret_outliers = mask_3D_points_with_2D_boolean_mask(
        ret_surf_coords, mask=outlier_map, axis=1, invert_mask=False
    )
    rpe_outlier_free = mask_3D_points_with_2D_boolean_mask(
        rpe_surf_coords, mask=outlier_map, axis=1, invert_mask=True
    )

    # generate centroid surface for retina
    retina_surf_centroids = get_points_centroid_along_axis(
        ret_outlier_free, rpe_outlier_free, target_axes=[1]
    )
    # generate 3D outlier labels intersections
    diff_3D = project_2d_mask(label_to_process, diff)
    diff2_3D = project_2d_mask(label_to_process, diff2)
    intersect = diff_3D.astype(bool) & label_to_process.astype(bool)
    # recover label components close to the center of the retinal surface

    # centroid surface required to recover connected components near retinal surface
    if retina_surf_centroids.size != 0:
        recovered_components = mask_components_within_theshold_of_points_distribution(
            intersect,
            point_distirbution=retina_surf_centroids,
            threshold=component_to_central_retina_threshold,
        )
    else:
        recovered_components = np.zeros_like(label_to_process,dtype=bool) # return empty boolean

    intersect2 = (diff2_3D.astype(bool) & label_to_process.astype(bool)) * 4
    clean = (
        label_to_process.astype(bool)
        & ~intersect.astype(bool)
        & ~intersect2.astype(bool)
    ) + recovered_components

    return (
        clean,
        intersect2,
        recovered_components,
        intersect,
        diff2_3D,
        diff_3D,
        retina_surf_centroids,
        rpe_outlier_free,
        ret_outliers,
        ret_surf_replaced,
        ret_outlier_free,
        outlier_filled_height_map,
        outlier_map,
        diff2,
        diff,
    )

def clean_labels(
    label_data: np.ndarray,
    imaging_range: float = 12.0,
    refractive_index: float = 1.33,
    gap_threshold: int = 8,
    thickness_threshold: float = 500.0,
    incedence_allowance: float = 1 / np.sin(np.pi / 4),
    component_to_pixel_thickness_ratio: float = (1 / 3),
    dust_threshold: float = 1e6,
    viewer="",
    visualization_processing_flag:bool=False,
    verbose:bool=False,
):
    """
    """
    import napari
    import cc3d
    if not isinstance(viewer,napari.Viewer):
        show_info("Not Using Napari viewer instance.")
        viewer_flag = False
    else:
        show_info("Using Napari viewer instance.")
        viewer_flag = True

    raw_depth_map, ret_surf_coords, rpe_surf_coords, thick_map, difference_map = (
            generate_maps(label_data)
        )

    conv_factor = (
        imaging_range / label_data.shape[1] * 1000 / refractive_index
    )  # mm/pixel * um/mm /refractive index = um/pixel

    pixel_thickness_threshold = (
        int(thickness_threshold / conv_factor) * incedence_allowance
    )
    component_to_pixel_thickness_ratio = 1 / 3  # 1/2 #1/6 #1/8 #1/4 #1/2
    component_to_central_retina_threshold = (
        pixel_thickness_threshold * component_to_pixel_thickness_ratio
    )

    if verbose:
        print(
            f"Thresholdss\ngap: {gap_threshold}\nthickness theshold: {thickness_threshold}\nincedence allowance = {incedence_allowance}\npixel thicknes theshold: {pixel_thickness_threshold}\ndust threshold: {dust_threshold}\n"
        )
        print(
            f"component to pixel thickness ratio: {component_to_pixel_thickness_ratio}\ncomponent to central retina threshold: {component_to_central_retina_threshold}\n"
        )

    (
        clean,
        intersect2,
        recovered_components,
        intersect,
        diff2_3D,
        diff_3D,
        retina_surf_centroids,
        rpe_outlier_free,
        ret_outliers,
        ret_surf_replaced,
        ret_outlier_free,
        outlier_filled_height_map,
        outlier_map,
        diff2,
        diff,
    ) = find_and_process_outliers(
        label_data,
        raw_depth_map,
        difference_map,
        ret_surf_coords,
        rpe_surf_coords,
        gap_threshold=gap_threshold,
        pixel_thickness_threshold=pixel_thickness_threshold,
        component_to_central_retina_threshold=component_to_central_retina_threshold,
    )

    # process small remnants
    show_info("Processing Small Components\n")
    dust_free = cc3d.dust(clean, threshold=dust_threshold).astype("bool") * 6
    outliers = (label_data.astype("bool") & ~dust_free.astype("bool")) * 10
    outliers_coord_mask = (
        outliers.sum(axis=1).astype(bool) * 1
        - recovered_components.sum(axis=1).astype(bool) * 1
    ).astype(bool)
    num_outliers = np.count_nonzero(outliers)
    num_labeled = np.count_nonzero(label_data)
    percent_outliers = (num_outliers / num_labeled) * 100
    show_info(
        f"{num_outliers}/{num_labeled} outliers {percent_outliers}%\n"
    )

    # TODO revisit this intersect would need to be recalculated to get an accurate reading with the recovered data
    # TODO verify that outlier_coord_mask is accurate and also recalculate. Then perform thickness calculations!!
    show_info("Processing Final Cleanup\n")
    (
        raw_depth_map2,
        ret_surf_coords2,
        rpe_surf_coords2,
        thick_map2,
        difference_map2,
    ) = generate_maps(dust_free.astype("bool") * 1)

    if visualization_processing_flag:
        (
            clean2,
            intersect2_2,
            recovered_components2,
            intersect_2,
            diff2_3D2,
            diff_3D2,
            retina_surf_centroids2,
            rpe_outlier_free2,
            ret_outliers2,
            ret_surf_replaced2,
            ret_outlier_free2,
            outlier_filled_height_map2,
            outlier_map2,
            diff2_2,
            diff_2,
        ) = find_and_process_outliers(
            dust_free,
            raw_depth_map2,
            difference_map2,
            ret_surf_coords2,
            rpe_surf_coords2,
            gap_threshold=gap_threshold,
            pixel_thickness_threshold=pixel_thickness_threshold,
            component_to_central_retina_threshold=component_to_central_retina_threshold,
        )

        recovered_components2 = mask_components_within_theshold_of_points_distribution(
            intersect_2,
            point_distirbution=retina_surf_centroids2,
            threshold=component_to_central_retina_threshold,
        )

        squeaky_clean = (clean2.astype(bool) + recovered_components2) * 36

        outliers2 = (dust_free.astype("bool") & ~squeaky_clean.astype("bool")) * 10
        num_outliers2 = np.count_nonzero(outliers2)
        num_labeled2 = np.count_nonzero(dust_free)
        percent_outliers2 = (num_outliers2 / num_labeled2) * 100
        show_info(
            f"{num_outliers2}/{num_labeled2} outliers {percent_outliers2}%\n"
        )

    if viewer_flag:
        viewer.add_image(label_data, visible=False)
        viewer.add_image(raw_depth_map, visible=False)
        viewer.add_image(thick_map, visible=False)
        viewer.add_image(difference_map, visible=False)
        viewer.add_points(
            ret_surf_coords,
            size=4,
            face_color="red",
            name="retinal_surface",
            visible=False,
        )
        viewer.add_points(
            rpe_surf_coords,
            size=4,
            face_color="blue",
            name="rpe_surface",
            visible=False,
        )
        # viewer.add_points(retina_surf_centroids,size=4,face_color="magenta",name="retina_surf_centroids",visible=True)
        viewer.add_labels(diff, name="diff", visible=False)
        viewer.add_labels(diff2, name="diff2", visible=False)
        viewer.add_points(
            ret_surf_replaced,
            size=4,
            face_color="green",
            border_color="yellow",
            name="ret_surf_replaced",
            visible=True,
        )
        viewer.add_points(
            ret_outliers, size=4, face_color="red", border_color="orange", visible=True
        )
        viewer.add_points(
            ret_outlier_free,
            size=4,
            face_color="green",
            border_color="blue",
            visible=True,
        )
        viewer.add_labels(diff_3D, visible=False)
        viewer.add_labels(diff2_3D, visible=False)
        viewer.add_labels(intersect, visible=False)
        viewer.add_labels(recovered_components, visible=True)
        viewer.add_labels(intersect2, visible=False)
        viewer.add_image(outlier_filled_height_map, visible=False)
        viewer.add_labels(clean, visible=False)
        viewer.add_labels(dust_free, visible=True)
        viewer.add_labels(outliers)
        viewer.add_image(outliers_coord_mask)
        viewer.add_image(raw_depth_map2, visible=False)
        viewer.add_image(thick_map2, visible=False)
        viewer.add_image(difference_map2, visible=False)
        viewer.add_points(
            ret_surf_coords2,
            size=4,
            face_color="yellow",
            name="retinal_surface2",
            visible=True,
        )
        viewer.add_points(
            rpe_surf_coords2,
            size=4,
            face_color="purple",
            name="rpe_surface2",
            visible=True,
        )

        if visualization_processing_flag:
            viewer.add_image(outlier_filled_height_map2, visible=False)
            viewer.add_labels(diff_3D2, visible=False)
            viewer.add_labels(diff2_3D2, visible=False)
            viewer.add_labels(outliers2)
            viewer.add_labels(recovered_components2, visible=True)
            viewer.add_labels(squeaky_clean)
            viewer.add_points(
                retina_surf_centroids2,
                size=4,
                face_color="magenta",
                name="retina_surf_centroids",
                visible=True,
            )
            viewer.add_points(
                ret_surf_replaced2,
                size=4,
                face_color="green",
                border_color="yellow",
                name="ret_surf_replaced",
                visible=True,
            )

    output = (
        raw_depth_map2,
        ret_surf_coords2,
        rpe_surf_coords2,
        thick_map2,
        difference_map2,
        outliers_coord_mask,
        dust_free,
        percent_outliers,
    )
    return output

####
####

# Dr. Young Stuff 

def convert_spherical(coord):
    #coord = [theta_x, theta_y, r]
    #theta_x and theta_y are in radians units
    theta_x = coord[0]
    theta_y = coord[1]
    r = coord[2]
    theta = np.arctan2(theta_x,theta_y)
    phi = np.sqrt(theta_x**2 + theta_y**2)
    #phi = np.arctan(np.sqrt(np.tan(theta_x) + np.tan(theta_x)))
    #phi = np.arctan2(np.sqrt(np.tan(theta_x)**2 + np.tan(theta_y)**2), 1.0)  # polar

    x = r*np.sin(phi)*np.cos(theta)
    y = r*np.sin(phi)*np.sin(theta)
    z = r*np.cos(phi)
    return (x,y,z)

#def spherical_to_cartesian_old(points_3D:np.ndarray,input_shape:tuple[int],scan_angle:float,padding_pixel):
def spherical_to_cartesian(points_3D:np.ndarray,input_shape:tuple[int],scan_angle:float,padding_pixel):
        """
        """
        x = points_3D[:,0]
        y = points_3D[:,2]
        z = points_3D[:,1]
        x_center = input_shape[0]//2
        y_center = input_shape[2]//2
        p_t_x = ((x - x_center) / (input_shape[0] / 2)) * (scan_angle / 2) * (np.pi / 180)
        p_t_y = ((y - y_center) / (input_shape[2] / 2)) * (scan_angle / 2) * (np.pi / 180)
        p_r = z + padding_pixel
        #return np.array(convert_spherical([p_t_x, p_t_y, p_r])) * pixel_spacing

        spherical_tuple = convert_spherical([p_t_x,p_t_y,p_r])

        return np.column_stack((spherical_tuple[0],spherical_tuple[2],spherical_tuple[1])) #* pixel_spacing
        #return np.array((spherical_tuple[0],spherical_tuple[2],spherical_tuple[1])) #* pixel_spacing

#def spherical_to_cartesian(points_3D:np.ndarray,input_shape:tuple[int],angle_func:Callable,padding_pixel):
def spherical_to_cartesian_corrected(points_3D:np.ndarray,input_shape:tuple[int],angle_func:Callable,padding_pixel):
        """
        """
        slow_axis = points_3D[:,0]
        fast_axis = points_3D[:,2]
        axial_axis = points_3D[:,1]
        slow_center = input_shape[0]//2
        fast_center = input_shape[2]//2
        # p_t_x = ((x - x_center) / (input_shape[0] / 2)) * (scan_angle / 2) * (np.pi / 180)
        # p_t_y = ((y - y_center) / (input_shape[2] / 2)) * (scan_angle / 2) * (np.pi / 180)
        slow_axis_nonlinear_degree_map = scan_angle_fit_func(input_shape[0])
        fast_axis_nonlinear_degree_map = scan_angle_fit_func(input_shape[2])
        slow_axis_points = slow_axis_nonlinear_degree_map[slow_axis.astype(int)]
        fast_axis_points = fast_axis_nonlinear_degree_map[fast_axis.astype(int)]
        axial_points = axial_axis + padding_pixel
        #return np.array(convert_spherical([p_t_x, p_t_y, p_r])) * pixel_spacing

        #spherical_tuple = convert_spherical([p_t_x,p_t_y,p_r])
        spherical_tuple = convert_spherical([fast_axis_points,slow_axis_points,axial_points])

        return np.column_stack((spherical_tuple[0],spherical_tuple[2],spherical_tuple[1])) #* pixel_spacing
        #return np.array((spherical_tuple[0],spherical_tuple[2],spherical_tuple[1])) #* pixel_spacing

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

def generate_thick_map(mask):
    """
    """

    slow_shape, axial_shape, fast_shape = mask.shape
    # get retinal and RPE surfaces
    rpe_coords = np.array(RPE_layer_coords(mask))  # Shape: (N, 3)
    retina_coords = np.array(Retinal_surface_coords(mask))  # Shape: (M, 3)

    retina_axial = retina_coords[:,1]
    rpe_axial = rpe_coords[:,1]
    thickness = rpe_axial - retina_axial

    # # find closest point on RPE to retinal surface
    # rpe_depth = rpe_coords[:,1].nonzero()[0]
    # retina_depth = retina_coords[:,1].nonzero()[0]
    # print(rpe_coords.shape)
    # print(rpe_depth,rpe_depth.shape)
    # surface_distance = rpe_depth - retina_depth
    # print(surface_distance,surface_distance.shape)

    # # create thickness map
    thick_map = np.zeros((slow_shape,fast_shape))
    thick_map[rpe_coords[:,0],rpe_coords[:,2]] = thickness
    # thick_map[retina_depth] = 1
    # thick_map[rpe_depth] = 2
    #thick_map[np.arange(slow_shape), np.arange(fast_shape)] = surface_distance

    return retina_coords,rpe_coords,thick_map


def sphere_fit_thick_map(mask, refractive_index=1.33, imaging_motor_position=0.0, reference_motor_position=85.0, imaging_range=12, pivot_point=19.2, scan_angle=105,ret_to_rpe:bool=True,micron_output:bool=True,debug:bool=False):
    """
    Optimized function to fit a sphere to the retinal surface and compute z-difference.
    Returns Cartesian coordinates and color values for visualization.
    """
    
    y_shape, z_shape, x_shape = mask.shape
    if debug:
        print(f"retchor_mask shape: {mask.shape}\n")
    #x_shift, y_shift = [0,0]
    imaging_range /= refractive_index
    pixel_spacing = imaging_range / z_shape
    center = np.array([x_shape // 2 , y_shape // 2])

    # Adjust reference arm shift
    #reference_arm_shift = reference_motor_position - imaging_motor_position
    reference_arm_shift = reference_motor_position - imaging_motor_position
    reference_arm_shift = (reference_arm_shift * 0.5) / refractive_index
    #reference_arm_shift = reference_arm_shift / refractive_index
    if debug:
        print(f"reference_arm_shift: {reference_arm_shift}\n")
    #padding = pivot_point - imaging_range - reference_arm_shift
    padding = pivot_point - (imaging_range + reference_arm_shift) # this is from fixing Dr. Young curve correction to match Yakub curve correction
    
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

    if debug:
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

    if debug:
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
        #curve_correct_height, _ = tree.query(curv_rpe_coords, k=1) #switch to avoid mesuring distance to base of curve
        
        curve_correct_height, _ = tree.query(curv_ret_coords, k=1)
    else:
        #curve_correct_height, _ = tree.query(cart_coords, k=1)
        curve_correct_height, _ = tree.query(curv_ret_coords, k=1)

    #convert from mm to pixels
    #curve_correct_height = curve_correct_height /((6/(mask.shape[1]*1.33))) # I believe the magic number 6 needs to be the imaging range and 1.33 is the index of refraction of water.
    
    #curve_correct_height = curve_correct_height /((imaging_range/(mask.shape[1]*refractive_index)))

    if micron_output:
        conv_factor = pixel_spacing * 1000 / refractive_index # mm/pixel * 1000 um/mm / refractive index = um/pixel
    else:
        conv_factor = 1.0

    
    raw_pixel_thickness_map = np.full((y_shape, x_shape), 0.0)
    pixel_thickness_map = np.full((y_shape, x_shape), 0.0)
    raw_pixel_thickness_map[rpe_y,rpe_x] = standard_height
    pixel_thickness_map[rpe_y,rpe_x] = curve_correct_height
    curve_correct_height = curve_correct_height * conv_factor

    #z_diffs = rpe_z-retina_z
    thickness_map = np.full((y_shape, x_shape), 0.0) #np.nan)  # use NaN for missing pixels
    
    # Fill thickness map at (y, x) positions
    thickness_map[rpe_y, rpe_x] = curve_correct_height

    #return thickness_map
    #return thickness_map, cart_coords, rpe_coords
    return thickness_map, retina_coords, rpe_coords, curv_ret_coords, curv_rpe_coords, raw_pixel_thickness_map, pixel_thickness_map

def sphere_fit_thick_map_corrected(mask, refractive_index=1.33, imaging_motor_position=0.0, reference_motor_position=85.0, imaging_range=12, pivot_point=19.2, scan_angle=105,ret_to_rpe:bool=True,micron_output:bool=True,debug:bool=False):
    """
    Optimized function to fit a sphere to the retinal surface and compute z-difference.
    Returns Cartesian coordinates and color values for visualization.
    """
    
    y_shape, z_shape, x_shape = mask.shape
    if debug:
        print(f"retchor_mask shape: {mask.shape}\n")

    imaging_range /= refractive_index
    pixel_spacing = imaging_range / z_shape
    center = np.array([x_shape // 2 , y_shape // 2])

    # Adjust reference arm shift
    reference_arm_shift = reference_motor_position - imaging_motor_position
    reference_arm_shift = (reference_arm_shift * 0.5) / refractive_index

    if debug:
        print(f"reference_arm_shift: {reference_arm_shift}\n")

    padding = pivot_point - (imaging_range + reference_arm_shift) # this is from fixing Dr. Young curve correction to match Yakub curve correction
    
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


    if debug:
        print(f"retina_coords len: {len(retina_coords)}\n\nrpe_coords len: {len(rpe_coords)}\n")

    # Convert 3D coordinates to 2D indices
    rpe_y, rpe_z, rpe_x = rpe_coords.T
    retina_y, retina_z, retina_x = retina_coords.T
    standard_height = rpe_z - retina_z
    
    # Vectorized spherical to Cartesian conversion
    # def spherical_to_cartesian(y, z, x):
    #     p_t_x = ((x - center[0]) / (x_shape / 2)) * (scan_angle / 2) * (np.pi / 180)
    #     p_t_y = ((y - center[1]) / (y_shape / 2)) * (scan_angle / 2) * (np.pi / 180)
    #     p_r = z + padding_pixel
    #     #return np.array(convert_spherical([p_t_x, p_t_y, p_r])) * pixel_spacing

    #     spherical_tuple = convert_spherical([p_t_x,p_t_y,p_r])

    #     return np.array((spherical_tuple[0],spherical_tuple[2],spherical_tuple[1])) #* pixel_spacing

    # Apply spherical-to-Cartesian conversion vectorized

    # curv_ret_coords= np.array([spherical_to_cartesian(y, z, x) for y, z, x in retina_coords])
    # curv_rpe_coords = np.array([spherical_to_cartesian(y, z, x) for y, z, x in rpe_coords])

    curv_ret_coords = spherical_to_cartesian_corrected(retina_coords,input_shape=mask.shape,angle_func=scan_angle_fit_func,padding_pixel=padding_pixel)
    curv_rpe_coords = spherical_to_cartesian_corrected(rpe_coords,input_shape=mask.shape,angle_func=scan_angle_fit_func,padding_pixel=padding_pixel)

    if debug:
        print(f"retina_coords len: {len(curv_ret_coords)}\n\nrpe_coords len: {len(curv_rpe_coords)}\n")

    if ret_to_rpe:
        tree = cKDTree(curv_rpe_coords)
    else:
        tree = cKDTree(curv_ret_coords)
    
    # Query nearest neighbor for each inner point

    if ret_to_rpe:
        curve_correct_height, _ = tree.query(curv_ret_coords, k=1)
    else:
        curve_correct_height, _ = tree.query(curv_ret_coords, k=1)

    #convert from mm to pixels

    if micron_output:
        conv_factor = pixel_spacing * 1000 / refractive_index # mm/pixel * 1000 um/mm / refractive index = um/pixel
    else:
        conv_factor = 1.0

    
    raw_pixel_thickness_map = np.full((y_shape, x_shape), 0.0)
    pixel_thickness_map = np.full((y_shape, x_shape), 0.0)
    raw_pixel_thickness_map[rpe_y,rpe_x] = standard_height
    pixel_thickness_map[rpe_y,rpe_x] = curve_correct_height
    curve_correct_height = curve_correct_height * conv_factor

    #z_diffs = rpe_z-retina_z
    thickness_map = np.full((y_shape, x_shape), 0.0) #np.nan)  # use NaN for missing pixels
    
    # Fill thickness map at (y, x) positions
    thickness_map[rpe_y, rpe_x] = curve_correct_height

    #return thickness_map
    #return thickness_map, cart_coords, rpe_coords
    return thickness_map, retina_coords, rpe_coords, curv_ret_coords, curv_rpe_coords, raw_pixel_thickness_map, pixel_thickness_map


def sphere_fit_thick_map_corrected_v2(mask,pixel_spacing:float,padding_pixel:float,refractive_index:float=1.33,ret_to_rpe:bool=True,micron_output:bool=True,debug:bool=False):
    """
    Optimized function to fit a sphere to the retinal surface and compute z-difference.
    Returns Cartesian coordinates and color values for visualization.
    """
    
    y_shape, z_shape, x_shape = mask.shape
    if debug:
        print(f"retchor_mask shape: {mask.shape}\n")

    # Extract layer coordinates
    #note to self, this essentially calls valid_coordinates three separate times; for efficiency can probably condense
    rpe_coords = np.array(RPE_layer_coords(mask))  # Shape: (N, 3)
    retina_coords = np.array(Retinal_surface_coords(mask))  # Shape: (M, 3)

    if debug:
        print(f"retina_coords len: {len(retina_coords)}\n\nrpe_coords len: {len(rpe_coords)}\n")

    # Convert 3D coordinates to 2D indices
    rpe_y, rpe_z, rpe_x = rpe_coords.T
    retina_y, retina_z, retina_x = retina_coords.T
    standard_height = rpe_z - retina_z

    curv_ret_coords = spherical_to_cartesian_corrected(retina_coords,input_shape=mask.shape,angle_func=scan_angle_fit_func,padding_pixel=padding_pixel)
    curv_rpe_coords = spherical_to_cartesian_corrected(rpe_coords,input_shape=mask.shape,angle_func=scan_angle_fit_func,padding_pixel=padding_pixel)

    if debug:
        print(f"retina_coords len: {len(curv_ret_coords)}\n\nrpe_coords len: {len(curv_rpe_coords)}\n")

    if ret_to_rpe:
        tree = cKDTree(curv_rpe_coords)
    else:
        tree = cKDTree(curv_ret_coords)
    
    # Query nearest neighbor for each inner point

    if ret_to_rpe:
        curve_correct_height, _ = tree.query(curv_ret_coords, k=1)
    else:
        curve_correct_height, _ = tree.query(curv_ret_coords, k=1)

    #convert from mm to pixels

    if micron_output:
        conv_factor = pixel_spacing * 1000 / refractive_index # mm/pixel * 1000 um/mm / refractive index = um/pixel
        #conv_factor = pixel_spacing * 1000 # mm/pixel * 1000 um/mm = um/pixel
    else:
        conv_factor = 1.0

    
    raw_pixel_thickness_map = np.full((y_shape, x_shape), 0.0)
    pixel_thickness_map = np.full((y_shape, x_shape), 0.0)
    raw_pixel_thickness_map[rpe_y,rpe_x] = standard_height
    pixel_thickness_map[rpe_y,rpe_x] = curve_correct_height
    curve_correct_height = curve_correct_height * conv_factor

    #z_diffs = rpe_z-retina_z
    thickness_map = np.full((y_shape, x_shape), 0.0) #np.nan)  # use NaN for missing pixels
    
    # Fill thickness map at (y, x) positions
    thickness_map[rpe_y, rpe_x] = curve_correct_height

    #return thickness_map
    #return thickness_map, cart_coords, rpe_coords
    return thickness_map, retina_coords, rpe_coords, curv_ret_coords, curv_rpe_coords, raw_pixel_thickness_map, pixel_thickness_map